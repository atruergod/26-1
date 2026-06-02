"""Rasch 1PL / 2PL + DIF 응답 자료 시뮬레이터.

설계 원칙
---------
- 자료생성과정(data-generating process, DGP)을 명시적으로 통제하여
  학습자가 "정답"을 알고 추정 결과와 비교할 수 있게 한다.
- DIF 유형(uniform / non-uniform)을 깔끔히 구분한다.
- 두 집단(group): 0 = reference, 1 = focal.

표기 (notation)
---------------
- θ_i  : 사람 i의 능력(ability/latent trait)
- b_j  : 문항 j의 난이도(difficulty), reference group 기준
- a_j  : 문항 j의 변별도(discrimination), 2PL에서만
- Δb_j : focal group에서 b_j 에 더해지는 DIF 효과 (uniform DIF)
- Δa_j : focal group에서 a_j 에 곱해지는 효과 (non-uniform DIF, 2PL)
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import numpy as np


# ---------------------------------------------------------------------------
# 자료 객체 (data container)
# ---------------------------------------------------------------------------
@dataclass
class DIFData:
    """시뮬레이션 결과 컨테이너.

    Attributes
    ----------
    Y : (N, J) ndarray of int (0/1)
        응답 행렬 (response matrix).
    group : (N,) ndarray of int (0=ref, 1=focal)
    theta : (N,) ndarray of float
        진짜 능력 (true latent ability) — 학습 목적상 노출.
    b_true : (J,) ndarray of float
        참조집단 기준 진짜 난이도.
    delta_b_true : (J,) ndarray of float
        진짜 uniform DIF 효과 (focal − reference).
    a_true : Optional[(J,) ndarray]
        2PL일 때 진짜 변별도, 1PL이면 None.
    delta_a_true : Optional[(J,) ndarray]
        2PL에서 focal group 변별도 변화 (multiplicative), 1PL이면 None.
    meta : dict
        시뮬레이션 메타데이터 (난수 시드, 집단 크기 등).
    """

    Y: np.ndarray
    group: np.ndarray
    theta: np.ndarray
    b_true: np.ndarray
    delta_b_true: np.ndarray
    a_true: Optional[np.ndarray] = None
    delta_a_true: Optional[np.ndarray] = None
    meta: dict = field(default_factory=dict)

    @property
    def N(self) -> int:
        return self.Y.shape[0]

    @property
    def J(self) -> int:
        return self.Y.shape[1]

    @property
    def n_ref(self) -> int:
        return int((self.group == 0).sum())

    @property
    def n_focal(self) -> int:
        return int((self.group == 1).sum())


# ---------------------------------------------------------------------------
# 기본 1PL DGP
# ---------------------------------------------------------------------------
def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def simulate_rasch_dif(
    n_ref: int = 300,
    n_focal: int = 300,
    b_true: Optional[np.ndarray] = None,
    delta_b_true: Optional[np.ndarray] = None,
    mu_ref: float = 0.0,
    mu_focal: float = 0.0,
    sd_theta: float = 1.0,
    seed: int = 2026,
) -> DIFData:
    """Rasch 1PL + uniform DIF 응답 자료 생성.

    응답확률:
        P(Y_ij = 1 | θ_i, b_j, group_i) = sigmoid(θ_i − (b_j + g_i * Δb_j))
    여기서 g_i = 1(focal), 0(reference).

    Parameters
    ----------
    n_ref, n_focal : int
        각 집단 표본 크기.
    b_true : (J,) array, optional
        참조집단 기준 진짜 난이도. None이면 J=10의 -2..2 등간 사용.
    delta_b_true : (J,) array, optional
        진짜 DIF 효과. None이면 모두 0 (DIF 없음).
    mu_ref, mu_focal : float
        두 집단 능력 평균. 차이는 "impact"로 해석됨.
    sd_theta : float
        능력 표준편차.
    seed : int
        난수 시드.

    Returns
    -------
    DIFData
    """
    rng = np.random.default_rng(seed)

    # 기본 문항 난이도: 10개 문항, -2 ~ 2 등간
    if b_true is None:
        b_true = np.linspace(-2.0, 2.0, 10)
    b_true = np.asarray(b_true, dtype=float)
    J = b_true.shape[0]

    if delta_b_true is None:
        delta_b_true = np.zeros(J)
    delta_b_true = np.asarray(delta_b_true, dtype=float)

    # 능력 표집
    theta_ref = rng.normal(mu_ref, sd_theta, size=n_ref)
    theta_focal = rng.normal(mu_focal, sd_theta, size=n_focal)
    theta = np.concatenate([theta_ref, theta_focal])
    group = np.concatenate([np.zeros(n_ref, dtype=int), np.ones(n_focal, dtype=int)])

    # 집단별 난이도 = b + g * Δb
    b_eff = b_true[None, :] + group[:, None] * delta_b_true[None, :]
    # 응답확률
    logits = theta[:, None] - b_eff
    P = _sigmoid(logits)
    Y = (rng.uniform(size=P.shape) < P).astype(int)

    return DIFData(
        Y=Y,
        group=group,
        theta=theta,
        b_true=b_true,
        delta_b_true=delta_b_true,
        a_true=None,
        delta_a_true=None,
        meta=dict(
            model="1PL",
            n_ref=n_ref,
            n_focal=n_focal,
            mu_ref=mu_ref,
            mu_focal=mu_focal,
            sd_theta=sd_theta,
            seed=seed,
        ),
    )


# ---------------------------------------------------------------------------
# 2PL DGP (uniform + non-uniform DIF)
# ---------------------------------------------------------------------------
def simulate_2pl_dif(
    n_ref: int = 300,
    n_focal: int = 300,
    a_true: Optional[np.ndarray] = None,
    b_true: Optional[np.ndarray] = None,
    delta_a_true: Optional[np.ndarray] = None,
    delta_b_true: Optional[np.ndarray] = None,
    mu_ref: float = 0.0,
    mu_focal: float = 0.0,
    sd_theta: float = 1.0,
    seed: int = 2026,
) -> DIFData:
    """2PL + uniform/non-uniform DIF 응답 자료 생성.

    응답확률 (focal group의 변별도는 a_j * Δa_j 형태로 곱셈적 변화):
        a_eff = a_j * exp(g_i * log(Δa_j))     # log-multiplicative
        b_eff = b_j + g_i * Δb_j
        P(Y_ij = 1) = sigmoid(a_eff * (θ_i − b_eff))

    Parameters
    ----------
    a_true : (J,) array, optional, default = ones (Rasch-like)
    b_true : (J,) array, optional, default = -2..2 등간 10문항
    delta_a_true : (J,) array, optional, default = ones (non-uniform DIF 없음)
    delta_b_true : (J,) array, optional, default = zeros (uniform DIF 없음)
    """
    rng = np.random.default_rng(seed)

    if b_true is None:
        b_true = np.linspace(-2.0, 2.0, 10)
    b_true = np.asarray(b_true, dtype=float)
    J = b_true.shape[0]

    if a_true is None:
        a_true = np.ones(J)
    a_true = np.asarray(a_true, dtype=float)

    if delta_b_true is None:
        delta_b_true = np.zeros(J)
    delta_b_true = np.asarray(delta_b_true, dtype=float)

    if delta_a_true is None:
        delta_a_true = np.ones(J)   # ratio = 1 means no non-uniform DIF
    delta_a_true = np.asarray(delta_a_true, dtype=float)

    theta_ref = rng.normal(mu_ref, sd_theta, size=n_ref)
    theta_focal = rng.normal(mu_focal, sd_theta, size=n_focal)
    theta = np.concatenate([theta_ref, theta_focal])
    group = np.concatenate([np.zeros(n_ref, dtype=int), np.ones(n_focal, dtype=int)])

    a_eff = a_true[None, :] * (delta_a_true[None, :] ** group[:, None])
    b_eff = b_true[None, :] + group[:, None] * delta_b_true[None, :]
    logits = a_eff * (theta[:, None] - b_eff)
    P = _sigmoid(logits)
    Y = (rng.uniform(size=P.shape) < P).astype(int)

    return DIFData(
        Y=Y,
        group=group,
        theta=theta,
        b_true=b_true,
        delta_b_true=delta_b_true,
        a_true=a_true,
        delta_a_true=delta_a_true,
        meta=dict(
            model="2PL",
            n_ref=n_ref,
            n_focal=n_focal,
            mu_ref=mu_ref,
            mu_focal=mu_focal,
            sd_theta=sd_theta,
            seed=seed,
        ),
    )


# ---------------------------------------------------------------------------
# 사전 정의 시나리오 (튜토리얼용 편의 함수)
# ---------------------------------------------------------------------------
def scenario_intro_10items(seed: int = 2026) -> DIFData:
    """Notebook 00용 기본 시나리오.

    - 10문항, 두 집단 각 300명
    - 문항 5(index=4): 강한 uniform DIF (Δb = +0.8, focal에 불리)
    - 문항 8(index=7): 약한 uniform DIF (Δb = -0.4, focal에 유리)
    - 나머지 8문항: DIF 없음
    """
    b_true = np.linspace(-2.0, 2.0, 10)
    delta_b_true = np.zeros(10)
    delta_b_true[4] = 0.8
    delta_b_true[7] = -0.4
    return simulate_rasch_dif(
        n_ref=300,
        n_focal=300,
        b_true=b_true,
        delta_b_true=delta_b_true,
        seed=seed,
    )


def scenario_sparse_dif_10items(
    n_ref: int = 300,
    n_focal: int = 300,
    seed: int = 2026,
) -> DIFData:
    """Notebook 03/04용: 10문항 중 2개만 진짜 DIF."""
    b_true = np.linspace(-2.0, 2.0, 10)
    delta_b_true = np.zeros(10)
    delta_b_true[2] = 0.7
    delta_b_true[7] = -0.6
    return simulate_rasch_dif(
        n_ref=n_ref,
        n_focal=n_focal,
        b_true=b_true,
        delta_b_true=delta_b_true,
        seed=seed,
    )


def scenario_2pl_nonuniform(seed: int = 2026) -> DIFData:
    """부록 A용 2PL 시나리오.

    - 10문항, 두 집단 각 400명
    - 문항 3 (idx=2): uniform DIF (Δb=+0.6)
    - 문항 6 (idx=5): non-uniform DIF (Δa=1.6 → focal에서 변별도 1.6배)
    - 문항 9 (idx=8): 두 가지 모두 (Δb=-0.4, Δa=0.7)
    """
    b_true = np.linspace(-2.0, 2.0, 10)
    a_true = np.full(10, 1.0)
    delta_b_true = np.zeros(10)
    delta_a_true = np.ones(10)
    delta_b_true[2] = 0.6
    delta_a_true[5] = 1.6
    delta_b_true[8] = -0.4
    delta_a_true[8] = 0.7
    return simulate_2pl_dif(
        n_ref=400,
        n_focal=400,
        a_true=a_true,
        b_true=b_true,
        delta_a_true=delta_a_true,
        delta_b_true=delta_b_true,
        seed=seed,
    )
