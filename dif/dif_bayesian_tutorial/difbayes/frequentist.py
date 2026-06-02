"""빈도주의 DIF 검정 — 베이지안 결과와의 비교 기준선.

- Mantel-Haenszel (MH) 통계량 + Δ_MH 효과크기
- 로지스틱 회귀 DIF 검정 (uniform + non-uniform)

scipy/scikit-learn에 가벼운 의존만 두며, 외부 IRT 패키지는 사용하지 않는다.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import numpy as np
from scipy import stats


# ---------------------------------------------------------------------------
# Mantel-Haenszel DIF
# ---------------------------------------------------------------------------
@dataclass
class MHResult:
    item: int
    alpha_mh: float       # MH common odds ratio
    delta_mh: float       # ETS-style effect size = -2.35 * log(alpha_mh)
    chi2: float
    pvalue: float
    ets_class: str        # "A", "B", "C"
    n_strata_used: int


def _ets_classify(delta_mh: float, pvalue: float) -> str:
    """ETS A/B/C 분류 (Holland & Thayer).

    - A: |Δ_MH| < 1.0 또는 비유의
    - B: 1.0 ≤ |Δ_MH| < 1.5 이고 유의
    - C: |Δ_MH| ≥ 1.5 이고 유의
    """
    if pvalue >= 0.05 or abs(delta_mh) < 1.0:
        return "A"
    if abs(delta_mh) < 1.5:
        return "B"
    return "C"


def mantel_haenszel(
    Y: np.ndarray,
    group: np.ndarray,
    n_strata: int = 5,
    item: int = 0,
    matching_score: Optional[np.ndarray] = None,
) -> MHResult:
    """단일 문항에 대한 MH DIF 통계.

    매칭 점수(matching score)는 기본적으로 "해당 문항을 제외한 총점"
    (rest-score)을 사용하며, 이를 동일 크기 분위(stratum)로 나눈다.

    Parameters
    ----------
    Y : (N, J) int 0/1
    group : (N,) 0/1
    n_strata : int
        능력 매칭 분위 수.
    item : int
        검정할 문항 index.
    matching_score : (N,) array, optional
        외부에서 제공할 매칭 점수. None이면 rest-score 사용.
    """
    Y = np.asarray(Y)
    group = np.asarray(group)
    if matching_score is None:
        # rest-score: 해당 문항을 뺀 총점
        matching_score = Y.sum(axis=1) - Y[:, item]
    matching_score = np.asarray(matching_score)

    # 분위 나누기
    quantiles = np.quantile(matching_score, np.linspace(0, 1, n_strata + 1))
    quantiles[0] -= 1e-9
    strata = np.digitize(matching_score, quantiles[1:-1])

    # 2x2 분할표 만들어서 MH 통계 계산
    num_alpha = 0.0
    den_alpha = 0.0
    e_sum = 0.0
    v_sum = 0.0
    o_sum = 0.0
    n_used = 0
    for s in np.unique(strata):
        idx = strata == s
        a = ((group[idx] == 0) & (Y[idx, item] == 1)).sum()
        b = ((group[idx] == 0) & (Y[idx, item] == 0)).sum()
        c = ((group[idx] == 1) & (Y[idx, item] == 1)).sum()
        d = ((group[idx] == 1) & (Y[idx, item] == 0)).sum()
        n_s = a + b + c + d
        if n_s == 0 or (a + c) == 0 or (b + d) == 0:
            continue
        n_used += 1
        num_alpha += a * d / n_s
        den_alpha += b * c / n_s

        # MH chi-square 분자/분모 누적
        n1s = a + b   # reference
        m1s = a + c   # success
        E = n1s * m1s / n_s
        V = (n1s * (n_s - n1s) * m1s * (n_s - m1s)) / (n_s * n_s * (n_s - 1)) if n_s > 1 else 0.0
        e_sum += E
        v_sum += V
        o_sum += a

    alpha_mh = (num_alpha / den_alpha) if den_alpha > 0 else np.nan
    delta_mh = -2.35 * np.log(alpha_mh) if alpha_mh > 0 else np.nan
    chi2 = ((abs(o_sum - e_sum) - 0.5) ** 2 / v_sum) if v_sum > 0 else 0.0
    pvalue = 1 - stats.chi2.cdf(chi2, df=1)
    ets = _ets_classify(delta_mh, pvalue)

    return MHResult(
        item=item, alpha_mh=alpha_mh, delta_mh=delta_mh,
        chi2=chi2, pvalue=pvalue, ets_class=ets, n_strata_used=n_used,
    )


def mantel_haenszel_all(Y, group, n_strata=5, matching_score=None):
    """모든 문항에 대한 MH 결과 리스트."""
    Y = np.asarray(Y); group = np.asarray(group)
    J = Y.shape[1]
    return [mantel_haenszel(Y, group, n_strata=n_strata, item=j,
                            matching_score=matching_score) for j in range(J)]


# ---------------------------------------------------------------------------
# 로지스틱 회귀 DIF (Swaminathan & Rogers, 1990)
# ---------------------------------------------------------------------------
@dataclass
class LogRegDIFResult:
    item: int
    beta_group: float
    beta_interaction: float
    pvalue_uniform: float
    pvalue_nonuniform: float


def _logreg_irls(X, y, max_iter=100, tol=1e-6):
    """간단한 IRLS 로지스틱 회귀. (외부 의존 회피)
    반환: (beta, se)"""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    n, p = X.shape
    beta = np.zeros(p)
    for _ in range(max_iter):
        eta = X @ beta
        mu = 1.0 / (1.0 + np.exp(-eta))
        W = mu * (1 - mu) + 1e-9
        z = eta + (y - mu) / W
        # WLS
        XtWX = (X.T * W) @ X
        XtWz = (X.T * W) @ z
        try:
            beta_new = np.linalg.solve(XtWX + 1e-8 * np.eye(p), XtWz)
        except np.linalg.LinAlgError:
            break
        if np.max(np.abs(beta_new - beta)) < tol:
            beta = beta_new
            break
        beta = beta_new
    # 표준오차
    eta = X @ beta
    mu = 1.0 / (1.0 + np.exp(-eta))
    W = mu * (1 - mu) + 1e-9
    XtWX = (X.T * W) @ X
    try:
        cov = np.linalg.inv(XtWX + 1e-8 * np.eye(p))
        se = np.sqrt(np.maximum(np.diag(cov), 0))
    except np.linalg.LinAlgError:
        se = np.full(p, np.nan)
    return beta, se


def logreg_dif(Y, group, item=0, matching_score=None) -> LogRegDIFResult:
    """단일 문항에 대한 로지스틱 회귀 DIF 검정.

    모형: logit(P(Y_ij=1)) = β0 + β1 * score + β2 * group + β3 * score * group
    - β2 검정 → uniform DIF
    - β3 검정 → non-uniform DIF
    """
    Y = np.asarray(Y); group = np.asarray(group)
    if matching_score is None:
        matching_score = Y.sum(axis=1) - Y[:, item]
    score = np.asarray(matching_score, dtype=float)
    score = (score - score.mean()) / (score.std() + 1e-9)
    y = Y[:, item]

    X = np.column_stack([np.ones_like(score), score, group, score * group])
    beta, se = _logreg_irls(X, y)
    z_uniform = beta[2] / (se[2] + 1e-12)
    z_nonuniform = beta[3] / (se[3] + 1e-12)
    p_uniform = 2 * (1 - stats.norm.cdf(abs(z_uniform)))
    p_nonuniform = 2 * (1 - stats.norm.cdf(abs(z_nonuniform)))
    return LogRegDIFResult(
        item=item,
        beta_group=beta[2],
        beta_interaction=beta[3],
        pvalue_uniform=p_uniform,
        pvalue_nonuniform=p_nonuniform,
    )


def logreg_dif_all(Y, group, matching_score=None):
    J = np.asarray(Y).shape[1]
    return [logreg_dif(Y, group, item=j, matching_score=matching_score) for j in range(J)]
