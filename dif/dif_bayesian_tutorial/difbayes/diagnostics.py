"""사후 진단(posterior diagnostics) 유틸리티.

ArviZ가 설치되어 있으면 그것을 우선 사용하고, 없으면 간단한 자체 구현을 사용한다.
"""

from __future__ import annotations
from typing import Optional
import numpy as np
import pandas as pd

try:
    import arviz as az
    _HAS_ARVIZ = True
except ImportError:
    _HAS_ARVIZ = False


# ---------------------------------------------------------------------------
# Effective Sample Size (간이 구현)
# ---------------------------------------------------------------------------
def _ess_simple(chains: np.ndarray) -> float:
    """간이 ESS — chains: (n_chains, n_draws). autocorr 합 기반."""
    chains = np.asarray(chains, dtype=float)
    n_chains, n_draws = chains.shape
    # 평균 0 으로 중심화
    x = chains - chains.mean(axis=1, keepdims=True)
    # 모든 체인 결합
    flat = x.flatten()
    n = flat.size
    # FFT autocorrelation
    f = np.fft.fft(np.concatenate([flat, np.zeros_like(flat)]))
    acf = np.fft.ifft(f * np.conj(f)).real[:n]
    acf = acf / acf[0]
    # 합 (음수가 처음 나오는 곳까지)
    cutoff = np.where(acf < 0.05)[0]
    k = cutoff[0] if len(cutoff) > 0 else min(50, n // 4)
    rho_sum = 1 + 2 * acf[1:k].sum()
    if rho_sum <= 0:
        rho_sum = 1.0
    return n / rho_sum


def _rhat_simple(chains: np.ndarray) -> float:
    """간이 R-hat — chains: (n_chains, n_draws)."""
    chains = np.asarray(chains, dtype=float)
    m, n = chains.shape
    if m < 2:
        return np.nan
    chain_means = chains.mean(axis=1)
    chain_vars = chains.var(axis=1, ddof=1)
    grand_mean = chain_means.mean()
    B = n * np.var(chain_means, ddof=1)
    W = chain_vars.mean()
    var_hat = (n - 1) / n * W + B / n
    return float(np.sqrt(var_hat / W)) if W > 0 else np.nan


# ---------------------------------------------------------------------------
# 통합 진단 보고서
# ---------------------------------------------------------------------------
def summarize_posterior(samples: dict, hdi_prob: float = 0.95) -> pd.DataFrame:
    """모수별 사후 요약 표.

    Parameters
    ----------
    samples : dict[str, ndarray]
        키 = 모수명, 값 = (n_chains, n_draws) 또는 (n_chains, n_draws, dim).
    """
    rows = []
    alpha = (1 - hdi_prob) / 2
    for name, arr in samples.items():
        arr = np.asarray(arr)
        if arr.ndim == 2:
            mean = arr.mean()
            sd = arr.std()
            lo = np.quantile(arr, alpha)
            hi = np.quantile(arr, 1 - alpha)
            ess = _ess_simple(arr)
            rhat = _rhat_simple(arr)
            rows.append(dict(parameter=name, mean=mean, sd=sd,
                             lo=lo, hi=hi, ess=ess, rhat=rhat))
        elif arr.ndim == 3:
            n_chains, n_draws, D = arr.shape
            for d in range(D):
                slab = arr[:, :, d]
                mean = slab.mean()
                sd = slab.std()
                lo = np.quantile(slab, alpha)
                hi = np.quantile(slab, 1 - alpha)
                ess = _ess_simple(slab)
                rhat = _rhat_simple(slab)
                rows.append(dict(parameter=f"{name}[{d}]",
                                 mean=mean, sd=sd, lo=lo, hi=hi,
                                 ess=ess, rhat=rhat))
    df = pd.DataFrame(rows)
    df = df.rename(columns={"lo": f"{int(hdi_prob*100)}%_lo",
                            "hi": f"{int(hdi_prob*100)}%_hi"})
    return df


def posterior_prob_above_threshold(samples: np.ndarray, threshold: float = 0.0,
                                   direction: str = "two-sided") -> float:
    """사후확률 P(|Δ| > threshold | data) 또는 P(Δ > threshold | data)."""
    s = np.asarray(samples).ravel()
    if direction == "two-sided":
        return float(np.mean(np.abs(s) > threshold))
    elif direction == "greater":
        return float(np.mean(s > threshold))
    elif direction == "less":
        return float(np.mean(s < threshold))
    raise ValueError("direction must be 'two-sided'|'greater'|'less'")


def rope_probability(samples: np.ndarray, rope=(-0.2, 0.2)) -> dict:
    """ROPE(Region of Practical Equivalence) 기반 판정에 쓰이는 확률들."""
    s = np.asarray(samples).ravel()
    in_rope = float(np.mean((s >= rope[0]) & (s <= rope[1])))
    above = float(np.mean(s > rope[1]))
    below = float(np.mean(s < rope[0]))
    return dict(in_rope=in_rope, above=above, below=below)
