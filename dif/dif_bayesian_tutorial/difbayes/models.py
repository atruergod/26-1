"""백엔드(stan / numpyro) 통합 디스패치.

설계
----
- Stan: cmdstanpy 로 .stan 파일을 컴파일·샘플링.
- NumPyro: jax 기반 NUTS. Windows에서는 자동 fallback 가능.

모든 적합 함수는 동일한 시그니처를 갖고, 동일한 형식의 결과 dict를 반환:
    {
        "samples": dict[str, ndarray (n_chains, n_draws, ...)],
        "summary": pandas.DataFrame,
        "backend": "stan" | "numpyro",
        "fit_object": (선택) 원본 fit 객체
    }
"""

from __future__ import annotations
from pathlib import Path
from typing import Optional
import importlib.util
import platform
import warnings
import numpy as np

from .diagnostics import summarize_posterior


# ---------------------------------------------------------------------------
# 백엔드 사용 가능성 점검
# ---------------------------------------------------------------------------
def numpyro_available() -> bool:
    """NumPyro + jax 가 import 가능한가?"""
    if platform.system() == "Windows":
        # Windows에는 jax 공식 빌드가 없음 → 비권장
        if importlib.util.find_spec("jax") is None:
            return False
    return (importlib.util.find_spec("numpyro") is not None
            and importlib.util.find_spec("jax") is not None)


def resolve_backend(requested: str) -> str:
    """사용자가 지정한 백엔드를 환경에 맞게 해소(resolve)."""
    requested = requested.lower()
    if requested == "numpyro":
        if numpyro_available():
            return "numpyro"
        warnings.warn(
            "[difbayes] numpyro/jax not available → falling back to Stan.",
            RuntimeWarning,
        )
        return "stan"
    return "stan"


# ---------------------------------------------------------------------------
# 경로 헬퍼
# ---------------------------------------------------------------------------
def _stan_dir() -> Path:
    return Path(__file__).resolve().parent.parent / "models" / "stan"


def _numpyro_dir() -> Path:
    return Path(__file__).resolve().parent.parent / "models" / "numpyro"


# ---------------------------------------------------------------------------
# 모형 이름 → Stan / NumPyro 모듈 매핑
# ---------------------------------------------------------------------------
_STAN_FILES = {
    "rasch_basic":         "rasch_basic.stan",
    "rasch_dif":           "rasch_dif.stan",
    "rasch_hierarchical_dif": "rasch_hierarchical_dif.stan",
    "rasch_spike_slab":    "rasch_spike_slab.stan",
    "rasch_horseshoe":     "rasch_horseshoe.stan",
    "rasch_2pl_dif":       "rasch_2pl_dif.stan",
}

_NUMPYRO_MODULES = {
    "rasch_basic":            "models_numpyro_rasch_basic",
    "rasch_dif":              "models_numpyro_rasch_dif",
    "rasch_hierarchical_dif": "models_numpyro_rasch_hier",
    "rasch_spike_slab":       "models_numpyro_rasch_spikeslab",
    "rasch_horseshoe":        "models_numpyro_rasch_horseshoe",
    "rasch_2pl_dif":          "models_numpyro_rasch_2pl",
}


# ---------------------------------------------------------------------------
# Stan 적합
# ---------------------------------------------------------------------------
def _fit_stan(model_name: str, data: dict, n_chains: int, n_warmup: int,
              n_samples: int, seed: int) -> dict:
    import cmdstanpy
    stan_file = _stan_dir() / _STAN_FILES[model_name]
    print(f"[difbayes] Resolving Stan model file: {stan_file} ...")
    if not stan_file.exists():
        raise FileNotFoundError(f"Stan model not found: {stan_file}")
    else:
        print(f"[difbayes] Compiling Stan model: {stan_file.name} ...")
    model = cmdstanpy.CmdStanModel(stan_file=str(stan_file))
    fit = model.sample(
        data=data,
        chains=n_chains,
        iter_warmup=n_warmup,
        iter_sampling=n_samples,
        seed=seed,
        show_progress=False,
        show_console=False,
    )
    # cmdstanpy → dict of arrays (n_chains, n_draws, ...)
    samples = {}
    raw = fit.stan_variables()
    # cmdstanpy stan_variables() returns (n_chains*n_draws, ...) by default
    # We reshape to (n_chains, n_draws, ...)
    for name, arr in raw.items():
        arr = np.asarray(arr)
        if arr.ndim == 1:
            samples[name] = arr.reshape(n_chains, n_samples)
        else:
            samples[name] = arr.reshape(n_chains, n_samples, *arr.shape[1:])

    summary = summarize_posterior(samples)
    return dict(samples=samples, summary=summary, backend="stan", fit_object=fit)


# ---------------------------------------------------------------------------
# NumPyro 적합
# ---------------------------------------------------------------------------
def _fit_numpyro(model_name: str, data: dict, n_chains: int, n_warmup: int,
                 n_samples: int, seed: int) -> dict:
    # numpyro 모듈을 동적으로 로드
    import importlib.util as _ilu
    module_file = _numpyro_dir() / f"{_NUMPYRO_MODULES[model_name]}.py"
    if not module_file.exists():
        raise FileNotFoundError(f"NumPyro module not found: {module_file}")

    spec = _ilu.spec_from_file_location(_NUMPYRO_MODULES[model_name], module_file)
    mod = _ilu.module_from_spec(spec)
    spec.loader.exec_module(mod)

    import jax
    import jax.numpy as jnp
    import numpyro
    from numpyro.infer import MCMC, NUTS

    numpyro.set_host_device_count(n_chains)

    kernel = NUTS(mod.model)
    mcmc = MCMC(kernel,
                num_warmup=n_warmup,
                num_samples=n_samples,
                num_chains=n_chains,
                progress_bar=False)

    rng_key = jax.random.PRNGKey(seed)
    # data dict의 ndarray들을 jnp.array로 변환
    data_jax = {}
    for k, v in data.items():
        if isinstance(v, np.ndarray):
            data_jax[k] = jnp.asarray(v)
        else:
            data_jax[k] = v
    mcmc.run(rng_key, **data_jax)

    raw = mcmc.get_samples(group_by_chain=True)
    samples = {k: np.asarray(v) for k, v in raw.items()}
    summary = summarize_posterior(samples)
    return dict(samples=samples, summary=summary, backend="numpyro", fit_object=mcmc)


# ---------------------------------------------------------------------------
# 공개 API — 모형별 데이터 준비 + 적합 디스패치
# ---------------------------------------------------------------------------
def fit_rasch_basic(Y: np.ndarray, backend: str = "stan",
                    n_chains: int = 4, n_warmup: int = 500,
                    n_samples: int = 1000, seed: int = 2026) -> dict:
    """DIF 없음 가정의 기본 1PL Rasch 모형."""
    N, J = Y.shape
    # long format
    i_idx = np.repeat(np.arange(N), J) + 1
    j_idx = np.tile(np.arange(J), N) + 1
    y_flat = Y.flatten().astype(int)
    data = dict(N=N, J=J, K=N * J,
                ii=i_idx, jj=j_idx, y=y_flat)
    backend = resolve_backend(backend)
    if backend == "stan":
        return _fit_stan("rasch_basic", data, n_chains, n_warmup, n_samples, seed)
    return _fit_numpyro("rasch_basic", data, n_chains, n_warmup, n_samples, seed)


def fit_rasch_dif(Y, group, backend="stan",
                  n_chains=4, n_warmup=500, n_samples=1000,
                  prior_sigma_delta: float = 1.0, seed=2026) -> dict:
    """문항별 독립 Δb 를 가진 1PL Rasch DIF 모형 (non-hierarchical)."""
    N, J = Y.shape
    i_idx = np.repeat(np.arange(N), J) + 1
    j_idx = np.tile(np.arange(J), N) + 1
    g_idx = np.repeat(group, J).astype(int)
    y_flat = Y.flatten().astype(int)
    data = dict(N=N, J=J, K=N * J,
                ii=i_idx, jj=j_idx, gg=g_idx, y=y_flat,
                prior_sigma_delta=float(prior_sigma_delta))
    backend = resolve_backend(backend)
    if backend == "stan":
        return _fit_stan("rasch_dif", data, n_chains, n_warmup, n_samples, seed)
    return _fit_numpyro("rasch_dif", data, n_chains, n_warmup, n_samples, seed)


def fit_rasch_hierarchical_dif(Y, group, backend="stan",
                               n_chains=4, n_warmup=500, n_samples=1000,
                               seed=2026) -> dict:
    """위계 Δb_j ~ N(0, τ²) 1PL Rasch DIF 모형."""
    N, J = Y.shape
    i_idx = np.repeat(np.arange(N), J) + 1
    j_idx = np.tile(np.arange(J), N) + 1
    g_idx = np.repeat(group, J).astype(int)
    y_flat = Y.flatten().astype(int)
    data = dict(N=N, J=J, K=N * J,
                ii=i_idx, jj=j_idx, gg=g_idx, y=y_flat)
    backend = resolve_backend(backend)
    if backend == "stan":
        return _fit_stan("rasch_hierarchical_dif", data, n_chains, n_warmup, n_samples, seed)
    return _fit_numpyro("rasch_hierarchical_dif", data, n_chains, n_warmup, n_samples, seed)


def fit_rasch_spike_slab(Y, group, backend="stan",
                         n_chains=4, n_warmup=800, n_samples=1500,
                         slab_sd: float = 1.0, prior_inclusion: float = 0.2,
                         seed=2026) -> dict:
    """Spike-and-slab prior로 anchor-free DIF 검출."""
    N, J = Y.shape
    i_idx = np.repeat(np.arange(N), J) + 1
    j_idx = np.tile(np.arange(J), N) + 1
    g_idx = np.repeat(group, J).astype(int)
    y_flat = Y.flatten().astype(int)
    data = dict(N=N, J=J, K=N * J,
                ii=i_idx, jj=j_idx, gg=g_idx, y=y_flat,
                slab_sd=float(slab_sd),
                prior_inclusion=float(prior_inclusion))
    backend = resolve_backend(backend)
    if backend == "stan":
        return _fit_stan("rasch_spike_slab", data, n_chains, n_warmup, n_samples, seed)
    return _fit_numpyro("rasch_spike_slab", data, n_chains, n_warmup, n_samples, seed)


def fit_rasch_horseshoe(Y, group, backend="stan",
                        n_chains=4, n_warmup=800, n_samples=1500,
                        seed=2026) -> dict:
    """Horseshoe prior로 anchor-free DIF 검출."""
    N, J = Y.shape
    i_idx = np.repeat(np.arange(N), J) + 1
    j_idx = np.tile(np.arange(J), N) + 1
    g_idx = np.repeat(group, J).astype(int)
    y_flat = Y.flatten().astype(int)
    data = dict(N=N, J=J, K=N * J,
                ii=i_idx, jj=j_idx, gg=g_idx, y=y_flat)
    backend = resolve_backend(backend)
    if backend == "stan":
        return _fit_stan("rasch_horseshoe", data, n_chains, n_warmup, n_samples, seed)
    return _fit_numpyro("rasch_horseshoe", data, n_chains, n_warmup, n_samples, seed)


def fit_2pl_dif(Y, group, backend="stan",
                n_chains=4, n_warmup=800, n_samples=1500,
                seed=2026) -> dict:
    """2PL DIF: Δa(log-multiplicative) + Δb (uniform) 동시 추정."""
    N, J = Y.shape
    i_idx = np.repeat(np.arange(N), J) + 1
    j_idx = np.tile(np.arange(J), N) + 1
    g_idx = np.repeat(group, J).astype(int)
    y_flat = Y.flatten().astype(int)
    data = dict(N=N, J=J, K=N * J,
                ii=i_idx, jj=j_idx, gg=g_idx, y=y_flat)
    backend = resolve_backend(backend)
    if backend == "stan":
        return _fit_stan("rasch_2pl_dif", data, n_chains, n_warmup, n_samples, seed)
    return _fit_numpyro("rasch_2pl_dif", data, n_chains, n_warmup, n_samples, seed)
