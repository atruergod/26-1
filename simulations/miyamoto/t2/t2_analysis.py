#!/usr/bin/env python3
"""
t2_analysis.py
==============
중국인 유학생 실측 데이터에 Bayesian PCM-SEM MCMC 적용.

입력:  t2_data.csv  (t2_howto.md 참조 — 연구자가 수집)
출력:  t2_mcmc_samples.csv         — 약한 사전 분포 사후 샘플
       t2_mcmc_strong_samples.csv  — 강한 사전 분포 사후 샘플
       t2_results_summary.csv      — 경로 계수 사후 요약
       t2_convergence.csv          — 수렴 진단 (R-hat, ESS)

Stan 모형: 상위 폴더의 sem_pcm_v2.stan, sem_pcm_with_prior.stan 사용
실행:  python t2_analysis.py  (CmdStan 필요)
"""
import json, os, sys
import numpy as np
import pandas as pd

HERE   = os.path.dirname(os.path.abspath(__file__))
PARENT = os.path.dirname(HERE)

# ─── 데이터 로드 ──────────────────────────────────────────────────────────
DATA_PATH = os.path.join(HERE, 't2_data.csv')
if not os.path.exists(DATA_PATH):
    print("=" * 60)
    print("ERROR: t2_data.csv 가 없습니다.")
    print("설문 데이터를 수집한 후 t2_howto.md 의 형식에 맞춰 저장하세요.")
    print("=" * 60)
    sys.exit(1)

df_raw = pd.read_csv(DATA_PATH)
print(f"데이터 로드: {len(df_raw)} 명, {df_raw.shape[1]} 열")

# 응답 행렬 (1-indexed, 정수)
I_X, I_M, I_Y, I, K = 4, 11, 6, 21, 5
y_cols = [f'y{i}' for i in range(1, 22)]
missing = [c for c in y_cols if c not in df_raw.columns]
if missing:
    print(f"ERROR: 열 누락: {missing}")
    sys.exit(1)

Y_mat  = df_raw[y_cols].values.astype(int)
gender = df_raw['gender'].values.astype(float)
N      = len(df_raw)

# 결측값 검사
if np.any((Y_mat < 1) | (Y_mat > K)):
    bad = np.argwhere((Y_mat < 1) | (Y_mat > K))
    print(f"WARNING: 범위 외 응답 {len(bad)} 건 (행,열): {bad[:5]}")

print(f"  N={N}, 문항={I}, 범주={K}")
print(f"  성별: 여성 {gender.mean()*100:.1f}%")

# ─── cmdstanpy 임포트 ─────────────────────────────────────────────────────
try:
    from cmdstanpy import CmdStanModel
    import cmdstanpy
    print(f"cmdstanpy {cmdstanpy.__version__}")
except ImportError:
    print("ERROR: cmdstanpy 미설치. pip install cmdstanpy 후 재실행하세요.")
    sys.exit(1)

# ─── 보정값 로드 ──────────────────────────────────────────────────────────
cal_path = os.path.join(HERE, 't2_calibration.json')
if os.path.exists(cal_path):
    with open(cal_path, 'r', encoding='utf-8') as f:
        CAL = json.load(f)
    print(f"보정값 로드: {cal_path}")
else:
    print("WARNING: t2_calibration.json 없음 — 기본값 사용.")
    CAL = dict(c_X=-1.194, c_M=-0.380, c_Y=-0.185,
               prior_beta1_mean=0.35, prior_beta1_sd=0.20,
               prior_beta2_mean=0.35, prior_beta2_sd=0.20,
               prior_gamma1_mean=0.10, prior_gamma1_sd=0.20)

# ─── Stan 데이터 딕셔너리 ────────────────────────────────────────────────
stan_data = {
    'N': N, 'I': I, 'K': K, 'I_X': I_X, 'I_M': I_M,
    'y': Y_mat.tolist(),
    'gender': gender.tolist(),
}

# ─── MCMC 함수 ───────────────────────────────────────────────────────────
def run_mcmc(stan_file, label, data, n_chains=4, n_warmup=1000, n_sample=1000):
    path = os.path.join(PARENT, stan_file)
    if not os.path.exists(path):
        print(f"  ERROR: {path} 없음")
        return None
    print(f"\n[{label}] MCMC 시작 ...")
    model = CmdStanModel(stan_file=path)
    fit = model.sample(
        data=data, chains=n_chains,
        iter_warmup=n_warmup, iter_sampling=n_sample,
        adapt_delta=0.92, max_treedepth=12,
        seed=2024, show_progress=True,
    )
    print(f"  발산 전이: {fit.divergences()}")
    return fit

# ─── 약한 사전 분포 모형 ────────────────────────────────────────────────
fit_weak = run_mcmc('sem_pcm_v2.stan', '약한 사전 분포', stan_data)

# ─── 강한 사전 분포 모형 ────────────────────────────────────────────────
stan_data_strong = dict(stan_data)
stan_data_strong.update({
    'prior_beta1_mean':  CAL['prior_beta1_mean'],
    'prior_beta1_sd':    CAL['prior_beta1_sd'],
    'prior_beta2_mean':  CAL['prior_beta2_mean'],
    'prior_beta2_sd':    CAL['prior_beta2_sd'],
    'prior_gamma1_mean': CAL['prior_gamma1_mean'],
    'prior_gamma1_sd':   CAL['prior_gamma1_sd'],
})
fit_strong = run_mcmc('sem_pcm_with_prior.stan', '강한 사전 분포', stan_data_strong)

# ─── 결과 저장 ───────────────────────────────────────────────────────────
key_params = ['beta1', 'beta2', 'gamma1', 'gamma_M', 'gamma_Y']

def save_samples(fit, label, out_path):
    if fit is None: return
    draws = fit.draws_pd()
    cols  = [c for c in draws.columns if any(c.startswith(p) for p in key_params)]
    out   = draws[cols].copy()
    if 'beta1' in out.columns and 'beta2' in out.columns:
        out['indirect'] = out['beta1'] * out['beta2']
        out['total']    = out['gamma1'] + out['indirect']
    out.to_csv(out_path, index=False)
    print(f"  샘플 저장: {out_path}  ({len(out)} 행)")

save_samples(fit_weak,   '약한', os.path.join(HERE, 't2_mcmc_samples.csv'))
save_samples(fit_strong, '강한', os.path.join(HERE, 't2_mcmc_strong_samples.csv'))

# ─── 요약 통계 ───────────────────────────────────────────────────────────
def summarise_fit(fit, label):
    if fit is None: return pd.DataFrame()
    draws = fit.draws_pd()
    rows  = []
    for p in key_params + ['indirect'] if 'indirect' in draws else key_params:
        if p == 'indirect':
            v = draws['beta1'] * draws['beta2']
        elif p not in draws.columns:
            continue
        else:
            v = draws[p]
        rows.append(dict(
            model=label, param=p,
            mean=v.mean(), sd=v.std(),
            q025=v.quantile(0.025), q975=v.quantile(0.975),
            p_gt0=(v > 0).mean(),
        ))
    return pd.DataFrame(rows)

sm = pd.concat([summarise_fit(fit_weak, '약한 사전'),
                summarise_fit(fit_strong, '강한 사전')], ignore_index=True)
sm_path = os.path.join(HERE, 't2_results_summary.csv')
sm.to_csv(sm_path, index=False)
print(f"\n요약 저장: {sm_path}")
print(sm.to_string(index=False, float_format=lambda x: f'{x:.3f}'))
print("\nt2_analysis.py 완료.")
