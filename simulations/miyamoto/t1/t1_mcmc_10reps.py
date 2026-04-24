#!/usr/bin/env python3
"""
t1_mcmc_10reps.py
=================
N=86 시뮬레이션 데이터 10개 독립 생성 → 각각 Full-Bayesian MCMC 적용
(약한 사전 분포 + 강한 사전 분포 모형 비교)

목적:
  단일 시뮬레이션 결과의 우연성(난수 의존성)을 극복하고,
  Bayesian PCM-SEM이 반복 적용에서 일관된 사후 분포를 산출함을 실증.

산출:
  t1_mcmc_10reps.csv        — 반복별 사후 요약 (mean, SD, CrI, P>0)
  t1_mcmc_10reps_paths.png  — 10개 데이터셋의 경로 계수 분포 시각화

실행: python t1_mcmc_10reps.py [--reps 10] [--chains 4]
  예상 시간: 10회 × 2모형 × ~8분 ≈ 약 150분
  빠른 테스트:  python t1_mcmc_10reps.py --reps 3 --chains 2
"""
import argparse, os, sys, time
import numpy as np
import pandas as pd

HERE   = os.path.dirname(os.path.abspath(__file__))
PARENT = os.path.dirname(HERE)

try:
    from cmdstanpy import CmdStanModel
    import cmdstanpy
    print(f"cmdstanpy {cmdstanpy.__version__}")
except ImportError:
    print("ERROR: cmdstanpy 미설치."); sys.exit(1)

STAN_WEAK   = os.path.join(PARENT, 'sem_pcm_v2.stan')
STAN_STRONG = os.path.join(PARENT, 'sem_pcm_with_prior.stan')
for f in [STAN_WEAK, STAN_STRONG]:
    if not os.path.exists(f):
        print(f"ERROR: {f} 없음"); sys.exit(1)

# ─── 설정 ────────────────────────────────────────────────────────────────
N_SIM  = 86
I_X, I_M, I_Y, I, K = 4, 11, 6, 21, 5
C_X, C_M, C_Y       = -1.194, -0.380, -0.185

# 데이터 생성 참 파라미터 (medium 효과 크기)
TRUE = dict(beta1=0.50, beta2=0.40, gamma1=0.20,
            gamma_M=0.15, gamma_Y=0.10)

# PCM 기준 임계값 (원논문 평균 역산, base=[-1.5,-0.5,0.5,1.5])
_BASE = np.array([-1.5, -0.5, 0.5, 1.5])
_DELTA_X = (C_X + _BASE).tolist()   # [-2.694, -1.694, -0.694,  0.306]
_DELTA_M = (C_M + _BASE).tolist()   # [-1.880, -0.880,  0.120,  1.120]
_DELTA_Y = (C_Y + _BASE).tolist()   # [-1.685, -0.685,  0.315,  1.315]

# 강한 사전 분포 파라미터 (sem_pcm_with_prior.stan data 블록과 일치)
STRONG_PRIORS = dict(
    prior_b1_mu=0.35,  prior_b1_sd=0.20,   # β₁: X→M
    prior_b2_mu=0.35,  prior_b2_sd=0.20,   # β₂: M→Y
    prior_g1_mu=0.15,  prior_g1_sd=0.20,   # γ₁: X→Y 직접
    prior_aM_mu=0.0,   prior_aM_sd=1.0,    # α_M 절편
    prior_aY_mu=0.0,   prior_aY_sd=1.0,    # α_Y 절편
    prior_gM_mu=0.20,  prior_gM_sd=0.20,   # 성별→M
    prior_gY_mu=0.10,  prior_gY_sd=0.20,   # 성별→Y
    prior_delta_X=_DELTA_X,
    prior_delta_M=_DELTA_M,
    prior_delta_Y=_DELTA_Y,
    prior_delta_sd=0.30,
    prior_only=0,
)

# ─── 데이터 생성 ─────────────────────────────────────────────────────────
def sample_pcm(theta, c_off, rng, K=5):
    base = np.array([-1.5, -0.5, 0.5, 1.5]) + c_off
    lp = np.zeros((len(theta), K))
    for k in range(1, K):
        lp[:, k] = lp[:, k-1] + (theta - base[k-1])
    lp -= lp.max(axis=1, keepdims=True)
    p = np.exp(lp); p /= p.sum(axis=1, keepdims=True)
    u = rng.uniform(size=(len(theta), 1))
    return (u > np.cumsum(p, axis=1)).sum(axis=1) + 1

def generate_dataset(seed):
    rng = np.random.default_rng(seed)
    N   = N_SIM
    tp  = TRUE
    G   = (rng.uniform(size=N) < 0.686).astype(float)
    tX  = rng.standard_normal(N)
    tM  = tp['gamma_M']*G + tp['beta1']*tX  + rng.standard_normal(N)
    tY  = tp['gamma_Y']*G + tp['gamma1']*tX + tp['beta2']*tM + rng.standard_normal(N)
    Y   = np.zeros((N, I), dtype=int)
    for i in range(I_X):   Y[:, i]         = sample_pcm(tX, C_X+rng.normal(0,.2), rng)
    for i in range(I_M):   Y[:, I_X+i]     = sample_pcm(tM, C_M+rng.normal(0,.2), rng)
    for i in range(I_Y):   Y[:, I_X+I_M+i] = sample_pcm(tY, C_Y+rng.normal(0,.2), rng)
    return {'N':N,'I':I,'K':K,'I_X':I_X,'I_M':I_M,
            'y':Y.tolist(),'gender':G.tolist()}

# ─── MCMC 실행 ───────────────────────────────────────────────────────────
def run_mcmc(model, data, n_chains, n_warmup, n_sample, seed):
    try:
        fit = model.sample(
            data=data, chains=n_chains,
            iter_warmup=n_warmup, iter_sampling=n_sample,
            adapt_delta=0.92, max_treedepth=12,
            seed=seed, show_progress=False, show_console=False)
        return fit
    except Exception as e:
        print(f"    MCMC 실패: {e}")
        return None

def _get_divergences(fit):
    """발산 전이 수를 안전하게 추출 (cmdstanpy 버전 무관)."""
    try:
        # cmdstanpy 1.x
        return int(fit.method_variables()['divergent__'].sum())
    except Exception:
        pass
    try:
        # 구버전 fallback
        return int(fit.divergences())
    except Exception:
        return np.nan

def _get_summary_col(fit, param, col):
    """summary DataFrame에서 컬럼을 버전 호환적으로 읽기."""
    sumdf = fit.summary()
    if param not in sumdf.index:
        return np.nan
    # 컬럼 이름 후보 목록 (cmdstanpy 버전에 따라 다름)
    candidates = {
        'R_hat':   ['R_hat', 'R_hat'],
        'ESS':     ['ESS_bulk', 'N_Eff'],
    }
    for cname in candidates.get(col, [col]):
        if cname in sumdf.columns:
            return sumdf.loc[param, cname]
    return np.nan

def summarise_fit(fit, rep_id, model_label, seed):
    if fit is None:
        return None
    draws = fit.draws_pd()
    # Stan 모형 파라미터 이름: b1, b2, g1
    # Stan generated quantities: indirect_effect, total_effect, prop_mediated
    params = ['b1', 'b2', 'g1', 'gamma_M', 'gamma_Y',
              'indirect_effect', 'total_effect', 'prop_mediated']
    rows = []

    divs = _get_divergences(fit)

    for p in params:
        if p not in draws.columns:
            continue
        v = draws[p].dropna()
        lo, hi = np.percentile(v, [2.5, 97.5])
        rows.append(dict(
            rep=rep_id, model=model_label, seed=seed,
            param=p,
            mean=v.mean(), sd=v.std(),
            q025=lo, q975=hi,
            p_gt0=(v > 0).mean(),
            rhat=_get_summary_col(fit, p, 'R_hat'),
            ess=_get_summary_col(fit, p, 'ESS'),
            divergences=divs,
        ))
    return rows

# ─── 메인 ────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--reps',    type=int, default=10)
    ap.add_argument('--chains',  type=int, default=4)
    ap.add_argument('--warmup',  type=int, default=1000)
    ap.add_argument('--samples', type=int, default=1000)
    args = ap.parse_args()

    print("="*60)
    print(f"t1 MCMC {args.reps}회 반복 분석 (N={N_SIM})")
    print(f"  체인: {args.chains}  워밍업: {args.warmup}  샘플: {args.samples}")
    est_min = args.reps * 2 * args.chains * (args.warmup + args.samples) // 6000
    print(f"  예상 시간: {est_min}~{est_min*2}분")
    print("="*60)

    print("Stan 모형 컴파일 (force_compile=True — exe 재생성)...")
    model_weak   = CmdStanModel(stan_file=STAN_WEAK,   force_compile=True)
    model_strong = CmdStanModel(stan_file=STAN_STRONG, force_compile=True)
    print("컴파일 완료.")

    all_rows = []
    t0 = time.time()

    for rep in range(1, args.reps + 1):
        seed = 1000 + rep * 7
        print(f"\n[반복 {rep:02d}/{args.reps}] 데이터 생성 (seed={seed})")
        stan_data = generate_dataset(seed)

        # 약한 사전 분포
        print(f"  약한 사전 분포 MCMC ...")
        t1 = time.time()
        fit_w = run_mcmc(model_weak, stan_data,
                         args.chains, args.warmup, args.samples, seed)
        rows_w = summarise_fit(fit_w, rep, 'weak', seed)
        if rows_w:
            all_rows.extend(rows_w)
            b1w = next((r['mean'] for r in rows_w if r['param']=='b1'), np.nan)
            b2w = next((r['mean'] for r in rows_w if r['param']=='b2'), np.nan)
            div = next((r['divergences'] for r in rows_w), '?')
            print(f"    β₁={b1w:.3f}  β₂={b2w:.3f}  발산={div}  {time.time()-t1:.0f}s")

        # 강한 사전 분포
        print(f"  강한 사전 분포 MCMC ...")
        t1 = time.time()
        stan_data_s = dict(stan_data, **STRONG_PRIORS)
        fit_s = run_mcmc(model_strong, stan_data_s,
                         args.chains, args.warmup, args.samples, seed+1)
        rows_s = summarise_fit(fit_s, rep, 'strong', seed)
        if rows_s:
            all_rows.extend(rows_s)
            b1s = next((r['mean'] for r in rows_s if r['param']=='b1'), np.nan)
            b2s = next((r['mean'] for r in rows_s if r['param']=='b2'), np.nan)
            div = next((r['divergences'] for r in rows_s), '?')
            print(f"    β₁={b1s:.3f}  β₂={b2s:.3f}  발산={div}  {time.time()-t1:.0f}s")

        elapsed = time.time() - t0
        remain  = elapsed / rep * (args.reps - rep)
        print(f"  누적 {elapsed/60:.1f}분  잔여 약 {remain/60:.1f}분")

    # 저장
    df = pd.DataFrame(all_rows)
    out = os.path.join(HERE, 't1_mcmc_10reps.csv')
    df.to_csv(out, index=False)
    print(f"\n결과 저장: {out}  ({len(df)} 행)")

    # 간단 요약 출력
    for model_lbl, lbl in [('weak','약한 사전'), ('strong','강한 사전')]:
        print(f"\n[{lbl} 모형 — 주요 파라미터 사후 평균 요약]")
        for par, true_val in [('b1', TRUE['beta1']), ('b2', TRUE['beta2']),
                               ('g1', TRUE['gamma1']), ('indirect_effect', TRUE['beta1']*TRUE['beta2'])]:
            sub = df[(df['model']==model_lbl) & (df['param']==par)]
            if sub.empty:
                print(f"  {par}: (없음)")
                continue
            cover = ((sub['q025']<=true_val) & (sub['q975']>=true_val)).mean()
            sig   = (sub['q025'] > 0).mean()
            print(f"  {par:18s}  평균={sub['mean'].mean():.3f}  "
                  f"SD={sub['mean'].std():.3f}  "
                  f"참값={true_val:.3f}  포함={cover:.0%}  P(>0)={sig:.0%}")

    print(f"\nt1_mcmc_10reps.py 완료. 총 {(time.time()-t0)/60:.1f}분")
