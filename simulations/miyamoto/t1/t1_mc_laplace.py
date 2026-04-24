#!/usr/bin/env python3
"""
t1_mc_laplace.py
================
MAP + Laplace 근사를 이용한 몬테카를로 시뮬레이션 (CmdStan 필요)

비교:
  CS-OLS     — 합산 점수 OLS (기존 방법, 편향 있음)
  MAP+Laplace— Bayesian PCM-SEM 근사 (실용적 베이지안 추정)
  LV-OLS     — 잠재 변수 오라클 (이론 상한, 참조용)

조건:
  N: 50, 86, 100, 150, 200, 300
  효과 시나리오: small (β₁=0.30), medium (β₁=0.50)
  반복: 50회 per condition  (~총 90-120분, 병렬 미적용 기준)

산출:
  t1_mc_laplace_results.csv  — 반복별 원시 추정치
  t1_mc_laplace_summary.csv  — 조건별 요약 (bias, RMSE, coverage, power)

실행: python t1_mc_laplace.py [--reps 30] [--n 86]
"""

import argparse, os, sys, time, traceback
import numpy as np
import pandas as pd
from scipy import stats

HERE   = os.path.dirname(os.path.abspath(__file__))
PARENT = os.path.dirname(HERE)

# ─── cmdstanpy 로드 ──────────────────────────────────────────────────────
try:
    from cmdstanpy import CmdStanModel
    import cmdstanpy
    print(f"cmdstanpy {cmdstanpy.__version__} 로드됨")
except ImportError:
    print("ERROR: cmdstanpy 미설치. pip install cmdstanpy 후 재실행하세요.")
    sys.exit(1)

STAN_FILE = os.path.join(PARENT, 'sem_pcm_v2.stan')
if not os.path.exists(STAN_FILE):
    print(f"ERROR: {STAN_FILE} 없음")
    sys.exit(1)

# ─── 설정 ────────────────────────────────────────────────────────────────
SEED         = 2024
N_DRAWS      = 1000      # Laplace 샘플 수

SAMPLE_SIZES = [50, 86, 100, 150, 200, 300]
SCENARIOS = {
    'small':  dict(beta1=0.30, beta2=0.30, gamma1=0.00,
                   gamma_M=0.10, gamma_Y=0.05),
    'medium': dict(beta1=0.50, beta2=0.40, gamma1=0.20,
                   gamma_M=0.15, gamma_Y=0.10),
}
I_X, I_M, I_Y, I, K = 4, 11, 6, 21, 5
C_X, C_M, C_Y       = -1.194, -0.380, -0.185

# ─── 데이터 생성 ─────────────────────────────────────────────────────────
def sample_pcm(theta, c_off, rng, K=5):
    base = np.array([-1.5, -0.5, 0.5, 1.5]) + c_off
    lp = np.zeros((len(theta), K))
    for k in range(1, K):
        lp[:, k] = lp[:, k-1] + (theta - base[k-1])
    lp -= lp.max(axis=1, keepdims=True)
    u = rng.uniform(size=(len(theta), 1))
    return (u > np.cumsum(np.exp(lp) / np.exp(lp).sum(axis=1, keepdims=True),
                          axis=1)).sum(axis=1) + 1

def generate_data(N, tp, rng):
    G  = (rng.uniform(size=N) < 0.686).astype(float)
    tX = rng.standard_normal(N)
    tM = tp['gamma_M']*G + tp['beta1']*tX  + rng.standard_normal(N)
    tY = tp['gamma_Y']*G + tp['gamma1']*tX + tp['beta2']*tM + rng.standard_normal(N)
    Y  = np.zeros((N, I), dtype=int)
    for i in range(I_X):
        Y[:, i]          = sample_pcm(tX, C_X + rng.normal(0,.2), rng)
    for i in range(I_M):
        Y[:, I_X+i]      = sample_pcm(tM, C_M + rng.normal(0,.2), rng)
    for i in range(I_Y):
        Y[:, I_X+I_M+i]  = sample_pcm(tY, C_Y + rng.normal(0,.2), rng)
    return Y, tX, tM, tY, G

# ─── OLS 보조 ────────────────────────────────────────────────────────────
def ols_beta(X, M, Y, G):
    N = len(X)
    Xm  = np.column_stack([np.ones(N), X, G])
    bM  = np.linalg.lstsq(Xm, M, rcond=None)[0]
    resM = M - Xm @ bM
    seM  = np.sqrt(((resM**2).sum()/(N-3)) * np.linalg.inv(Xm.T@Xm)[1,1])
    b1   = bM[1]
    Xy   = np.column_stack([np.ones(N), X, M, G])
    bY   = np.linalg.lstsq(Xy, Y, rcond=None)[0]
    resY = Y - Xy @ bY
    seY  = np.sqrt(((resY**2).sum()/(N-4)) * np.linalg.inv(Xy.T@Xy)[2,2])
    b2   = bY[2]
    return b1, b2, seM, seY

def ci_ok(est, se, true): return int(est - 1.96*se <= true <= est + 1.96*se)

# ─── Laplace 추정 ────────────────────────────────────────────────────────
def laplace_estimate(model, stan_data, N_draws, rng_seed):
    try:
        map_fit = model.optimize(
            data=stan_data, seed=rng_seed, jacobian=True,
            iter=5000, refresh=0)
        lap_fit = model.laplace_sample(
            data=stan_data, mode=map_fit,
            draws=N_draws, seed=rng_seed, refresh=0)
        draws = lap_fit.draws_pd()
        res = {}
        for p in ['beta1', 'beta2', 'gamma1']:
            if p in draws.columns:
                v = draws[p]
                lo, hi = np.percentile(v, [2.5, 97.5])
                res[p] = (v.mean(), lo, hi)
        if 'beta1' in res and 'beta2' in res:
            ind = draws['beta1'] * draws['beta2']
            res['indirect'] = (ind.mean(), *np.percentile(ind, [2.5, 97.5]))
        return res
    except Exception:
        return None

# ─── 메인 ────────────────────────────────────────────────────────────────
def run(n_reps, sample_sizes):
    print(f"Stan 모형 컴파일: {STAN_FILE}")
    model = CmdStanModel(stan_file=STAN_FILE)
    rng   = np.random.default_rng(SEED)
    records = []
    total = len(sample_sizes) * len(SCENARIOS) * n_reps
    done  = 0
    t0    = time.time()

    for sc, tp in SCENARIOS.items():
        for N in sample_sizes:
            for rep in range(n_reps):
                Y, tX, tM, tY, G = generate_data(N, tp, rng)
                csX = Y[:, :I_X].mean(1)
                csM = Y[:, I_X:I_X+I_M].mean(1)
                csY = Y[:, I_X+I_M:].mean(1)

                # CS-OLS
                b1c, b2c, sec1, sec2 = ols_beta(csX, csM, csY, G)
                # LV-OLS
                b1l, b2l, sel1, sel2 = ols_beta(tX,  tM,  tY,  G)

                # MAP+Laplace
                sd = {'N':N,'I':I,'K':K,'I_X':I_X,'I_M':I_M,
                      'y':Y.tolist(),'gender':G.tolist()}
                lap = laplace_estimate(model, sd, N_DRAWS, rep*7+done)

                rec = dict(scenario=sc, N=N, rep=rep,
                           true_b1=tp['beta1'], true_b2=tp['beta2'],
                           true_ind=tp['beta1']*tp['beta2'],
                           cs_b1=b1c, cs_b1_cov=ci_ok(b1c, sec1, tp['beta1']),
                           cs_b2=b2c, cs_b2_cov=ci_ok(b2c, sec2, tp['beta2']),
                           lv_b1=b1l, lv_b1_cov=ci_ok(b1l, sel1, tp['beta1']),
                           lv_b2=b2l, lv_b2_cov=ci_ok(b2l, sel2, tp['beta2']))

                if lap:
                    b1m,b1lo,b1hi = lap.get('beta1',(np.nan,np.nan,np.nan))
                    b2m,b2lo,b2hi = lap.get('beta2',(np.nan,np.nan,np.nan))
                    indm,indlo,indhi = lap.get('indirect',(np.nan,np.nan,np.nan))
                    rec.update(dict(
                        lap_b1=b1m, lap_b1_cov=int(b1lo<=tp['beta1']<=b1hi),
                        lap_b2=b2m, lap_b2_cov=int(b2lo<=tp['beta2']<=b2hi),
                        lap_ind=indm,
                        lap_ind_sig=int(~np.isnan(indlo) and indlo>0 or indhi<0),
                    ))
                else:
                    rec.update(dict(lap_b1=np.nan,lap_b1_cov=np.nan,
                                    lap_b2=np.nan,lap_b2_cov=np.nan,
                                    lap_ind=np.nan,lap_ind_sig=np.nan))
                records.append(rec)
                done += 1
                if done % 20 == 0:
                    print(f"  {done}/{total} ({100*done/total:.0f}%) "
                          f"  {time.time()-t0:.0f}s  [{sc} N={N}]")

    return pd.DataFrame(records)

def summarise(df):
    rows = []
    for (sc,N), g in df.groupby(['scenario','N']):
        tb1 = g['true_b1'].iloc[0]
        tb2 = g['true_b2'].iloc[0]
        ind = g['true_ind'].iloc[0]
        lap = g['lap_b1'].notna()
        rows.append(dict(scenario=sc, N=N,
            cs_b1_bias=(g['cs_b1']-tb1).mean(), cs_b1_cov=g['cs_b1_cov'].mean(),
            lv_b1_bias=(g['lv_b1']-tb1).mean(), lv_b1_cov=g['lv_b1_cov'].mean(),
            lap_b1_bias=(g.loc[lap,'lap_b1']-tb1).mean() if lap.any() else np.nan,
            lap_b1_cov=g.loc[lap,'lap_b1_cov'].mean() if lap.any() else np.nan,
            cs_b2_bias=(g['cs_b2']-tb2).mean(), cs_b2_cov=g['cs_b2_cov'].mean(),
            lv_b2_bias=(g['lv_b2']-tb2).mean(), lv_b2_cov=g['lv_b2_cov'].mean(),
            lap_b2_bias=(g.loc[lap,'lap_b2']-tb2).mean() if lap.any() else np.nan,
            lap_b2_cov=g.loc[lap,'lap_b2_cov'].mean() if lap.any() else np.nan,
            lap_ind_power=g.loc[lap,'lap_ind_sig'].mean() if lap.any() else np.nan,
        ))
    return pd.DataFrame(rows)

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--reps', type=int, default=50)
    ap.add_argument('--n',    type=int, default=None,
                    help='단일 N만 실행 (예: --n 86)')
    args = ap.parse_args()
    sizes = [args.n] if args.n else SAMPLE_SIZES

    print("="*60)
    print("t1 MAP+Laplace 몬테카를로")
    print(f"조건: {sizes} × {len(SCENARIOS)} 시나리오 × {args.reps} 반복")
    print(f"예상 시간: {len(sizes)*len(SCENARIOS)*args.reps*20//60}~"
          f"{len(sizes)*len(SCENARIOS)*args.reps*40//60} 분")
    print("="*60)

    df = run(args.reps, sizes)
    df.to_csv(os.path.join(HERE,'t1_mc_laplace_results.csv'), index=False)
    sm = summarise(df)
    sm.to_csv(os.path.join(HERE,'t1_mc_laplace_summary.csv'), index=False)
    print("\n[medium 시나리오 β₁ 비교]")
    med = sm[sm['scenario']=='medium'][
        ['N','cs_b1_bias','lap_b1_bias','lv_b1_bias',
         'cs_b1_cov','lap_b1_cov','lv_b1_cov']]
    print(med.to_string(index=False, float_format=lambda x:f'{x:+.3f}'))
