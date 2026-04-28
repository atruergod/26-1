#!/usr/bin/env python3
"""
ss_map_laplace.py
=================
PCM-SEM: cmdstanpy MAP + Laplace Approximation 기반 Monte Carlo 시뮬레이션

방법론:
  1. model.optimize(jacobian=True)  →  MAP 추정치 (사후 최빈값)
  2. model.laplace_sample(mode=map_fit, draws=N_DRAWS)  →  Laplace 근사 사후 분포
  3. 사후 분포에서 95% CI 계산 → 포함 확률, 검출력 산출

요구사항:
  - cmdstanpy >= 1.1.0  (pip install cmdstanpy --upgrade)
  - CmdStan >= 2.33     (cmdstanpy.install_cmdstan())
  - sem_pcm_v2.stan 이 동일 디렉토리에 존재해야 함

저장 파일:
  ss_laplace_results.csv   — 반복별 원시 추정치 및 CI
  ss_laplace_summary.csv   — 조건별 편향/RMSE/포함확률/검출력 요약

실행 예시:
  python ss_map_laplace.py               # 기본 (200 reps, 두 시나리오)
  python ss_map_laplace.py --reps 50     # 빠른 테스트
  python ss_map_laplace.py --scenario medium --n 86  # 단일 조건

주의: N=86에서 reps=200 기준 약 10~30분 소요 (CPU 성능에 따라 다름)
"""

import argparse
import json
import os
import sys
import time
import traceback

import numpy as np
import pandas as pd

# ── cmdstanpy import ──────────────────────────────────────────────
try:
    import cmdstanpy
    from cmdstanpy import CmdStanModel
    print(f"cmdstanpy {cmdstanpy.__version__} 로드됨")
except ImportError:
    print("ERROR: cmdstanpy 가 설치되지 않았습니다.")
    print("  pip install cmdstanpy")
    print("  python -c \"import cmdstanpy; cmdstanpy.install_cmdstan()\"")
    sys.exit(1)

# ─────────────────────────────────────────────────────────────────
# 설정
# ─────────────────────────────────────────────────────────────────
SEED        = 2024
N_DRAWS     = 1000       # Laplace 샘플 수 (포함 확률 계산용)
N_REPS      = 200        # 기본 반복 횟수 (전체 실행; 테스트는 --reps 50 권장)
OUT_DIR     = os.path.dirname(os.path.abspath(__file__))
STAN_FILE   = os.path.join(OUT_DIR, 'sem_pcm_v2.stan')
CSV_RAW     = os.path.join(OUT_DIR, 'ss_laplace_results.csv')
CSV_SUM     = os.path.join(OUT_DIR, 'ss_laplace_summary.csv')

SAMPLE_SIZES = [50, 86, 100, 150, 200, 300]

EFFECT_SCENARIOS = {
    'small':  dict(beta1=0.30, beta2=0.30, gamma1=0.00,
                   gamma_M=0.10, gamma_Y=0.05, alpha_M=-0.15, alpha_Y=-0.20),
    'medium': dict(beta1=0.50, beta2=0.40, gamma1=0.20,
                   gamma_M=0.15, gamma_Y=0.10, alpha_M=-0.20, alpha_Y=-0.30),
}

# 문항 구성
I_X, I_M, I_Y, I, K = 4, 11, 6, 21, 5

# ─────────────────────────────────────────────────────────────────
# PCM 데이터 생성 (ss_monte_carlo.py 와 동일 로직)
# ─────────────────────────────────────────────────────────────────
def pcm_sample_vectorized(theta, deltas, rng):
    N  = len(theta)
    lp = np.zeros((N, K))
    for k in range(1, K):
        lp[:, k] = lp[:, k-1] + (theta - deltas[k-1])
    lp -= lp.max(axis=1, keepdims=True)
    gumbel = -np.log(-np.log(rng.uniform(size=(N, K)) + 1e-15))
    return np.argmax(lp + gumbel, axis=1) + 1   # 1-indexed

def _make_item_deltas(rng, tp, c_X=-1.0, c_M=0.3, c_Y=0.5):
    base = np.array([-1.5, -0.5, 0.5, 1.5])
    deltas = []
    for _ in range(I_X): deltas.append(c_X + base + rng.normal(0, 0.2, 4))
    for _ in range(I_M): deltas.append(c_M + base + rng.normal(0, 0.2, 4))
    for _ in range(I_Y): deltas.append(c_Y + base + rng.normal(0, 0.2, 4))
    return deltas

def generate_data(N, tp, rng, item_deltas=None):
    gender = (rng.uniform(size=N) < 0.60).astype(float)
    tX = rng.standard_normal(N)
    tM = tp['alpha_M'] + tp['beta1']*tX  + tp['gamma_M']*gender + rng.standard_normal(N)
    tY = tp['alpha_Y'] + tp['gamma1']*tX + tp['beta2']*tM + tp['gamma_Y']*gender + rng.standard_normal(N)

    if item_deltas is None:
        item_deltas = _make_item_deltas(rng, tp)

    y = np.zeros((N, I), dtype=np.int32)
    for i in range(I):
        if   i < I_X:          theta_use = tX
        elif i < I_X + I_M:    theta_use = tM
        else:                  theta_use = tY
        y[:, i] = pcm_sample_vectorized(theta_use, item_deltas[i], rng)

    return y, tX, tM, tY, gender

# ─────────────────────────────────────────────────────────────────
# Stan 데이터 딕셔너리 변환
# ─────────────────────────────────────────────────────────────────
def to_stan_data(y, gender):
    N = y.shape[0]
    return {
        'N':      N,
        'I':      I,
        'K':      K,
        'I_X':    I_X,
        'I_M':    I_M,
        'y':      y.tolist(),
        'gender': gender.tolist(),
    }

# ─────────────────────────────────────────────────────────────────
# CI 로부터 커버리지 / 검출력 계산
# ─────────────────────────────────────────────────────────────────
def coverage_and_power(lo, hi, true_val):
    cov = int(lo <= true_val <= hi)
    sig = int(lo > 0 or hi < 0)
    return cov, sig

# ─────────────────────────────────────────────────────────────────
# 단일 반복: MAP + Laplace
# ─────────────────────────────────────────────────────────────────
def run_one_laplace(model, N, tp, item_deltas, rng, scenario_name, rep_seed):
    y, tX, tM, tY, gender = generate_data(N, tp, rng, item_deltas)
    stan_data = to_stan_data(y, gender)

    true_vals = {
        'b1':  tp['beta1'],
        'b2':  tp['beta2'],
        'g1':  tp['gamma1'],
        'ind': tp['beta1'] * tp['beta2'],
        'tot': tp['gamma1'] + tp['beta1'] * tp['beta2'],
    }

    row = dict(N=N, scenario=scenario_name,
               beta1=tp['beta1'], beta2=tp['beta2'], gamma1=tp['gamma1'],
               method='laplace', status='ok')

    try:
        # ── Step 1: MAP ──────────────────────────────────────────
        map_fit = model.optimize(
            data=stan_data,
            jacobian=True,          # Bayesian MAP (포함 야코비안 보정)
            seed=rep_seed,
            show_console=False,
        )

        # ── Step 2: Laplace 근사 사후 분포 ──────────────────────
        laplace_fit = model.laplace_sample(
            data=stan_data,
            mode=map_fit,
            draws=N_DRAWS,
            seed=rep_seed,
            show_console=False,
        )

        draws = laplace_fit.draws_pd()

        # ── 파라미터 추출 ──────────────────────────────────────
        b1_s   = draws['b1'].values
        b2_s   = draws['b2'].values
        g1_s   = draws['g1'].values
        ind_s  = draws['indirect_effect'].values if 'indirect_effect' in draws.columns \
                 else b1_s * b2_s
        tot_s  = draws['total_effect'].values if 'total_effect' in draws.columns \
                 else g1_s + ind_s

        def ci_stats(samples, true_val, name):
            lo, hi = np.percentile(samples, [2.5, 97.5])
            mean   = samples.mean()
            bias   = mean - true_val
            rmse   = np.sqrt(((samples - true_val)**2).mean())
            cov, sig = coverage_and_power(lo, hi, true_val)
            return {
                f'{name}_mean': mean,
                f'{name}_lo':   lo,
                f'{name}_hi':   hi,
                f'{name}_bias': bias,
                f'{name}_rmse': rmse,
                f'{name}_cov':  cov,
                f'{name}_sig':  sig,
            }

        row.update(ci_stats(b1_s,  true_vals['b1'],  'b1'))
        row.update(ci_stats(b2_s,  true_vals['b2'],  'b2'))
        row.update(ci_stats(g1_s,  true_vals['g1'],  'g1'))
        row.update(ci_stats(ind_s, true_vals['ind'], 'ind'))
        row.update(ci_stats(tot_s, true_vals['tot'], 'tot'))

    except Exception as e:
        row['status'] = f'error: {str(e)[:80]}'
        # 실패 시 NaN 으로 채움
        for par in ['b1','b2','g1','ind','tot']:
            for suffix in ['_mean','_lo','_hi','_bias','_rmse','_cov','_sig']:
                row[f'{par}{suffix}'] = np.nan

    return row

# ─────────────────────────────────────────────────────────────────
# 메인 시뮬레이션 루프
# ─────────────────────────────────────────────────────────────────
def main(n_reps=N_REPS, scenarios=None, sample_sizes=None):
    if not os.path.exists(STAN_FILE):
        print(f"ERROR: Stan 파일을 찾을 수 없습니다: {STAN_FILE}")
        sys.exit(1)

    if scenarios is None:    scenarios    = list(EFFECT_SCENARIOS.keys())
    if sample_sizes is None: sample_sizes = SAMPLE_SIZES

    print(f"Stan 모델 컴파일 중: {STAN_FILE}")
    model = CmdStanModel(stan_file=STAN_FILE)
    print("컴파일 완료.\n")

    rng  = np.random.default_rng(SEED)
    rows = []
    total_cond = len(scenarios) * len(sample_sizes)
    cond_idx   = 0
    t_start    = time.time()
    n_errors   = 0

    for scenario_name in scenarios:
        tp = EFFECT_SCENARIOS[scenario_name]
        item_deltas = _make_item_deltas(rng, tp)

        for N in sample_sizes:
            cond_idx += 1
            t0 = time.time()
            cond_errors = 0
            print(f"[{cond_idx}/{total_cond}] scenario={scenario_name}, N={N}, reps={n_reps}")

            for r in range(n_reps):
                rep_seed = int(rng.integers(1, 100000))
                row = run_one_laplace(model, N, tp, item_deltas, rng, scenario_name, rep_seed)
                rows.append(row)
                if row['status'] != 'ok':
                    cond_errors += 1
                    n_errors    += 1
                if (r + 1) % 20 == 0:
                    elapsed = time.time() - t0
                    eta = elapsed / (r+1) * (n_reps - r - 1)
                    ok_rate = (r + 1 - cond_errors) / (r + 1)
                    print(f"  rep {r+1:3d}/{n_reps}  OK율={ok_rate:.1%}  ETA={eta:.0f}s")

            elapsed = time.time() - t0
            ok_n = n_reps - cond_errors
            print(f"  → 완료: {elapsed:.1f}s  OK={ok_n}/{n_reps}\n")

    df = pd.DataFrame(rows)
    df.to_csv(CSV_RAW, index=False)
    total_time = time.time() - t_start
    print(f"원시 결과 저장: {CSV_RAW}  ({len(df)} rows, 총 {total_time:.1f}s)")
    print(f"전체 오류 수: {n_errors}/{len(df)}")

    # ── 요약 통계 ─────────────────────────────────────────────
    ok_df = df[df.status == 'ok'].copy()
    summary_rows = []
    for (scenario_name, N), g in ok_df.groupby(['scenario', 'N']):
        row_s = dict(scenario=scenario_name, N=N, method='laplace', n_ok=len(g))
        for par in ['b1','b2','g1','ind']:
            tv = {'b1': g['beta1'].iloc[0], 'b2': g['beta2'].iloc[0],
                  'g1': g['gamma1'].iloc[0],
                  'ind': g['beta1'].iloc[0]*g['beta2'].iloc[0]}[par]
            row_s[f'{par}_bias']     = g[f'{par}_bias'].mean()
            row_s[f'{par}_rmse']     = g[f'{par}_rmse'].mean()
            row_s[f'{par}_coverage'] = g[f'{par}_cov'].mean()
            row_s[f'{par}_power']    = g[f'{par}_sig'].mean()
        summary_rows.append(row_s)

    df_sum = pd.DataFrame(summary_rows).round(4)
    df_sum.to_csv(CSV_SUM, index=False)
    print(f"요약 저장: {CSV_SUM}\n")
    print("=== 요약 (MAP+Laplace) ===")
    print(df_sum[['scenario','N','b1_bias','b1_coverage','ind_bias','ind_power']].to_string(index=False))
    return df, df_sum

# ─────────────────────────────────────────────────────────────────
# 단일 데이터셋 데모 (시뮬레이션 없이 한 번만 실행)
# ─────────────────────────────────────────────────────────────────
def demo_single(N=86, scenario='medium'):
    """
    단일 데이터셋에 대해 MAP + Laplace 를 실행하고 결과를 출력.
    전체 Monte Carlo 실행 전 동작 확인용.
    """
    print(f"=== 단일 데모: N={N}, scenario={scenario} ===")
    print(">> Reading Stan model from:", STAN_FILE)
    model = CmdStanModel(stan_file=STAN_FILE)
    rng   = np.random.default_rng(SEED)
    tp    = EFFECT_SCENARIOS[scenario]
    item_deltas = _make_item_deltas(rng, tp)

    y, tX, tM, tY, gender = generate_data(N, tp, rng, item_deltas)
    stan_data = to_stan_data(y, gender)

    # Save data as JSON for reproducibility
    data_file = os.path.join(OUT_DIR, f'ss_demo_data_N{N}_{scenario}.json')
    with open(data_file, 'w') as f:
        json.dump(stan_data, f)
    print(f"데이터 저장: {data_file}")

    print("MAP 최적화 실행 중...")
    t0 = time.time()
    map_fit = model.optimize(data=stan_data, jacobian=True, seed=SEED,
                             show_console=False)
    print(f"MAP 완료: {time.time()-t0:.1f}s")
    print(f"  MAP b1={map_fit.stan_variable('b1'):.4f}  (진값={tp['beta1']})")
    print(f"  MAP b2={map_fit.stan_variable('b2'):.4f}  (진값={tp['beta2']})")
    print(f"  MAP g1={map_fit.stan_variable('g1'):.4f}  (진값={tp['gamma1']})")

    print("Laplace 근사 샘플링 중...")
    t0 = time.time()
    laplace_fit = model.laplace_sample(
        data=stan_data, mode=map_fit, draws=2000, seed=SEED,
        show_console=False,
    )
    print(f"Laplace 완료: {time.time()-t0:.1f}s")

    draws = laplace_fit.draws_pd()
    print("\n=== Laplace 근사 사후 분포 요약 ===")
    print(f"{'파라미터':15s} {'진값':>8s} {'사후평균':>10s} {'95%CI 하한':>12s} {'95%CI 상한':>12s} {'편향':>8s}")
    print("-" * 65)
    params = [('b1', tp['beta1']), ('b2', tp['beta2']), ('g1', tp['gamma1'])]
    if 'indirect_effect' in draws.columns:
        params.append(('indirect_effect', tp['beta1']*tp['beta2']))
    for pname, tv in params:
        s = draws[pname].values
        lo, hi = np.percentile(s, [2.5, 97.5])
        print(f"  {pname:13s} {tv:8.4f} {s.mean():10.4f} {lo:12.4f} {hi:12.4f} {s.mean()-tv:8.4f}")

    # Laplace 결과 저장
    out_csv = os.path.join(OUT_DIR, f'ss_laplace_demo_N{N}_{scenario}.csv')
    draws.to_csv(out_csv, index=False)
    print(f"\nLaplace 샘플 저장: {out_csv}  ({len(draws)} rows × {len(draws.columns)} cols)")
    return draws

# ─────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PCM-SEM MAP+Laplace Monte Carlo')
    parser.add_argument('--demo',     action='store_true', help='단일 데이터셋 데모만 실행')
    parser.add_argument('--reps',     type=int,   default=N_REPS,    help='반복 횟수 (기본 200)')
    parser.add_argument('--scenario', type=str,   default=None,
                        help='시나리오 선택: small, medium (기본: 둘 다)')
    parser.add_argument('--n',        type=int,   default=None,
                        help='단일 표본 크기 선택 (기본: 전체 6개)')
    args = parser.parse_args()

    scenarios    = [args.scenario] if args.scenario else None
    sample_sizes = [args.n]        if args.n        else None

    if args.demo:
        N_demo        = args.n        if args.n        else 86
        sc_demo       = args.scenario if args.scenario else 'medium'
        demo_single(N=N_demo, scenario=sc_demo)
    else:
        print("=" * 60)
        print("PCM-SEM  MAP + Laplace Approximation  Monte Carlo")
        print("=" * 60)
        print(f"Stan 파일   : {STAN_FILE}")
        print(f"반복 횟수   : {args.reps}")
        print(f"시나리오    : {scenarios or list(EFFECT_SCENARIOS.keys())}")
        print(f"표본 크기   : {sample_sizes or SAMPLE_SIZES}")
        print(f"Laplace 드로우: {N_DRAWS}")
        print()
        print("권장 테스트: python ss_map_laplace.py --demo")
        print("전체 실행  : python ss_map_laplace.py --reps 200")
        print()
        main(n_reps=args.reps, scenarios=scenarios, sample_sizes=sample_sizes)
