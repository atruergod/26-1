#!/usr/bin/env python3
"""
ss_mcmc_runner.py
=================
PCM-SEM 풀 베이지안 MCMC 실행기 (사용자 로컬에서 실행)

이 스크립트는 고정된 데이터셋을 생성하고 Stan NUTS-HMC MCMC를 실행한다.
결과 CSV 파일을 연구자에게 전달하면 베이지안 분석에 활용할 수 있다.

요구사항:
  - cmdstanpy >= 1.1.0  (pip install cmdstanpy --upgrade)
  - CmdStan >= 2.33     (python -c "import cmdstanpy; cmdstanpy.install_cmdstan()")
  - sem_pcm_v2.stan 이 동일 디렉토리에 존재

생성 파일:
  ss_mcmc_data_N{N}_{scenario}.json    — 재현을 위한 Stan 입력 데이터
  ss_mcmc_samples_N{N}_{scenario}.csv  — MCMC 후행 샘플 (draws × params)
  ss_mcmc_diagnostics.txt              — R-hat, ESS, divergence 요약

실행 예시:
  python ss_mcmc_runner.py              # 기본 (N=86, medium)
  python ss_mcmc_runner.py --all        # 세 조건 모두 (N=50, 86, 200 / medium)
  python ss_mcmc_runner.py --n 86 --scenario small

소요 시간 예상 (CPU 4코어 기준):
  N=50   medium  →  약 10~20분
  N=86   medium  →  약 20~40분
  N=200  medium  →  약 60~120분
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd

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
SEED    = 2024
OUT_DIR = os.path.dirname(os.path.abspath(__file__))
STAN_FILE = os.path.join(OUT_DIR, 'sem_pcm_v2.stan')

# MCMC 설정
CHAINS       = 4
ITER_WARMUP  = 1000
ITER_SAMPLING = 1000    # chains × iter_sampling = 4000 유효 샘플
ADAPT_DELTA  = 0.90     # 높은 수용률 (복잡한 계층 모형에 권장)
MAX_TREEDEPTH = 12

# 문항 구성
I_X, I_M, I_Y, I, K = 4, 11, 6, 21, 5

EFFECT_SCENARIOS = {
    'small':  dict(beta1=0.30, beta2=0.30, gamma1=0.00,
                   gamma_M=0.10, gamma_Y=0.05, alpha_M=-0.15, alpha_Y=-0.20),
    'medium': dict(beta1=0.50, beta2=0.40, gamma1=0.20,
                   gamma_M=0.15, gamma_Y=0.10, alpha_M=-0.20, alpha_Y=-0.30),
}

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
    return np.argmax(lp + gumbel, axis=1) + 1

def _make_item_deltas(rng, tp, c_X=-1.0, c_M=0.3, c_Y=0.5):
    base = np.array([-1.5, -0.5, 0.5, 1.5])
    deltas = []
    for _ in range(I_X): deltas.append(c_X + base + rng.normal(0, 0.2, 4))
    for _ in range(I_M): deltas.append(c_M + base + rng.normal(0, 0.2, 4))
    for _ in range(I_Y): deltas.append(c_Y + base + rng.normal(0, 0.2, 4))
    return deltas

def generate_data(N, tp, rng, item_deltas):
    gender = (rng.uniform(size=N) < 0.60).astype(float)
    tX = rng.standard_normal(N)
    tM = tp['alpha_M'] + tp['beta1']*tX  + tp['gamma_M']*gender + rng.standard_normal(N)
    tY = tp['alpha_Y'] + tp['gamma1']*tX + tp['beta2']*tM + tp['gamma_Y']*gender + rng.standard_normal(N)

    y = np.zeros((N, I), dtype=np.int32)
    for i in range(I):
        if   i < I_X:          theta_use = tX
        elif i < I_X + I_M:    theta_use = tM
        else:                   theta_use = tY
        y[:, i] = pcm_sample_vectorized(theta_use, item_deltas[i], rng)

    return y, tX, tM, tY, gender

# ─────────────────────────────────────────────────────────────────
# 단일 조건 MCMC 실행
# ─────────────────────────────────────────────────────────────────
def run_mcmc(model, N, scenario, verbose=True):
    tp  = EFFECT_SCENARIOS[scenario]
    rng = np.random.default_rng(SEED)
    item_deltas = _make_item_deltas(rng, tp)

    print(f"\n{'='*60}")
    print(f"MCMC: N={N}, scenario={scenario}")
    print(f"  진모수: β₁={tp['beta1']}, β₂={tp['beta2']}, γ₁={tp['gamma1']}")
    print(f"  간접 효과 진값: {tp['beta1']*tp['beta2']:.4f}")
    print(f"{'='*60}")

    # 데이터 생성
    y, tX, tM, tY, gender = generate_data(N, tp, rng, item_deltas)
    stan_data = {
        'N': N, 'I': I, 'K': K, 'I_X': I_X, 'I_M': I_M,
        'y': y.tolist(), 'gender': gender.tolist(),
    }

    # 데이터 저장 (재현성)
    data_file = os.path.join(OUT_DIR, f'ss_mcmc_data_N{N}_{scenario}.json')
    with open(data_file, 'w') as f:
        # Stan JSON 형식: 잠재 진값도 함께 저장 (검증용)
        save_data = dict(stan_data)
        save_data['true_params'] = {
            'beta1':  tp['beta1'],   'beta2':  tp['beta2'],
            'gamma1': tp['gamma1'],  'alpha_M': tp['alpha_M'],
            'alpha_Y': tp['alpha_Y'], 'gamma_M': tp['gamma_M'],
            'gamma_Y': tp['gamma_Y'],
            'indirect': tp['beta1']*tp['beta2'],
            'total':    tp['gamma1'] + tp['beta1']*tp['beta2'],
        }
        json.dump(save_data, f, indent=2)
    print(f"데이터 저장: {data_file}")

    # ── MCMC 실행 ───────────────────────────────────────────────
    print(f"\nMCMC 실행 중 ({CHAINS} chains × {ITER_WARMUP} warmup + {ITER_SAMPLING} sampling)...")
    print(f"예상 소요 시간: N={N} → {_estimate_time(N)}")
    t0 = time.time()

    try:
        fit = model.sample(
            data=stan_data,
            chains=CHAINS,
            iter_warmup=ITER_WARMUP,
            iter_sampling=ITER_SAMPLING,
            seed=SEED,
            adapt_delta=ADAPT_DELTA,
            max_treedepth=MAX_TREEDEPTH,
            show_progress=verbose,
            show_console=False,
        )
    except Exception as e:
        print(f"MCMC 실행 오류: {e}")
        raise

    elapsed = time.time() - t0
    print(f"\nMCMC 완료: {elapsed:.1f}s ({elapsed/60:.1f}분)")

    # ── 진단 ────────────────────────────────────────────────────
    print("\n=== MCMC 진단 ===")
    summary = fit.summary()
    key_params = ['b1', 'b2', 'g1', 'gamma_M', 'gamma_Y', 'alpha_M', 'alpha_Y',
                  'indirect_effect', 'total_effect', 'prop_mediated']
    key_rows = summary[summary.index.isin(key_params)].copy()

    print(f"\n{'파라미터':22s} {'진값':>7s} {'사후평균':>9s} {'5%':>8s} {'95%':>8s} {'R-hat':>7s} {'ESS':>7s}")
    print("-" * 70)
    true_p = {
        'b1':              tp['beta1'],
        'b2':              tp['beta2'],
        'g1':              tp['gamma1'],
        'gamma_M':         tp['gamma_M'],
        'gamma_Y':         tp['gamma_Y'],
        'alpha_M':         tp['alpha_M'],
        'alpha_Y':         tp['alpha_Y'],
        'indirect_effect': tp['beta1']*tp['beta2'],
        'total_effect':    tp['gamma1']+tp['beta1']*tp['beta2'],
        'prop_mediated':   (tp['beta1']*tp['beta2']) / (tp['gamma1']+tp['beta1']*tp['beta2'])
                           if abs(tp['gamma1']+tp['beta1']*tp['beta2']) > 1e-10 else float('nan'),
    }
    for pname in key_params:
        if pname in summary.index:
            row = summary.loc[pname]
            tv  = true_p.get(pname, float('nan'))
            mean_val = row.get('Mean', row.get('mean', float('nan')))
            q5  = row.get('5%',  row.get('5.0%', float('nan')))
            q95 = row.get('95%', row.get('95.0%', float('nan')))
            rhat = row.get('R_hat', row.get('r_hat', float('nan')))
            ess  = row.get('N_Eff', row.get('ess_bulk', float('nan')))
            print(f"  {pname:20s} {tv:7.4f} {mean_val:9.4f} {q5:8.4f} {q95:8.4f} {rhat:7.3f} {ess:7.0f}")

    # R-hat 경고
    rhat_col = 'R_hat' if 'R_hat' in summary.columns else 'r_hat'
    if rhat_col in summary.columns:
        bad_rhat = summary[summary[rhat_col] > 1.05]
        if len(bad_rhat) > 0:
            print(f"\n⚠️  R-hat > 1.05 인 파라미터: {len(bad_rhat)}개")
            print("   더 많은 반복 수 또는 adapt_delta 증가 권장")
        else:
            print(f"\n✓ 모든 R-hat ≤ 1.05 (수렴 양호)")

    # Divergence 확인
    try:
        divs = fit.method_variables().get('divergent__', None)
        if divs is not None:
            n_div = int(divs.sum())
            if n_div > 0:
                print(f"⚠️  발산 전이(divergent transitions): {n_div}개")
            else:
                print("✓ 발산 전이 없음")
    except Exception:
        pass

    # ── 사후 샘플 CSV 저장 ──────────────────────────────────────
    draws_df = fit.draws_pd()

    # 관심 파라미터만 추출
    cols_to_keep = ([c for c in draws_df.columns
                     if c in ['b1','b2','g1','gamma_M','gamma_Y',
                               'alpha_M','alpha_Y',
                               'indirect_effect','total_effect','prop_mediated',
                               'lp__', 'chain__', 'iter__']]
                    + [c for c in draws_df.columns if c.startswith('chain') or c.startswith('iter')])

    # 중복 제거 + 순서 유지
    seen = set()
    cols_final = []
    for c in cols_to_keep:
        if c not in seen and c in draws_df.columns:
            cols_final.append(c)
            seen.add(c)

    draws_out = draws_df[cols_final].copy()
    # 메타 정보 추가
    draws_out['N']        = N
    draws_out['scenario'] = scenario
    draws_out['true_b1']  = tp['beta1']
    draws_out['true_b2']  = tp['beta2']
    draws_out['true_g1']  = tp['gamma1']
    draws_out['true_ind'] = tp['beta1'] * tp['beta2']

    csv_out = os.path.join(OUT_DIR, f'ss_mcmc_samples_N{N}_{scenario}.csv')
    draws_out.to_csv(csv_out, index=False)
    total_draws = len(draws_out)
    print(f"\n사후 샘플 저장: {csv_out}")
    print(f"  {total_draws} draws × {len(draws_out.columns)} columns")

    # ── 진단 텍스트 저장 ──────────────────────────────────────
    diag_file = os.path.join(OUT_DIR, f'ss_mcmc_diagnostics_N{N}_{scenario}.txt')
    with open(diag_file, 'w', encoding='utf-8') as f:
        f.write(f"PCM-SEM MCMC 진단 리포트\n")
        f.write(f"N={N}, scenario={scenario}, elapsed={elapsed:.1f}s\n")
        f.write(f"chains={CHAINS}, warmup={ITER_WARMUP}, sampling={ITER_SAMPLING}\n")
        f.write(f"adapt_delta={ADAPT_DELTA}, max_treedepth={MAX_TREEDEPTH}\n\n")
        f.write("=== 파라미터 요약 ===\n")
        f.write(key_rows.to_string())
        f.write("\n\n=== 전체 파라미터 요약 ===\n")
        f.write(summary.to_string())
    print(f"진단 저장: {diag_file}")

    return fit, draws_out

def _estimate_time(N):
    if N <= 50:   return "약 10~20분"
    if N <= 86:   return "약 20~40분"
    if N <= 100:  return "약 25~50분"
    if N <= 200:  return "약 60~120분"
    return "약 120분 이상"

# ─────────────────────────────────────────────────────────────────
# 사후 샘플 분석 (MCMC CSV 로드 후 통계 계산)
# ─────────────────────────────────────────────────────────────────
def analyze_mcmc_samples(N=86, scenario='medium'):
    """
    저장된 MCMC 샘플 CSV를 로드하여 베이지안 통계 요약을 출력.
    분석자(Claude)가 CSV 수신 후 이 함수를 호출하여 결과를 확인.
    """
    csv_file = os.path.join(OUT_DIR, f'ss_mcmc_samples_N{N}_{scenario}.csv')
    if not os.path.exists(csv_file):
        print(f"파일 없음: {csv_file}")
        return None

    df = pd.read_csv(csv_file)
    tp = EFFECT_SCENARIOS[scenario]
    true_vals = {
        'b1':  tp['beta1'],  'b2':  tp['beta2'],
        'g1':  tp['gamma1'], 'indirect_effect': tp['beta1']*tp['beta2'],
        'total_effect': tp['gamma1']+tp['beta1']*tp['beta2'],
    }

    print(f"\n=== MCMC 사후 분포 분석: N={N}, scenario={scenario} ===")
    print(f"{'파라미터':22s} {'진값':>7s} {'사후평균':>9s} {'95%CI 하한':>11s} {'95%CI 상한':>11s} {'편향':>8s} {'포함':>5s}")
    print("-" * 78)
    for pname, tv in true_vals.items():
        if pname in df.columns:
            s = df[pname].values
            lo, hi = np.percentile(s, [2.5, 97.5])
            bias = s.mean() - tv
            cov  = "✓" if lo <= tv <= hi else "✗"
            print(f"  {pname:20s} {tv:7.4f} {s.mean():9.4f} {lo:11.4f} {hi:11.4f} {bias:8.4f} {cov:>5s}")

    print(f"\n간접 효과 유의성 (95% CI 내 0 포함 여부):")
    if 'indirect_effect' in df.columns:
        s_ind = df['indirect_effect'].values
        lo, hi = np.percentile(s_ind, [2.5, 97.5])
        sig = "유의" if (lo > 0 or hi < 0) else "비유의"
        prob_pos = (s_ind > 0).mean()
        print(f"  95% CI: [{lo:.4f}, {hi:.4f}] → {sig}")
        print(f"  P(indirect > 0 | data) = {prob_pos:.4f}")

    return df

# ─────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PCM-SEM MCMC 실행기')
    parser.add_argument('--n',        type=int,  default=86,
                        help='표본 크기 (기본 86)')
    parser.add_argument('--scenario', type=str,  default='medium',
                        choices=['small','medium'], help='시나리오 (기본 medium)')
    parser.add_argument('--all',      action='store_true',
                        help='N=50, 86, 200 세 조건 모두 실행')
    parser.add_argument('--analyze',  action='store_true',
                        help='기존 CSV 파일 분석만 실행 (MCMC 없이)')
    parser.add_argument('--chains',   type=int,  default=CHAINS)
    parser.add_argument('--warmup',   type=int,  default=ITER_WARMUP)
    parser.add_argument('--sampling', type=int,  default=ITER_SAMPLING)
    args = parser.parse_args()

    if args.analyze:
        # MCMC 없이 기존 CSV 분석
        analyze_mcmc_samples(N=args.n, scenario=args.scenario)
        sys.exit(0)

    if not os.path.exists(STAN_FILE):
        print(f"ERROR: Stan 파일 없음: {STAN_FILE}")
        sys.exit(1)

    print("Stan 모델 컴파일 중...")
    model = CmdStanModel(stan_file=STAN_FILE)
    print("컴파일 완료.\n")

    # 전역 설정 업데이트
    CHAINS        = args.chains
    ITER_WARMUP   = args.warmup
    ITER_SAMPLING = args.sampling

    conditions = [(args.n, args.scenario)]
    if args.all:
        conditions = [(50, 'medium'), (86, 'medium'), (200, 'medium')]

    print(f"실행 조건: {conditions}")
    print(f"MCMC 설정: {CHAINS} chains × {ITER_WARMUP}+{ITER_SAMPLING} iterations")
    print(f"예상 총 소요 시간: {len(conditions)} 조건 × 조건별 시간\n")

    for N, scenario in conditions:
        try:
            fit, draws = run_mcmc(model, N, scenario, verbose=True)
            print(f"\n✓ N={N} {scenario} 완료")
        except Exception as e:
            print(f"\n✗ N={N} {scenario} 실패: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "="*60)
    print("모든 MCMC 완료.")
    print("\n생성된 파일:")
    for N, scenario in conditions:
        for suffix in ['samples', 'data', 'diagnostics']:
            if suffix == 'data':
                fname = f'ss_mcmc_data_N{N}_{scenario}.json'
            elif suffix == 'diagnostics':
                fname = f'ss_mcmc_diagnostics_N{N}_{scenario}.txt'
            else:
                fname = f'ss_mcmc_samples_N{N}_{scenario}.csv'
            fpath = os.path.join(OUT_DIR, fname)
            if os.path.exists(fpath):
                size = os.path.getsize(fpath) / 1024
                print(f"  {fname}  ({size:.0f} KB)")

    print("\n다음 단계:")
    print("  1. ss_mcmc_samples_N*_*.csv 파일을 Claude에게 전달")
    print("  2. Claude가 analyze_mcmc_samples() 로 베이지안 분석 수행")
    print("  3. 논문에 풀 베이지안 결과 반영")
