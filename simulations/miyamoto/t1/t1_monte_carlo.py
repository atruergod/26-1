#!/usr/bin/env python3
"""
t1_monte_carlo.py
=================
PCM-SEM 방법론 우위 검증: 합산 점수 OLS vs 잠재 변수 오라클 OLS
몬테카를로 시뮬레이션 (CmdStan 불필요)

비교 방법:
  CS-OLS  — composite-score OLS (합산 점수, 측정 오차 무시)
  LV-OLS  — latent-variable OLS (잠재 변수 직접 사용, PCM-SEM 이론 상한)

조건:
  표본 크기 N: 50, 86, 100, 150, 200, 300
  효과 시나리오: small (β₁=0.30), medium (β₁=0.50)
  반복 횟수: 500회 per condition

산출 파일 (t1/ 폴더):
  t1_mc_results.csv   — 반복별 원시 추정치
  t1_mc_summary.csv   — 조건별 편향/RMSE/포함확률/검출력 요약

실행: python t1_monte_carlo.py
"""

import os
import time
import numpy as np
import pandas as pd
from scipy import stats

# ─── 설정 ──────────────────────────────────────────────────────────────────
SEED         = 2024
N_REPS       = 500
SAMPLE_SIZES = [50, 86, 100, 150, 200, 300]
OUT_DIR      = os.path.dirname(os.path.abspath(__file__))

SCENARIOS = {
    'small':  dict(beta1=0.30, beta2=0.30, gamma1=0.00,
                   gamma_M=0.10, gamma_Y=0.05),
    'medium': dict(beta1=0.50, beta2=0.40, gamma1=0.20,
                   gamma_M=0.15, gamma_Y=0.10),
}

# 문항 구성 (원논문과 동일)
I_X, I_M, I_Y, I, K = 4, 11, 6, 21, 5

# PCM 임계값 오프셋 (원논문 평균 기반 보정값)
C_X, C_M, C_Y = -1.194, -0.380, -0.185

# ─── PCM 유틸리티 ──────────────────────────────────────────────────────────
def pcm_probs_vec(theta, c_offset, K=5):
    """N × K 확률 행렬 반환."""
    base = np.array([-1.5, -0.5, 0.5, 1.5])
    deltas = base + c_offset          # (K-1,)
    # log unnorm: shape (N, K)
    lp = np.zeros((len(theta), K))
    for k in range(1, K):
        lp[:, k] = lp[:, k-1] + (theta - deltas[k-1])
    lp -= lp.max(axis=1, keepdims=True)
    p = np.exp(lp)
    return p / p.sum(axis=1, keepdims=True)

def sample_pcm(theta, c_offset, rng, K=5):
    """각 응답자에 대해 PCM 범주 표집 (1-indexed)."""
    probs = pcm_probs_vec(theta, c_offset, K)
    cumP  = np.cumsum(probs, axis=1)
    u     = rng.uniform(size=(len(theta), 1))
    return (u > cumP).sum(axis=1) + 1   # 1…K

def composite_score(responses, n_items):
    """단순 합산 점수 (평균)."""
    return responses.mean(axis=1)

# ─── 데이터 생성 ───────────────────────────────────────────────────────────
def generate_data(N, tp, rng, c_x=C_X, c_m=C_M, c_y=C_Y):
    """잠재 변수 + PCM 반응 행렬 생성."""
    gender = (rng.uniform(size=N) < 0.686).astype(float)
    tX = rng.standard_normal(N)
    tM = (tp['gamma_M'] * gender + tp['beta1'] * tX
          + rng.standard_normal(N))
    tY = (tp['gamma_Y'] * gender + tp['gamma1'] * tX
          + tp['beta2'] * tM + rng.standard_normal(N))

    # PCM 응답 행렬 (문항별 오프셋 변이 추가)
    Y_mat = np.zeros((N, I), dtype=int)
    for i in range(I_X):
        di = c_x + rng.normal(0, 0.20)
        Y_mat[:, i] = sample_pcm(tX, di, rng)
    for i in range(I_M):
        di = c_m + rng.normal(0, 0.20)
        Y_mat[:, I_X + i] = sample_pcm(tM, di, rng)
    for i in range(I_Y):
        di = c_y + rng.normal(0, 0.20)
        Y_mat[:, I_X + I_M + i] = sample_pcm(tY, di, rng)

    return Y_mat, tX, tM, tY, gender

# ─── OLS 추정 ──────────────────────────────────────────────────────────────
def ols_coeffs(X, M, Y, G):
    """
    모형: M = a0 + b1*X + gM*G + e
          Y = a1 + g1*X + b2*M + gY*G + e
    반환: (b1, b2, g1, indirect_b1b2, se_b1, se_b2, se_g1)
    """
    N = len(X)
    # M 방정식
    Xm = np.column_stack([np.ones(N), X, G])
    bM = np.linalg.lstsq(Xm, M, rcond=None)[0]
    resM = M - Xm @ bM
    s2M = (resM**2).sum() / (N - 3)
    covM = s2M * np.linalg.inv(Xm.T @ Xm)
    b1, se_b1 = bM[1], np.sqrt(covM[1, 1])

    # Y 방정식
    Xy = np.column_stack([np.ones(N), X, M, G])
    bY = np.linalg.lstsq(Xy, Y, rcond=None)[0]
    resY = Y - Xy @ bY
    s2Y = (resY**2).sum() / (N - 4)
    covY = s2Y * np.linalg.inv(Xy.T @ Xy)
    g1, b2 = bY[1], bY[2]
    se_g1, se_b2 = np.sqrt(covY[1, 1]), np.sqrt(covY[2, 2])

    indirect = b1 * b2
    return b1, b2, g1, indirect, se_b1, se_b2, se_g1

def ci_contains(est, se, true_val, alpha=0.05):
    z = stats.norm.ppf(1 - alpha / 2)
    return int(est - z * se <= true_val <= est + z * se)

# ─── 메인 시뮬레이션 ───────────────────────────────────────────────────────
def run_simulation():
    rng = np.random.default_rng(SEED)
    records = []
    total = len(SAMPLE_SIZES) * len(SCENARIOS) * N_REPS
    done = 0
    t0 = time.time()

    for sc_name, tp in SCENARIOS.items():
        for N in SAMPLE_SIZES:
            for rep in range(N_REPS):
                Y_mat, tX, tM, tY, G = generate_data(N, tp, rng)

                # 합산 점수
                csX = composite_score(Y_mat[:, :I_X], I_X)
                csM = composite_score(Y_mat[:, I_X:I_X+I_M], I_M)
                csY = composite_score(Y_mat[:, I_X+I_M:], I_Y)

                # CS-OLS
                (b1c, b2c, g1c, indc,
                 se_b1c, se_b2c, se_g1c) = ols_coeffs(csX, csM, csY, G)

                # LV-OLS (oracle: 잠재 변수 직접 사용)
                (b1l, b2l, g1l, indl,
                 se_b1l, se_b2l, se_g1l) = ols_coeffs(tX, tM, tY, G)

                records.append(dict(
                    scenario=sc_name, N=N, rep=rep,
                    true_b1=tp['beta1'], true_b2=tp['beta2'],
                    true_g1=tp['gamma1'],
                    true_ind=tp['beta1'] * tp['beta2'],
                    # CS-OLS
                    cs_b1=b1c, cs_b2=b2c, cs_g1=g1c, cs_ind=indc,
                    cs_b1_cov=ci_contains(b1c, se_b1c, tp['beta1']),
                    cs_b2_cov=ci_contains(b2c, se_b2c, tp['beta2']),
                    cs_g1_cov=ci_contains(g1c, se_g1c, tp['gamma1']),
                    cs_ind_sig=int(abs(indc) / (abs(b1c)*se_b2c + abs(b2c)*se_b1c + 1e-9) > 1.96),
                    # LV-OLS
                    lv_b1=b1l, lv_b2=b2l, lv_g1=g1l, lv_ind=indl,
                    lv_b1_cov=ci_contains(b1l, se_b1l, tp['beta1']),
                    lv_b2_cov=ci_contains(b2l, se_b2l, tp['beta2']),
                    lv_g1_cov=ci_contains(g1l, se_g1l, tp['gamma1']),
                    lv_ind_sig=int(abs(indl) / (abs(b1l)*se_b2l + abs(b2l)*se_b1l + 1e-9) > 1.96),
                ))

                done += 1
                if done % 500 == 0:
                    elapsed = time.time() - t0
                    print(f"  {done}/{total} ({100*done/total:.0f}%)  {elapsed:.0f}s  "
                          f"[{sc_name} N={N}]")

    return pd.DataFrame(records)

def summarise(df):
    rows = []
    for (sc, N), g in df.groupby(['scenario', 'N']):
        tb1 = g['true_b1'].iloc[0]
        tb2 = g['true_b2'].iloc[0]
        tind = g['true_ind'].iloc[0]
        rows.append(dict(
            scenario=sc, N=N,
            # CS-OLS
            cs_b1_bias=(g['cs_b1'] - tb1).mean(),
            cs_b1_rmse=np.sqrt(((g['cs_b1'] - tb1)**2).mean()),
            cs_b1_cov=g['cs_b1_cov'].mean(),
            cs_b2_bias=(g['cs_b2'] - tb2).mean(),
            cs_b2_rmse=np.sqrt(((g['cs_b2'] - tb2)**2).mean()),
            cs_b2_cov=g['cs_b2_cov'].mean(),
            cs_ind_bias=(g['cs_ind'] - tind).mean(),
            cs_ind_power=g['cs_ind_sig'].mean(),
            # LV-OLS
            lv_b1_bias=(g['lv_b1'] - tb1).mean(),
            lv_b1_rmse=np.sqrt(((g['lv_b1'] - tb1)**2).mean()),
            lv_b1_cov=g['lv_b1_cov'].mean(),
            lv_b2_bias=(g['lv_b2'] - tb2).mean(),
            lv_b2_rmse=np.sqrt(((g['lv_b2'] - tb2)**2).mean()),
            lv_b2_cov=g['lv_b2_cov'].mean(),
            lv_ind_bias=(g['lv_ind'] - tind).mean(),
            lv_ind_power=g['lv_ind_sig'].mean(),
        ))
    return pd.DataFrame(rows)

if __name__ == '__main__':
    print("=" * 60)
    print("t1 몬테카를로 시뮬레이션")
    print(f"조건: {len(SAMPLE_SIZES)} 표본 크기 × {len(SCENARIOS)} 시나리오 × {N_REPS} 반복")
    print("=" * 60)
    df = run_simulation()
    raw_path = os.path.join(OUT_DIR, 't1_mc_results.csv')
    df.to_csv(raw_path, index=False)
    print(f"\n원시 결과 저장: {raw_path}  ({len(df):,} 행)")

    sm = summarise(df)
    sum_path = os.path.join(OUT_DIR, 't1_mc_summary.csv')
    sm.to_csv(sum_path, index=False)
    print(f"요약 결과 저장: {sum_path}  ({len(sm)} 행)")

    # 주요 결과 출력
    print("\n[medium 시나리오 β₁ 편향 비교]")
    med = sm[sm['scenario'] == 'medium'][['N', 'cs_b1_bias', 'lv_b1_bias',
                                           'cs_b1_cov', 'lv_b1_cov']]
    print(med.to_string(index=False, float_format=lambda x: f'{x:+.3f}'))
