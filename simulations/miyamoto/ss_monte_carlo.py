#!/usr/bin/env python3
"""
ss_monte_carlo.py
=================
Monte Carlo 시뮬레이션: Composite Score OLS vs. Latent Variable Oracle OLS
비교를 통해 PCM-SEM의 방법론적 우위를 검증한다.

저장 파일: ss_results.csv
"""

import numpy as np
import pandas as pd
import time
import os

SEED      = 2024
N_REPS    = 500          # 조건당 반복 횟수
OUT_DIR   = os.path.dirname(os.path.abspath(__file__))
CSV_OUT   = os.path.join(OUT_DIR, 'ss_results.csv')

# ─────────────────────────────────────────────────────────
# 시뮬레이션 조건 정의
# ─────────────────────────────────────────────────────────
SAMPLE_SIZES = [50, 86, 100, 150, 200, 300]

# 두 가지 효과 크기 시나리오
EFFECT_SCENARIOS = {
    'small':  dict(beta1=0.30, beta2=0.30, gamma1=0.00,
                   gamma_M=0.10, gamma_Y=0.05, alpha_M=-0.15, alpha_Y=-0.20),
    'medium': dict(beta1=0.50, beta2=0.40, gamma1=0.20,
                   gamma_M=0.15, gamma_Y=0.10, alpha_M=-0.20, alpha_Y=-0.30),
}

# 문항 구성 (21문항, 5점 척도)
I_X, I_M, I_Y, I, K = 4, 11, 6, 21, 5

# ─────────────────────────────────────────────────────────
# PCM 데이터 생성 함수 (완전 벡터화)
# ─────────────────────────────────────────────────────────
def pcm_sample_vectorized(theta, deltas, rng):
    """
    N명의 학생에 대해 하나의 문항의 응답을 동시에 샘플링.
    Gumbel-max trick으로 완전 벡터화.
    theta:  (N,)
    deltas: (K-1,) = 4개
    returns: (N,) integers 1..K
    """
    N  = len(theta)
    lp = np.zeros((N, K))
    for k in range(1, K):
        lp[:, k] = lp[:, k-1] + (theta - deltas[k-1])
    lp -= lp.max(axis=1, keepdims=True)
    gumbel = -np.log(-np.log(rng.uniform(size=(N, K)) + 1e-15))
    return np.argmax(lp + gumbel, axis=1) + 1   # 1-indexed

def generate_data(N, tp, rng, item_deltas=None):
    """
    구조 방정식 + PCM으로 시뮬레이션 데이터 생성.
    tp: true parameters dict
    Returns: (y_data (N×I), theta_X, theta_M, theta_Y, gender)
    """
    # 성별 (0/1)
    gender  = (rng.uniform(size=N) < 0.60).astype(float)
    # 잠재 변수
    tX = rng.standard_normal(N)
    tM = tp['alpha_M'] + tp['beta1']*tX  + tp['gamma_M']*gender + rng.standard_normal(N)
    tY = tp['alpha_Y'] + tp['gamma1']*tX + tp['beta2']*tM + tp['gamma_Y']*gender + rng.standard_normal(N)

    # 문항 임계값 (한 번 생성 후 재사용)
    if item_deltas is None:
        item_deltas = _make_item_deltas(rng, tp)

    # 응답 행렬
    y = np.zeros((N, I), dtype=np.int32)
    for i in range(I):
        if   i < I_X:      theta_use = tX
        elif i < I_X+I_M:  theta_use = tM
        else:               theta_use = tY
        y[:, i] = pcm_sample_vectorized(theta_use, item_deltas[i], rng)

    return y, tX, tM, tY, gender

def _make_item_deltas(rng, tp, c_X=-1.0, c_M=0.3, c_Y=0.5):
    """
    각 문항의 PCM 임계값 생성 (고정 오프셋 + 소음).
    원논문 평균을 대략 재현하는 오프셋 사용.
    """
    base = np.array([-1.5, -0.5, 0.5, 1.5])
    deltas = []
    for _ in range(I_X): deltas.append(c_X + base + rng.normal(0, 0.2, 4))
    for _ in range(I_M): deltas.append(c_M + base + rng.normal(0, 0.2, 4))
    for _ in range(I_Y): deltas.append(c_Y + base + rng.normal(0, 0.2, 4))
    return deltas

# ─────────────────────────────────────────────────────────
# OLS 추정 함수
# ─────────────────────────────────────────────────────────
def ols_with_se(X_mat, y):
    """
    OLS: β̂ = (X'X)^{-1} X'y, SE = sqrt(σ² diag((X'X)^{-1}))
    Returns: coef (p,), se (p,), residual_var
    """
    n, p = X_mat.shape
    try:
        XtX_inv = np.linalg.inv(X_mat.T @ X_mat)
    except np.linalg.LinAlgError:
        XtX_inv = np.linalg.pinv(X_mat.T @ X_mat)
    coef  = XtX_inv @ (X_mat.T @ y)
    resid = y - X_mat @ coef
    s2    = resid @ resid / max(n - p, 1)
    se    = np.sqrt(np.maximum(s2 * np.diag(XtX_inv), 1e-15))
    return coef, se, s2

def sobel_indirect_ci(b1, b2, se_b1, se_b2, z=1.96):
    """
    Sobel (1982) 델타법으로 간접 효과의 95% CI 계산.
    SE_indirect = sqrt(b2²·se_b1² + b1²·se_b2²)
    """
    se_ind = np.sqrt(b2**2 * se_b1**2 + b1**2 * se_b2**2)
    ind    = b1 * b2
    return ind, ind - z*se_ind, ind + z*se_ind

def estimate_sem(X_, M_, Y_, G, label):
    """
    두 단계 OLS로 SEM 경로 계수 추정.
    M ~ alpha_M + b1*X + gamma_M*G
    Y ~ alpha_Y + g1*X + b2*M + gamma_Y*G
    label: 'comp' or 'oracle'
    Returns: dict with all estimates and coverage bounds
    """
    N = len(X_)
    ones = np.ones(N)
    # --- 방정식 1: M ---
    Xd_M  = np.column_stack([ones, X_, G])
    c_M, se_M, _ = ols_with_se(Xd_M, M_)
    a_M, b1, gM  = c_M
    # --- 방정식 2: Y ---
    Xd_Y  = np.column_stack([ones, X_, M_, G])
    c_Y, se_Y, _ = ols_with_se(Xd_Y, Y_)
    a_Y, g1, b2, gY = c_Y
    # --- 간접 효과 ---
    ind, ind_lo, ind_hi = sobel_indirect_ci(b1, b2, se_M[1], se_Y[2])
    tot  = g1 + ind
    return {
        f'b1_{label}':    b1,   f'b1_{label}_lo': b1 - 1.96*se_M[1],   f'b1_{label}_hi': b1 + 1.96*se_M[1],
        f'b2_{label}':    b2,   f'b2_{label}_lo': b2 - 1.96*se_Y[2],   f'b2_{label}_hi': b2 + 1.96*se_Y[2],
        f'g1_{label}':    g1,   f'g1_{label}_lo': g1 - 1.96*se_Y[1],   f'g1_{label}_hi': g1 + 1.96*se_Y[1],
        f'ind_{label}':   ind,  f'ind_{label}_lo': ind_lo,               f'ind_{label}_hi': ind_hi,
        f'tot_{label}':   tot,
    }

# ─────────────────────────────────────────────────────────
# 단일 반복 실행
# ─────────────────────────────────────────────────────────
def run_one(N, tp, item_deltas, rng, scenario_name):
    y, tX, tM, tY, gender = generate_data(N, tp, rng, item_deltas)

    # 합산 평균 점수 (Composite Score)
    cX = y[:, :I_X].mean(1)
    cM = y[:, I_X:I_X+I_M].mean(1)
    cY = y[:, I_X+I_M:].mean(1)

    comp   = estimate_sem(cX, cM, cY,  gender, 'comp')
    oracle = estimate_sem(tX, tM, tY,  gender, 'oracle')

    # 참값
    tv = dict(b1=tp['beta1'], b2=tp['beta2'], g1=tp['gamma1'],
              ind=tp['beta1']*tp['beta2'],
              tot=tp['gamma1']+tp['beta1']*tp['beta2'])

    row = dict(N=N, scenario=scenario_name,
               **{k: tp[k] for k in ['beta1','beta2','gamma1']})
    row.update(comp); row.update(oracle)

    # 커버리지·검정력 지표
    for par in ['b1','b2','g1','ind']:
        for mth in ['comp','oracle']:
            lo = row[f'{par}_{mth}_lo']; hi = row[f'{par}_{mth}_hi']
            row[f'{par}_{mth}_cov']   = int(lo <= tv[par] <= hi)
            row[f'{par}_{mth}_sig']   = int(lo > 0 or hi < 0)  # CI excludes 0
    return row

# ─────────────────────────────────────────────────────────
# 메인 시뮬레이션 루프
# ─────────────────────────────────────────────────────────
def main():
    rng  = np.random.default_rng(SEED)
    rows = []
    total_cond = len(SAMPLE_SIZES) * len(EFFECT_SCENARIOS)
    cond_idx   = 0
    t_start    = time.time()

    for scenario_name, tp in EFFECT_SCENARIOS.items():
        # 문항 임계값: 시나리오당 고정 (반복 간 동일)
        item_deltas = _make_item_deltas(rng, tp)

        for N in SAMPLE_SIZES:
            cond_idx += 1
            t0 = time.time()
            for _ in range(N_REPS):
                rows.append(run_one(N, tp, item_deltas, rng, scenario_name))
            elapsed = time.time() - t0
            print(f"[{cond_idx:2d}/{total_cond}] scenario={scenario_name}, N={N:3d}  "
                  f"reps={N_REPS}  time={elapsed:.1f}s")

    df = pd.DataFrame(rows)
    df.to_csv(CSV_OUT, index=False)
    total_time = time.time() - t_start
    print(f"\nDone. {len(df)} rows saved → {CSV_OUT}  (total {total_time:.1f}s)")
    return df

if __name__ == '__main__':
    df = main()
    # Quick summary
    agg = df.groupby(['scenario','N']).agg(
        b1_comp_bias  = ('b1_comp',   lambda x: (x - df.loc[x.index,'beta1']).mean()),
        b1_oracle_bias= ('b1_oracle', lambda x: (x - df.loc[x.index,'beta1']).mean()),
        b1_comp_cov   = ('b1_comp_cov',   'mean'),
        b1_oracle_cov = ('b1_oracle_cov', 'mean'),
        ind_comp_pwr  = ('ind_comp_sig',   'mean'),
        ind_oracle_pwr= ('ind_oracle_sig', 'mean'),
    ).round(3)
    print("\n=== Quick Summary ===")
    print(agg.to_string())
