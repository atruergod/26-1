#!/usr/bin/env python3
"""
t2_model_check.py
=================
t2 연구 사전 준비:
  (1) PCM 임계값 보정 — t2 데이터 수집 전, 모형 초기값 산출
  (2) 사전 예측 분포(prior predictive check) — 모형이 현실적 데이터를 생성하는지 확인
  (3) t2_calibration.json 저장 — t2_analysis.py 에서 사용

실행: python t2_model_check.py  (CmdStan 불필요)
"""
import json, os, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')

HERE = os.path.dirname(os.path.abspath(__file__))
SEED = 2024
rng  = np.random.default_rng(SEED)

I_X, I_M, I_Y, I, K = 4, 11, 6, 21, 5

# ─── 임계값 보정 ──────────────────────────────────────────────────────────
def pcm_expected(c, K=5):
    base   = np.array([-1.5, -0.5, 0.5, 1.5]) + c
    lp     = np.zeros(K)
    for k in range(1, K):
        lp[k] = lp[k-1] + (0.0 - base[k-1])
    lp -= lp.max(); p = np.exp(lp); p /= p.sum()
    return np.dot(np.arange(1, K+1), p)

def calibrate_c(target):
    lo, hi = -5.0, 5.0
    for _ in range(80):
        mid = (lo + hi) / 2
        (hi if pcm_expected(mid) < target else lo).__class__
        if pcm_expected(mid) < target: hi = mid
        else:                          lo = mid
    return mid

# 원논문 통계 (주월랑, 2022) — 수정된 Y 평균 3.171
PAPER = dict(X_mean=4.015, M_mean=3.348, Y_mean=3.171,
             X_sd=0.642, M_sd=0.539, Y_sd=0.585,
             N=86, n_female=59, cronbach=0.854)

calib = {k: calibrate_c(PAPER[k]) for k in ['X_mean', 'M_mean', 'Y_mean']}
calib_named = {'c_X': calib['X_mean'],
               'c_M': calib['M_mean'],
               'c_Y': calib['Y_mean']}

print("─── PCM 임계값 보정 결과 ───")
for nm, key, c_key in [('X(쓰기인식)', 'X_mean', 'c_X'),
                        ('M(쓰기반응)', 'M_mean', 'c_M'),
                        ('Y(수행태도)', 'Y_mean', 'c_Y')]:
    v = pcm_expected(calib_named[c_key])
    print(f"  {nm}: 목표={PAPER[key]:.3f}  c={calib_named[c_key]:+.3f}  "
          f"검증 E[y|θ=0]={v:.3f}")

# ─── 사전 예측 분포 ───────────────────────────────────────────────────────
def pcm_probs(theta_scalar, c, K=5):
    base = np.array([-1.5, -0.5, 0.5, 1.5]) + c
    lp = np.zeros(K)
    for k in range(1, K): lp[k] = lp[k-1] + (theta_scalar - base[k-1])
    lp -= lp.max(); p = np.exp(lp); return p / p.sum()

def prior_predictive(n_samples=500):
    """사전 분포에서 파라미터 표집 후 데이터 생성."""
    records = []
    TRUE = dict(beta1=0.40, beta2=0.35, gamma1=0.10,
                gamma_M=0.15, gamma_Y=0.08)
    c_X, c_M, c_Y = calib_named['c_X'], calib_named['c_M'], calib_named['c_Y']

    for _ in range(n_samples):
        N = 100
        gender = (rng.uniform(size=N) < 0.55).astype(float)
        tX = rng.standard_normal(N)
        tM = TRUE['gamma_M']*gender + TRUE['beta1']*tX + rng.standard_normal(N)
        tY = TRUE['gamma_Y']*gender + TRUE['gamma1']*tX + TRUE['beta2']*tM + rng.standard_normal(N)

        # 합산 점수 계산
        for name, theta, c, n_it in [('X', tX, c_X, I_X),
                                      ('M', tM, c_M, I_M),
                                      ('Y', tY, c_Y, I_Y)]:
            scores = []
            for j in range(N):
                item_scores = [np.dot(np.arange(1, K+1),
                                      pcm_probs(theta[j], c + rng.normal(0, 0.2)))
                               for _ in range(n_it)]
                scores.append(np.mean(item_scores))
            records.append({'construct': name, 'mean': np.mean(scores), 'sd': np.std(scores)})

    return pd.DataFrame(records)

print("\n사전 예측 분포 계산 중 (N=100, 500 샘플)...")
pp = prior_predictive(500)

# ─── 그림: 사전 예측 vs 원논문 통계 ─────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(12, 4))
fig.suptitle('사전 예측 분포 점검: 원논문 통계와의 비교', fontsize=13, fontweight='bold')

for ax, (nm, key_m, key_s) in zip(axes, [
    ('X (쓰기인식)', 'X_mean', 'X_sd'),
    ('M (쓰기반응)', 'M_mean', 'M_sd'),
    ('Y (수행태도)', 'Y_mean', 'Y_sd'),
]):
    short = nm[0]
    d = pp[pp['construct'] == short]['mean']
    ax.hist(d, bins=40, density=True, alpha=0.6, color='#4393c3')
    ax.axvline(PAPER[key_m], color='#d6604d', lw=2.0, label=f'원논문 M={PAPER[key_m]}')
    ax.set_title(nm, fontsize=11)
    ax.set_xlabel('복합 점수 평균', fontsize=9)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
out_fig = os.path.join(HERE, 't2_prior_check.png')
plt.savefig(out_fig, dpi=180, bbox_inches='tight', facecolor='white')
print(f"  그림 저장: {out_fig}")
plt.close()

# ─── 보정값 저장 ──────────────────────────────────────────────────────────
calib_out = dict(
    c_X=round(calib_named['c_X'], 4),
    c_M=round(calib_named['c_M'], 4),
    c_Y=round(calib_named['c_Y'], 4),
    prior_beta1_mean=0.35, prior_beta1_sd=0.20,
    prior_beta2_mean=0.35, prior_beta2_sd=0.20,
    prior_gamma1_mean=0.10, prior_gamma1_sd=0.20,
    paper_X_mean=PAPER['X_mean'], paper_M_mean=PAPER['M_mean'],
    paper_Y_mean=PAPER['Y_mean'],
    note="원논문(주월랑, 2022) 통계 기반 보정값. t2_analysis.py 에서 사용."
)
cal_path = os.path.join(HERE, 't2_calibration.json')
with open(cal_path, 'w', encoding='utf-8') as f:
    json.dump(calib_out, f, ensure_ascii=False, indent=2)
print(f"  보정값 저장: {cal_path}")
print("\nt2_model_check.py 완료.")
