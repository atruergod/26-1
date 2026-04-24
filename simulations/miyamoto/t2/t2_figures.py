#!/usr/bin/env python3
"""
t2_figures.py
=============
t2 논문용 그림 생성 (t2_analysis.py 실행 후 사용).

그림 1 (t2_fig_posterior.png)      — 경로 계수 사후 분포 (약한 vs 강한)
그림 2 (t2_fig_path_estimates.png) — 경로 계수 점추정 + 95% CrI
그림 3 (t2_fig_mediation.png)      — 간접 효과 사후 분포 및 매개 비율
"""
import os, sys, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
warnings.filterwarnings('ignore')

HERE  = os.path.dirname(os.path.abspath(__file__))
BLUE  = '#2166ac'
ORANGE = '#d6604d'

plt.rcParams.update({'font.family': 'DejaVu Sans',
                     'axes.unicode_minus': False, 'figure.dpi': 150})

def load_samples(fname):
    path = os.path.join(HERE, fname)
    if not os.path.exists(path):
        print(f"  파일 없음: {path}")
        return None
    return pd.read_csv(path)

weak   = load_samples('t2_mcmc_samples.csv')
strong = load_samples('t2_mcmc_strong_samples.csv')

if weak is None or strong is None:
    print("MCMC 샘플 파일이 없습니다. t2_analysis.py 를 먼저 실행하세요.")
    sys.exit(0)

for df in [weak, strong]:
    if 'indirect' not in df.columns and 'beta1' in df.columns:
        df['indirect'] = df['beta1'] * df['beta2']
    if 'total' not in df.columns and 'gamma1' in df.columns:
        df['total'] = df['gamma1'] + df['indirect']

# ─── 그림 1: 사후 분포 ────────────────────────────────────────────────────
PARAMS = [('beta1','β₁  X→M'), ('beta2','β₂  M→Y'),
          ('gamma1','γ₁  X→Y 직접'), ('indirect','간접효과 β₁β₂')]

fig, axes = plt.subplots(1, 4, figsize=(16, 4))
fig.suptitle('중국인 유학생 쓰기 태도 PCM-SEM 경로 계수 사후 분포 (N=?)',
             fontsize=12, fontweight='bold')

for ax, (col, lbl) in zip(axes, PARAMS):
    for df, color, name in [(weak, BLUE, '약한 사전'), (strong, ORANGE, '강한 사전')]:
        if col not in df.columns: continue
        v = df[col].dropna()
        ax.hist(v, bins=60, density=True, alpha=0.45, color=color)
        lo, hi = np.percentile(v, [2.5, 97.5])
        ax.axvline(v.mean(), color=color, lw=1.8, label=name)
        ax.axvspan(lo, hi, alpha=0.12, color=color)
    ax.axvline(0, color='black', lw=1.0, ls='--', alpha=0.5)
    ax.set_title(lbl, fontsize=10)
    ax.set_xlabel('추정값', fontsize=9)
    ax.grid(True, alpha=0.25)

handles = [mpatches.Patch(color=BLUE, label='약한 사전 분포'),
           mpatches.Patch(color=ORANGE, label='강한 사전 분포')]
fig.legend(handles=handles, loc='lower center', ncol=2,
           fontsize=9, bbox_to_anchor=(0.5, -0.06))
plt.tight_layout()
out = os.path.join(HERE, 't2_fig_posterior.png')
plt.savefig(out, dpi=200, bbox_inches='tight', facecolor='white')
print(f"저장: {out}")
plt.close()

# ─── 그림 2: 점추정 + 95% CrI 비교 ──────────────────────────────────────
all_params = ['beta1','beta2','gamma1','gamma_M','gamma_Y','indirect']
labels_kr  = ['β₁ X→M','β₂ M→Y','γ₁ X→Y','γM 성별→M','γY 성별→Y','간접효과']

fig, ax = plt.subplots(figsize=(10, 5))
y_pos = np.arange(len(all_params))

for df, color, name, offset in [(weak, BLUE, '약한 사전', 0.15),
                                  (strong, ORANGE, '강한 사전', -0.15)]:
    means, lo_errs, hi_errs = [], [], []
    for p in all_params:
        v = df[p].dropna() if p in df.columns else pd.Series([0])
        m = v.mean()
        lo, hi = np.percentile(v, [2.5, 97.5])
        means.append(m); lo_errs.append(m - lo); hi_errs.append(hi - m)
    ax.errorbar(means, y_pos + offset, xerr=[lo_errs, hi_errs],
                fmt='o', color=color, lw=1.5, capsize=4, label=name, ms=6)

ax.axvline(0, color='black', lw=0.8, ls='--', alpha=0.5)
ax.set_yticks(y_pos)
ax.set_yticklabels(labels_kr, fontsize=11)
ax.set_xlabel('사후 평균 및 95% 신용 구간', fontsize=11)
ax.set_title('중국인 유학생 PCM-SEM 경로 계수 추정', fontsize=12, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3, axis='x')
plt.tight_layout()
out = os.path.join(HERE, 't2_fig_path_estimates.png')
plt.savefig(out, dpi=200, bbox_inches='tight', facecolor='white')
print(f"저장: {out}")
plt.close()

# ─── 그림 3: 간접 효과 + 매개 비율 ──────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
fig.suptitle('간접 효과 및 완전 매개 검증', fontsize=12, fontweight='bold')

# 간접 효과 사후 분포
ax = axes[0]
for df, color, name in [(weak, BLUE, '약한 사전'), (strong, ORANGE, '강한 사전')]:
    if 'indirect' not in df.columns: continue
    v = df['indirect'].dropna()
    ax.hist(v, bins=60, density=True, alpha=0.45, color=color, label=name)
    ax.axvline(v.mean(), color=color, lw=1.8)
ax.axvline(0, color='black', lw=1.0, ls='--', alpha=0.5)
ax.set_title('간접 효과 β₁β₂ 사후 분포', fontsize=11)
ax.set_xlabel('간접 효과', fontsize=10)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.25)

# P(β₁>0 & β₂>0) 막대
ax = axes[1]
probs = []
names_disp = []
for df, name in [(weak, '약한 사전'), (strong, '강한 사전')]:
    if 'beta1' in df.columns and 'beta2' in df.columns:
        p = ((df['beta1'] > 0) & (df['beta2'] > 0)).mean()
        probs.append(p)
        names_disp.append(name)

bars = ax.bar(names_disp, probs, color=[BLUE, ORANGE], alpha=0.75, width=0.4)
ax.axhline(0.95, color='black', ls='--', lw=1.0, alpha=0.5, label='0.95 기준선')
for bar, p in zip(bars, probs):
    ax.text(bar.get_x() + bar.get_width()/2, p + 0.01,
            f'{p:.3f}', ha='center', fontsize=11, fontweight='bold')
ax.set_ylim(0, 1.05)
ax.set_ylabel('확률', fontsize=10)
ax.set_title('P(β₁>0 ∧ β₂>0 | data)\nX→M→Y 인과 방향 확률', fontsize=11)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
out = os.path.join(HERE, 't2_fig_mediation.png')
plt.savefig(out, dpi=200, bbox_inches='tight', facecolor='white')
print(f"저장: {out}")
plt.close()
print("t2_figures.py 완료.")
