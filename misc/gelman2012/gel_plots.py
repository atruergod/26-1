"""
gel_plots.py
Shrinkage plot (Gelman 의 부분 풀링 효과 시각화)
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager

# Korean font
for fpath in ['/sessions/keen-wonderful-lamport/.fonts/NotoSansCJKkr-Regular.otf']:
    if os.path.exists(fpath):
        font_manager.fontManager.addfont(fpath)
plt.rcParams['font.family'] = 'Noto Sans CJK KR'
plt.rcParams['axes.unicode_minus'] = False

HERE = os.path.dirname(os.path.abspath(__file__))
shrink = pd.read_csv(os.path.join(HERE, 'gel_shrinkage.csv'))

# Subscale-wise plot
fig, axes = plt.subplots(1, 4, figsize=(15, 4))
years = ['1','2','3','4','EX']
year_labels = ['1학년','2학년','3학년','4학년','교환']

for ax, sub in zip(axes, ['쓰기인식','쓰기반응','수행태도','쓰기태도']):
    d = shrink[shrink['subscale'] == sub].set_index('year').loc[years].reset_index()
    x = np.arange(len(years))
    # raw means
    ax.plot(x - 0.15, d['raw_mean'], 'o', color='#1f77b4', markersize=8, label='Raw mean (no pooling)')
    # posterior means
    ax.plot(x + 0.15, d['post_mean'], 's', color='#d62728', markersize=8, label='Posterior mean (Bayesian shrunk)')
    # 95% credible intervals
    ax.errorbar(x + 0.15, d['post_mean'],
                yerr=[d['post_mean']-d['ci_lo'], d['ci_hi']-d['post_mean']],
                fmt='none', color='#d62728', alpha=0.4, capsize=3)
    # grand mean horizontal line
    gm = d['post_mean'].mean()
    ax.axhline(gm, color='gray', linestyle='--', alpha=0.5, label='Grand mean' if ax==axes[0] else None)
    # n labels
    for i, n in enumerate(d['n']):
        ax.text(i, d['raw_mean'].iloc[i] + 0.08, f'N={n}', ha='center', fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(year_labels, fontsize=9)
    ax.set_title(sub)
    ax.set_ylim(2.5, 4.5)
    if ax == axes[0]:
        ax.set_ylabel('점수 (mean)')
    ax.grid(alpha=0.3)

axes[0].legend(loc='lower right', fontsize=8)
fig.suptitle('학년별 표본평균 vs 베이지안 사후평균 — 부분 풀링(shrinkage) 효과', fontsize=12)
fig.tight_layout()
fig.savefig(os.path.join(HERE, 'gel_shrinkage_plot.pdf'), bbox_inches='tight')
fig.savefig(os.path.join(HERE, 'gel_shrinkage_plot.png'), bbox_inches='tight', dpi=120)
print("저장: gel_shrinkage_plot.pdf, .png")

# Posterior interval forest plot for pairwise comparisons
bayes_pair = pd.read_excel(os.path.join(HERE, 'gel_bayes_results.xlsx'),
                            sheet_name='pairwise_diffs')
fig, ax = plt.subplots(figsize=(8, 5))

# Show 수행태도 학년별 차이 vs 1학년
sub = '수행태도'
factor = 'alpha'
d = bayes_pair.query("subscale==@sub & factor==@factor & comparison.str.startswith('1 - ') | "
                     "subscale==@sub & factor==@factor & comparison.str.endswith(' - 1')",
                     engine='python')
# Get differences vs 1: 2-1, 3-1, 4-1, EX-1
import json
labels = []
means = []
los = []
his = []
for c in ['2','3','4','EX']:
    # Find the comparison row (could be '1 - X' or 'X - 1')
    row = bayes_pair.query(f"subscale==@sub & factor==@factor & "
                            f"(comparison == '1 - {c}' or comparison == '{c} - 1')")
    if len(row) == 0:
        continue
    row = row.iloc[0]
    if row['comparison'] == f'1 - {c}':
        # negate
        m, lo, hi = -row['posterior_mean_diff'], -row['ci_hi'], -row['ci_lo']
    else:
        m, lo, hi = row['posterior_mean_diff'], row['ci_lo'], row['ci_hi']
    labels.append(f'{c}학년 − 1학년')
    means.append(m)
    los.append(lo)
    his.append(hi)

y = np.arange(len(labels))
ax.errorbar(means, y, xerr=[np.array(means)-np.array(los), np.array(his)-np.array(means)],
            fmt='o', color='#d62728', capsize=4, markersize=8)
ax.axvline(0, color='k', linewidth=0.8)
ax.set_yticks(y)
ax.set_yticklabels(labels[::-1] if False else labels)
ax.invert_yaxis()
ax.set_xlabel('수행태도 차이 (사후평균 ± 95% credible interval)')
ax.set_title('수행태도: 학년별 사후 차이 (Bayesian)\n빈도주의 ANOVA는 p=0.026 유의였으나 베이지안에서는 대부분 0을 포함')
ax.grid(alpha=0.3, axis='x')
fig.tight_layout()
fig.savefig(os.path.join(HERE, 'gel_pairwise_plot.pdf'), bbox_inches='tight')
fig.savefig(os.path.join(HERE, 'gel_pairwise_plot.png'), bbox_inches='tight', dpi=120)
print("저장: gel_pairwise_plot.pdf, .png")
