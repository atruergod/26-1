#!/usr/bin/env python3
"""
ss_figures.py
=============
시뮬레이션 결과 시각화 — 논문 게재용 그래프 생성.
저장: ss_fig_bias.png, ss_fig_rmse.png, ss_fig_coverage.png,
     ss_fig_power.png, ss_fig_combined.png, ss_fig_attenuation.png
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

# ── Korean font for Windows ──────────────────────────────
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False
sns.set_theme(style='whitegrid', font_scale=1.15)
plt.rcParams.update({'figure.dpi': 130, 'font.family': 'Malgun Gothic'})

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
df      = pd.read_csv(os.path.join(OUT_DIR, 'ss_results.csv'))
summary = pd.read_csv(os.path.join(OUT_DIR, 'ss_summary.csv'))

COLORS = {'comp': '#DD8452', 'oracle': '#4C72B0'}
LABELS = {'comp': 'CS-OLS (합산 점수)', 'oracle': 'LV-OLS (잠재 변수)'}
MARKERS= {'comp': 'o', 'oracle': 's'}
SCENLAB= {'small': '소효과 (β₁=β₂=0.30, γ₁=0.00)', 'medium': '중효과 (β₁=0.50, β₂=0.40, γ₁=0.20)'}
PARAMS = ['b1', 'b2', 'g1', 'ind']
PAR_LABELS = {'b1': 'β₁ (X→M)', 'b2': 'β₂ (M→Y)', 'g1': 'γ₁ (X→Y 직접)', 'ind': '간접 효과 (β₁β₂)'}

def savefig(fig, name):
    path = os.path.join(OUT_DIR, f'ss_fig_{name}.png')
    fig.savefig(path, bbox_inches='tight', dpi=130)
    plt.close(fig)
    print(f"  Saved: {path}")

# ─────────────────────────────────────────────
# Fig 1 – 편향 (Bias) 비교
# ─────────────────────────────────────────────
def fig_bias():
    fig, axes = plt.subplots(2, 4, figsize=(16, 8), sharey=False)
    for row_idx, sc in enumerate(['small', 'medium']):
        sub = summary[summary['scenario'] == sc]
        for col_idx, par in enumerate(PARAMS):
            ax = axes[row_idx, col_idx]
            d  = sub[sub['param'] == par]
            for mth in ['comp', 'oracle']:
                dm = d[d['method'] == mth]
                ax.plot(dm['N'], dm['bias'], marker=MARKERS[mth],
                        color=COLORS[mth], lw=2, ms=7, label=LABELS[mth])
            ax.axhline(0, ls='--', color='black', lw=1, alpha=0.5)
            ax.set_title(PAR_LABELS[par], fontsize=10, fontweight='bold')
            ax.set_xlabel('표본 크기 N')
            if col_idx == 0: ax.set_ylabel(f'편향 (Bias)\n{SCENLAB[sc][:8]}…', fontsize=9)
            ax.set_xticks([50,86,150,200,300])
    axes[0, 0].legend(fontsize=9, loc='upper right')
    fig.suptitle('그림 1. 방법론별 경로 계수 편향 비교\n'
                 '(위: 소효과 시나리오, 아래: 중효과 시나리오)',
                 fontweight='bold', fontsize=13, y=1.01)
    plt.tight_layout()
    savefig(fig, 'bias')

# ─────────────────────────────────────────────
# Fig 2 – RMSE 비교
# ─────────────────────────────────────────────
def fig_rmse():
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    for row_idx, sc in enumerate(['small', 'medium']):
        sub = summary[summary['scenario'] == sc]
        for col_idx, par in enumerate(PARAMS):
            ax = axes[row_idx, col_idx]
            d  = sub[sub['param'] == par]
            for mth in ['comp', 'oracle']:
                dm = d[d['method'] == mth]
                ax.plot(dm['N'], dm['rmse'], marker=MARKERS[mth],
                        color=COLORS[mth], lw=2, ms=7, label=LABELS[mth])
            ax.set_title(PAR_LABELS[par], fontsize=10, fontweight='bold')
            ax.set_xlabel('표본 크기 N')
            if col_idx == 0: ax.set_ylabel('RMSE', fontsize=9)
            ax.set_xticks([50,86,150,200,300])
    axes[0, 0].legend(fontsize=9)
    fig.suptitle('그림 2. 방법론별 RMSE 비교',
                 fontweight='bold', fontsize=13, y=1.01)
    plt.tight_layout()
    savefig(fig, 'rmse')

# ─────────────────────────────────────────────
# Fig 3 – 구간 포괄 확률 (Coverage)
# ─────────────────────────────────────────────
def fig_coverage():
    fig, axes = plt.subplots(2, 4, figsize=(16, 8), sharey=True)
    for row_idx, sc in enumerate(['small', 'medium']):
        sub = summary[summary['scenario'] == sc]
        for col_idx, par in enumerate(PARAMS):
            ax = axes[row_idx, col_idx]
            d  = sub[sub['param'] == par]
            for mth in ['comp', 'oracle']:
                dm = d[d['method'] == mth]
                ax.plot(dm['N'], dm['coverage'], marker=MARKERS[mth],
                        color=COLORS[mth], lw=2, ms=7, label=LABELS[mth])
            ax.axhline(0.95, ls='--', color='green', lw=1.5, alpha=0.8, label='명목 수준 95%')
            ax.set_ylim(0.3, 1.05)
            ax.set_title(PAR_LABELS[par], fontsize=10, fontweight='bold')
            ax.set_xlabel('표본 크기 N')
            if col_idx == 0: ax.set_ylabel('Coverage Probability', fontsize=9)
            ax.set_xticks([50,86,150,200,300])
    axes[0, 0].legend(fontsize=8)
    fig.suptitle('그림 3. 95% 신뢰구간 포괄 확률 비교\n(점선: 명목 수준 0.95)',
                 fontweight='bold', fontsize=13, y=1.01)
    plt.tight_layout()
    savefig(fig, 'coverage')

# ─────────────────────────────────────────────
# Fig 4 – 검정력 (Power) 비교
# ─────────────────────────────────────────────
def fig_power():
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, sc in zip(axes, ['small', 'medium']):
        sub = summary[(summary['scenario']==sc) & (summary['param']=='ind')]
        for mth in ['comp', 'oracle']:
            dm = sub[sub['method']==mth]
            ax.plot(dm['N'], dm['power'], marker=MARKERS[mth],
                    color=COLORS[mth], lw=2.5, ms=9, label=LABELS[mth])
        ax.axhline(0.80, ls=':', color='gray', lw=1.5, label='관례적 검정력 0.80')
        ax.set_title(SCENLAB[sc], fontweight='bold', fontsize=11)
        ax.set_xlabel('표본 크기 N'); ax.set_ylabel('검정력 (Power)')
        ax.set_ylim(0, 1.05); ax.legend(fontsize=9)
        ax.set_xticks([50,86,100,150,200,300])
        # Annotate N=86 (original paper)
        for mth in ['comp','oracle']:
            dm = sub[sub['method']==mth]
            v86 = dm[dm['N']==86]['power'].values
            if len(v86):
                ax.annotate(f'{v86[0]:.2f}', xy=(86, v86[0]),
                            xytext=(86+5, v86[0]+0.03), fontsize=9,
                            color=COLORS[mth], fontweight='bold')
    fig.suptitle('그림 4. 간접 효과(β₁β₂) 검정력 비교\n(N=86: 원논문 표본 크기)',
                 fontweight='bold', fontsize=13, y=1.02)
    plt.tight_layout()
    savefig(fig, 'power')

# ─────────────────────────────────────────────
# Fig 5 – 종합 비교 (논문 메인 그림)
# ─────────────────────────────────────────────
def fig_combined():
    fig = plt.figure(figsize=(16, 12))
    gs  = gridspec.GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.35)

    metrics = [
        ('bias',     'β₁ 편향 (Bias)',            'b1',  0,0),
        ('coverage', 'β₁ 구간 포괄 확률',          'b1',  0,1),
        ('rmse',     'β₁ RMSE',                    'b1',  1,0),
        ('power',    '간접 효과 검정력',            'ind', 1,1),
        ('bias',     'β₂ 편향 (Bias)',             'b2',  2,0),
        ('coverage', 'β₂ 구간 포괄 확률',          'b2',  2,1),
    ]

    for metric, title, par, r, c in metrics:
        ax = fig.add_subplot(gs[r, c])
        for sc, ls in [('small','--'),('medium','-')]:
            sub = summary[(summary['scenario']==sc) & (summary['param']==par)]
            for mth in ['comp','oracle']:
                dm = sub[sub['method']==mth]
                lbl = f"{LABELS[mth]} ({sc[:3]})" if r+c==0 else None
                ax.plot(dm['N'], dm[metric], marker=MARKERS[mth],
                        color=COLORS[mth], lw=2, ms=6, ls=ls, label=lbl, alpha=0.9)
        if metric == 'coverage':
            ax.axhline(0.95, ls=':', color='green', lw=1.5, alpha=0.7)
            ax.set_ylim(0.3, 1.05)
        if metric == 'bias':
            ax.axhline(0,   ls=':', color='black', lw=1,   alpha=0.5)
        if metric == 'power':
            ax.axhline(0.80,ls=':', color='gray',  lw=1.5, alpha=0.7)
            ax.set_ylim(0, 1.05)
        ax.set_title(title, fontweight='bold', fontsize=11)
        ax.set_xlabel('표본 크기 N', fontsize=9)
        ax.set_xticks([50,86,100,150,200,300])
        ax.axvline(86, color='red', alpha=0.25, lw=3)  # 원논문 N=86 표시

    # 범례 (첫 번째 subplot)
    handles = []
    for mth in ['comp','oracle']:
        handles.append(plt.Line2D([],[],color=COLORS[mth],lw=2,marker=MARKERS[mth],ms=6,label=LABELS[mth]))
    for sc, ls in [('small','--'),('medium','-')]:
        handles.append(plt.Line2D([],[],color='gray',lw=2,ls=ls,label=SCENLAB[sc]))
    handles.append(plt.Line2D([],[],color='red',lw=3,alpha=0.4,label='N=86 (원논문)'))
    fig.legend(handles=handles, loc='upper center', ncol=3, fontsize=9,
               bbox_to_anchor=(0.5, 1.01), frameon=True)

    fig.suptitle('그림 5. CS-OLS vs. LV-OLS (PCM-SEM) 방법론 비교 요약\n'
                 '(붉은 수직선: 원논문 표본 크기 N=86)',
                 fontweight='bold', fontsize=13, y=1.06)
    savefig(fig, 'combined')

# ─────────────────────────────────────────────
# Fig 6 – 감쇠 편향 시각화 (scatter)
# ─────────────────────────────────────────────
def fig_attenuation():
    """N=86, medium scenario: scatter of comp vs oracle β₁ estimates."""
    sub = df[(df['scenario']=='medium') & (df['N']==86)]
    true_b1 = sub['beta1'].iloc[0]

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    for ax, (par, tv_col, lbl) in zip(axes, [
        ('b1','beta1','β₁ (X→M)'),
        ('b2','beta2','β₂ (M→Y)'),
        ('ind',None, '간접 효과 β₁β₂'),
    ]):
        tv = sub[tv_col].iloc[0] if tv_col else sub['beta1'].iloc[0]*sub['beta2'].iloc[0]
        c_est = sub[f'{par}_comp']; o_est = sub[f'{par}_oracle']
        ax.scatter(o_est, c_est, alpha=0.35, s=20, color='#4C72B0', label='각 반복')
        lo = min(o_est.min(), c_est.min(), tv-0.1)
        hi = max(o_est.max(), c_est.max(), tv+0.1)
        ax.plot([lo,hi],[lo,hi], 'k--', lw=1, alpha=0.5, label='y=x 기준선')
        ax.axvline(tv, color='green', lw=1.5, alpha=0.7, label=f'참값={tv:.2f}')
        ax.axhline(tv, color='green', lw=1.5, alpha=0.7)
        ax.set_xlabel(f'LV-OLS 추정치', fontsize=10)
        ax.set_ylabel(f'CS-OLS 추정치', fontsize=10)
        ax.set_title(lbl, fontweight='bold')
        # Annotation
        ax.text(0.05, 0.93,
                f'LV 평균={o_est.mean():.3f}\nCS 평균={c_est.mean():.3f}\n참값={tv:.3f}',
                transform=ax.transAxes, fontsize=9, va='top',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
        ax.legend(fontsize=8)
    fig.suptitle('그림 6. 감쇠 편향 시각화 (N=86, 중효과 시나리오, 500회 반복)\n'
                 'CS-OLS는 LV-OLS보다 일관되게 과소 추정함',
                 fontweight='bold', fontsize=13, y=1.02)
    plt.tight_layout()
    savefig(fig, 'attenuation')

# ─────────────────────────────────────────────
# Fig 7 – N=86 비교 막대그래프 (논문 핵심 그림)
# ─────────────────────────────────────────────
def fig_n86_summary():
    """Key figure: N=86 comparison across all metrics and parameters."""
    metrics = ['bias','rmse','coverage','power']
    met_lab = {'bias':'편향 (Bias)', 'rmse':'RMSE',
               'coverage':'구간 포괄 확률', 'power':'검정력 (Power)'}
    pars    = ['b1','b2','g1','ind']
    par_lab = {'b1':'β₁','b2':'β₂','g1':'γ₁','ind':'간접효과'}
    scenarios = ['small','medium']

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    x = np.arange(len(pars)); w = 0.35

    for row_i, sc in enumerate(scenarios):
        sub = summary[(summary['scenario']==sc) & (summary['N']==86)]
        for col_i, met in enumerate(metrics):
            ax = axes[row_i, col_i]
            vals_c = [sub[(sub['param']==p)&(sub['method']=='comp' )][met].values[0] for p in pars]
            vals_o = [sub[(sub['param']==p)&(sub['method']=='oracle')][met].values[0] for p in pars]
            bars_c = ax.bar(x-w/2, vals_c, w, color=COLORS['comp'],   alpha=0.85, label=LABELS['comp'])
            bars_o = ax.bar(x+w/2, vals_o, w, color=COLORS['oracle'], alpha=0.85, label=LABELS['oracle'])
            if met == 'coverage': ax.axhline(0.95, ls='--', color='green', lw=1.5)
            if met == 'bias':     ax.axhline(0,    ls='--', color='black', lw=1)
            if met == 'power':    ax.axhline(0.80, ls=':',  color='gray',  lw=1.5)
            ax.set_title(met_lab[met], fontweight='bold', fontsize=10)
            ax.set_xticks(x); ax.set_xticklabels([par_lab[p] for p in pars])
            if col_i == 0:
                ax.set_ylabel(SCENLAB[sc][:5]+'…', fontsize=8)
            for rect, v in zip(bars_c, vals_c):
                ax.text(rect.get_x()+rect.get_width()/2, rect.get_height()+0.005,
                        f'{v:.2f}', ha='center', fontsize=7.5, color=COLORS['comp'])
            for rect, v in zip(bars_o, vals_o):
                ax.text(rect.get_x()+rect.get_width()/2, rect.get_height()+0.005,
                        f'{v:.2f}', ha='center', fontsize=7.5, color=COLORS['oracle'])
    axes[0,0].legend(fontsize=8)
    fig.suptitle('그림 7. N=86 조건에서의 방법론 비교 요약\n'
                 '(원논문과 동일한 표본 크기)',
                 fontweight='bold', fontsize=13, y=1.02)
    plt.tight_layout()
    savefig(fig, 'n86_summary')

# ─────────────────────────────────────────────
# 실행
# ─────────────────────────────────────────────
if __name__ == '__main__':
    print("Generating figures...")
    fig_bias()
    fig_rmse()
    fig_coverage()
    fig_power()
    fig_combined()
    fig_attenuation()
    fig_n86_summary()
    print("All figures generated.")
