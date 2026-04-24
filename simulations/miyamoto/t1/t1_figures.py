#!/usr/bin/env python3
"""
t1_figures.py
=============
t1 논문용 그림 생성.

  그림 1 (t1_fig_mc_combined.png): β₁ 편향·RMSE·포함확률·검출력 — 2×2 패널
  그림 2 (t1_fig_mc_beta2.png)   : β₂ 포함확률 + 간접 효과 검출력 — 1×2 패널
  그림 3 (t1_fig_mcmc_example.png): MCMC 예시 사후 분포 (약한 vs 강한 사전)
    — 상위 폴더의 ss_mcmc_weakprior_N86.csv / ss_mcmc_strongprior_N86.csv 필요

실행: python t1_figures.py
"""
import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

warnings.filterwarnings('ignore')

HERE    = os.path.dirname(os.path.abspath(__file__))
PARENT  = os.path.dirname(HERE)
BLUE    = '#2166ac'
ORANGE  = '#d6604d'
GREY    = '#888888'
N86_X   = 86          # 원논문 표본 크기 수직선 위치

plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'axes.unicode_minus': False,
    'figure.dpi': 150,
})

# ─── 데이터 로드 ──────────────────────────────────────────────────────────
def load_summary():
    path = os.path.join(HERE, 't1_mc_summary.csv')
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} 가 없습니다. 먼저 t1_monte_carlo.py 를 실행하세요.")
    return pd.read_csv(path)

def load_mcmc(label):
    fname = f'ss_mcmc_{label}_N86.csv'
    path = os.path.join(PARENT, fname)
    if os.path.exists(path):
        return pd.read_csv(path)
    return None

# ─── 그림 1: 4-패널 β₁ 비교 ───────────────────────────────────────────────
def plot_fig1(sm):
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), sharex=True)
    fig.suptitle(
        'CS-OLS vs. LV-OLS (PCM-SEM 이론 상한): β₁ 추정 성능',
        fontsize=14, fontweight='bold', y=1.01)

    metrics = [
        ('cs_b1_bias',  'lv_b1_bias',  'Bias (β₁)',         (0, 0), True),
        ('cs_b1_rmse',  'lv_b1_rmse',  'RMSE (β₁)',          (0, 1), False),
        ('cs_b1_cov',   'lv_b1_cov',   '95% CI 포함확률 (β₁)', (1, 0), False),
        ('cs_ind_power','lv_ind_power', '간접 효과 검출력',      (1, 1), False),
    ]

    for cs_col, lv_col, ylabel, (r, c), add_zero in metrics:
        ax = axes[r][c]
        for sc, ls, lw in [('medium', '-', 2.2), ('small', '--', 1.5)]:
            d = sm[sm['scenario'] == sc].sort_values('N')
            ax.plot(d['N'], d[cs_col], color=ORANGE, ls=ls, lw=lw,
                    marker='o', ms=5)
            ax.plot(d['N'], d[lv_col], color=BLUE,   ls=ls, lw=lw,
                    marker='s', ms=5)
        ax.axvline(N86_X, color=GREY, lw=1.2, ls=':', alpha=0.8)
        ax.text(N86_X + 3, ax.get_ylim()[0] if add_zero else
                ax.get_ylim()[0], 'N=86', fontsize=8, color=GREY)
        if add_zero:
            ax.axhline(0, color='black', lw=0.8, ls='-', alpha=0.4)
        if not add_zero and 'cov' in cs_col:
            ax.axhline(0.95, color='black', lw=0.8, ls='-', alpha=0.4)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_xlabel('표본 크기 N', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_xticks([50, 86, 100, 150, 200, 300])

    legend_els = [
        mpatches.Patch(color=ORANGE, label='CS-OLS (합산 점수)'),
        mpatches.Patch(color=BLUE,   label='LV-OLS (잠재 변수 오라클 ≈ PCM-SEM)'),
        plt.Line2D([0],[0], color='grey', ls='-',  label='Medium 효과'),
        plt.Line2D([0],[0], color='grey', ls='--', label='Small 효과'),
    ]
    fig.legend(handles=legend_els, loc='lower center',
               ncol=4, fontsize=9, bbox_to_anchor=(0.5, -0.04))
    plt.tight_layout()
    out = os.path.join(HERE, 't1_fig_mc_combined.png')
    plt.savefig(out, dpi=200, bbox_inches='tight', facecolor='white')
    print(f'  저장: {out}')
    plt.close()

# ─── 그림 2: β₂ 포함확률 + 간접효과 검출력 ──────────────────────────────
def plot_fig2(sm):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    fig.suptitle('CS-OLS vs. LV-OLS: β₂ 포함확률 및 간접 효과 검출력',
                 fontsize=13, fontweight='bold')

    for ax, cs_col, lv_col, title in [
        (axes[0], 'cs_b2_cov', 'lv_b2_cov', '95% CI 포함확률 (β₂)'),
        (axes[1], 'cs_ind_power', 'lv_ind_power', '간접 효과 검출력 (β₁β₂)'),
    ]:
        for sc, ls in [('medium', '-'), ('small', '--')]:
            d = sm[sm['scenario'] == sc].sort_values('N')
            ax.plot(d['N'], d[cs_col], color=ORANGE, ls=ls, lw=2, marker='o', ms=5)
            ax.plot(d['N'], d[lv_col], color=BLUE,   ls=ls, lw=2, marker='s', ms=5)
        ax.axvline(N86_X, color=GREY, lw=1.2, ls=':', alpha=0.8)
        ax.axhline(0.95 if 'cov' in cs_col else 0.80,
                   color='black', lw=0.8, ls='-', alpha=0.35)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel('표본 크기 N', fontsize=10)
        ax.set_xticks([50, 86, 100, 150, 200, 300])
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1.05)

    legend_els = [
        mpatches.Patch(color=ORANGE, label='CS-OLS'),
        mpatches.Patch(color=BLUE,   label='LV-OLS (≈ PCM-SEM)'),
        plt.Line2D([0],[0], color='grey', ls='-',  label='Medium 효과'),
        plt.Line2D([0],[0], color='grey', ls='--', label='Small 효과'),
    ]
    fig.legend(handles=legend_els, loc='lower center',
               ncol=4, fontsize=9, bbox_to_anchor=(0.5, -0.05))
    plt.tight_layout()
    out = os.path.join(HERE, 't1_fig_mc_beta2.png')
    plt.savefig(out, dpi=200, bbox_inches='tight', facecolor='white')
    print(f'  저장: {out}')
    plt.close()

# ─── 그림 3: MCMC 예시 사후 분포 ─────────────────────────────────────────
def plot_fig3(weak_df, strong_df):
    params = [
        ('beta1', 'β₁ (X→M)'),
        ('beta2', 'β₂ (M→Y)'),
        ('gamma1', 'γ₁ (X→Y 직접)'),
        ('indirect', '간접 효과 β₁β₂'),
    ]
    # 간접 효과 계산
    for df in [weak_df, strong_df]:
        if 'indirect' not in df.columns:
            df['indirect'] = df['beta1'] * df['beta2']

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    fig.suptitle('MCMC 예시: N=86 시뮬레이션 데이터 — 경로 계수 사후 분포',
                 fontsize=13, fontweight='bold')

    for ax, (col, label) in zip(axes, params):
        for df, color, name in [
            (weak_df,   BLUE,   '약한 사전 분포'),
            (strong_df, ORANGE, '강한 사전 분포'),
        ]:
            if col in df.columns:
                vals = df[col].dropna()
                ax.hist(vals, bins=60, density=True, alpha=0.45, color=color)
                lo, hi = np.percentile(vals, [2.5, 97.5])
                ax.axvline(vals.mean(), color=color, lw=1.8, label=name)
                ax.axvspan(lo, hi, alpha=0.12, color=color)
        ax.axvline(0, color='black', lw=1.0, ls='--', alpha=0.6)
        ax.set_title(label, fontsize=10)
        ax.set_xlabel('추정값', fontsize=9)
        ax.grid(True, alpha=0.25)

    handles = [
        mpatches.Patch(color=BLUE,   label='약한 사전 분포'),
        mpatches.Patch(color=ORANGE, label='강한 사전 분포'),
    ]
    fig.legend(handles=handles, loc='lower center',
               ncol=2, fontsize=10, bbox_to_anchor=(0.5, -0.06))
    plt.tight_layout()
    out = os.path.join(HERE, 't1_fig_mcmc_example.png')
    plt.savefig(out, dpi=200, bbox_inches='tight', facecolor='white')
    print(f'  저장: {out}')
    plt.close()

# ─── 메인 ─────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print("t1 그림 생성 중...")
    sm = load_summary()

    print("  그림 1: β₁ 4-패널 비교")
    plot_fig1(sm)

    print("  그림 2: β₂ 포함확률 + 간접효과 검출력")
    plot_fig2(sm)

    weak   = load_mcmc('weakprior')
    strong = load_mcmc('strongprior')
    if weak is not None and strong is not None:
        print("  그림 3: MCMC 예시 사후 분포")
        plot_fig3(weak, strong)
    else:
        print("  그림 3: MCMC 샘플 파일 없음 — 건너뜀")
        print("    (ss_run_with_prior.py 실행 후 다시 시도)")

    print("완료.")

# ─── 그림 4: 3방향 MC 비교 (CS-OLS vs MAP+Laplace vs LV-OLS) ────────────
def plot_fig4(sm_base, sm_lap):
    """sm_base: t1_mc_summary.csv, sm_lap: t1_mc_laplace_summary.csv"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle('3방향 비교: CS-OLS vs MAP+Laplace vs LV-OLS — Medium 시나리오 β₁',
                 fontsize=13, fontweight='bold')
    GREEN = '#1a9641'
    metrics = [
        ('cs_b1_bias', 'lap_b1_bias', 'lv_b1_bias', 'Bias (β₁)', True),
        ('cs_b1_cov',  'lap_b1_cov',  'lv_b1_cov',  '95% CI 포함확률 (β₁)', False),
        ('cs_b2_cov',  'lap_b2_cov',  'lv_b2_cov',  '95% CI 포함확률 (β₂)', False),
    ]
    for ax, (cc, lc, vc, ylabel, zero) in zip(axes, metrics):
        db = sm_base[sm_base['scenario']=='medium'].sort_values('N')
        dl = sm_lap[sm_lap['scenario']=='medium'].sort_values('N') if sm_lap is not None else None
        ax.plot(db['N'], db[cc], color=ORANGE, lw=2, marker='o', ms=5, label='CS-OLS')
        ax.plot(db['N'], db[vc], color=BLUE,   lw=2, marker='s', ms=5,
                ls='--', label='LV-OLS (오라클)')
        if dl is not None and lc in dl.columns:
            ax.plot(dl['N'], dl[lc], color=GREEN, lw=2, marker='^', ms=6,
                    label='MAP+Laplace (Bayesian PCM-SEM)')
        ax.axvline(N86_X, color=GREY, lw=1.2, ls=':', alpha=0.7)
        if zero:  ax.axhline(0,    color='k', lw=0.8, alpha=0.4)
        else:     ax.axhline(0.95, color='k', lw=0.8, alpha=0.4)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_xlabel('표본 크기 N', fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_xticks([50, 86, 100, 150, 200, 300])
    plt.tight_layout()
    out = os.path.join(HERE, 't1_fig_3way_compare.png')
    plt.savefig(out, dpi=200, bbox_inches='tight', facecolor='white')
    print(f'  저장: {out}')
    plt.close()


# ─── 그림 5: MCMC 10회 반복 결과 요약 ───────────────────────────────────
def plot_fig5(reps_df):
    """reps_df: t1_mcmc_10reps.csv"""
    params   = ['beta1','beta2','gamma1','indirect']
    labels   = ['β₁ X→M','β₂ M→Y','γ₁ X→Y 직접','간접효과 β₁β₂']
    true_vals = {'beta1':0.50,'beta2':0.40,'gamma1':0.20,'indirect':0.20}

    fig, axes = plt.subplots(1, 4, figsize=(16, 5))
    fig.suptitle(
        f'MCMC {reps_df["rep"].nunique()}회 반복 결과 (N=86): 약한 vs 강한 사전 분포',
        fontsize=13, fontweight='bold')

    for ax, p, lbl in zip(axes, params, labels):
        for model, color, offset in [('weak', BLUE, -0.15), ('strong', ORANGE, 0.15)]:
            sub = reps_df[(reps_df['model']==model) & (reps_df['param']==p)]
            if sub.empty: continue
            y = np.arange(len(sub))
            ax.errorbar(sub['mean'], y + offset,
                        xerr=[sub['mean']-sub['q025'], sub['q975']-sub['mean']],
                        fmt='o', color=color, lw=1.2, capsize=3, ms=5,
                        label=f'{"약한" if model=="weak" else "강한"} 사전')
        tv = true_vals.get(p)
        if tv is not None:
            ax.axvline(tv, color='red', lw=1.5, ls='--', alpha=0.7, label=f'참값={tv}')
        ax.axvline(0, color='k', lw=0.8, ls='-', alpha=0.3)
        ax.set_title(lbl, fontsize=10)
        ax.set_xlabel('사후 평균 및 95% CrI', fontsize=9)
        ax.set_ylabel('반복 번호', fontsize=9)
        ax.grid(True, alpha=0.25, axis='x')
        if axes[0] == ax: ax.legend(fontsize=8)

    plt.tight_layout()
    out = os.path.join(HERE, 't1_fig_mcmc_10reps.png')
    plt.savefig(out, dpi=200, bbox_inches='tight', facecolor='white')
    print(f'  저장: {out}')
    plt.close()


# ─── 그림 6: 10회 반복 β₁·β₂ 분포 바이올린 ─────────────────────────────
def plot_fig6(reps_df):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle('MCMC 10회 반복: 사후 평균 분포 (약한 vs 강한 사전)',
                 fontsize=13, fontweight='bold')
    true_vals = {'beta1': 0.50, 'beta2': 0.40}

    for ax, (p, lbl) in zip(axes, [('beta1','β₁ (X→M)'), ('beta2','β₂ (M→Y)')]):
        data_w = reps_df[(reps_df['model']=='weak')   & (reps_df['param']==p)]['mean'].values
        data_s = reps_df[(reps_df['model']=='strong') & (reps_df['param']==p)]['mean'].values
        vp = ax.violinplot([data_w, data_s], positions=[1,2],
                           showmeans=True, showextrema=True)
        vp['bodies'][0].set_facecolor(BLUE);   vp['bodies'][0].set_alpha(0.55)
        vp['bodies'][1].set_facecolor(ORANGE); vp['bodies'][1].set_alpha(0.55)
        ax.scatter([1]*len(data_w), data_w, color=BLUE,   s=40, zorder=5, alpha=0.8)
        ax.scatter([2]*len(data_s), data_s, color=ORANGE, s=40, zorder=5, alpha=0.8)
        ax.axhline(true_vals[p], color='red', lw=1.5, ls='--', label=f'참값={true_vals[p]}')
        ax.set_xticks([1,2]); ax.set_xticklabels(['약한 사전','강한 사전'], fontsize=11)
        ax.set_ylabel('사후 평균', fontsize=10)
        ax.set_title(lbl, fontsize=11)
        ax.legend(fontsize=9); ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    out = os.path.join(HERE, 't1_fig_mcmc_violin.png')
    plt.savefig(out, dpi=200, bbox_inches='tight', facecolor='white')
    print(f'  저장: {out}')
    plt.close()


# ─── 메인 업데이트 ────────────────────────────────────────────────────────
if __name__ == '__main__':
    print("t1 그림 생성 중 (전체) ...")
    sm = load_summary()
    print("  그림 1: β₁ 4-패널 비교 (CS-OLS vs LV-OLS)")
    plot_fig1(sm)
    print("  그림 2: β₂ 포함확률 + 간접효과 검출력")
    plot_fig2(sm)

    weak   = load_mcmc('weakprior')
    strong = load_mcmc('strongprior')
    if weak is not None and strong is not None:
        print("  그림 3: MCMC 단일 예시 사후 분포")
        plot_fig3(weak, strong)
    else:
        print("  그림 3: MCMC 파일 없음 — 건너뜀")

    # MAP+Laplace 3방향 비교
    lap_path = os.path.join(HERE, 't1_mc_laplace_summary.csv')
    sm_lap = pd.read_csv(lap_path) if os.path.exists(lap_path) else None
    if sm_lap is not None:
        print("  그림 4: 3방향 비교 (CS-OLS vs MAP+Laplace vs LV-OLS)")
        plot_fig4(sm, sm_lap)
    else:
        print("  그림 4: t1_mc_laplace_summary.csv 없음 — 건너뜀")

    # MCMC 10회 반복
    reps_path = os.path.join(HERE, 't1_mcmc_10reps.csv')
    if os.path.exists(reps_path):
        reps_df = pd.read_csv(reps_path)
        print("  그림 5: MCMC 10회 반복 경로 계수")
        plot_fig5(reps_df)
        print("  그림 6: MCMC 10회 반복 바이올린 분포")
        plot_fig6(reps_df)
    else:
        print("  그림 5·6: t1_mcmc_10reps.csv 없음 — 건너뜀")

    print("완료.")
