"""
gel_freq.py
시뮬레이션 데이터에 주월랑(2023) 논문의 통계처리방식을 적용한다.
- 일원배치 분산분석 (one-way ANOVA): 학년, TOPIK
- 독립표본 t-검정 (independent samples t-test): 성별, 어학원
- Pearson 상관 분석
- 유의수준 alpha=0.05, 다중비교 보정 없음
"""

import os
import numpy as np
import pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
df = pd.read_csv(os.path.join(HERE, 'gel_data.csv'))

DV_COLS = {'P_score': '쓰기인식', 'R_score': '쓰기반응',
           'A_score': '수행태도', 'T_score': '쓰기태도'}

# 논문 보고된 검정 결과 (paper-reported)
PAPER = {
    'anova_year': {'P_score': (1.387, .246), 'R_score': (2.092, .089),
                   'A_score': (2.925, .026), 'T_score': (2.316, .064)},
    'ttest_gender': {'P_score': (-2.779, .007), 'R_score': (-0.323, .747),
                     'A_score': (-0.569, .571), 'T_score': (-1.092, .278)},
    'anova_topik': {'P_score': (1.978, .106), 'R_score': (0.780, .541),
                    'A_score': (1.657, .168), 'T_score': (1.129, .349)},
    'ttest_inst':  {'P_score': (-0.211, .833), 'R_score': (-0.316, .752),
                    'A_score': (0.261, .795), 'T_score': (-0.152, .880)},
}


# ---------- 1) 일원배치 분산분석 ----------
def run_anova(group_col):
    rows = []
    for col, name in DV_COLS.items():
        groups = [g[col].values for _, g in df.groupby(group_col)]
        F, p = stats.f_oneway(*groups)
        rows.append({'subscale': name, 'F': F, 'p': p})
    return pd.DataFrame(rows)


# ---------- 2) 독립표본 t-검정 ----------
def run_ttest(group_col, level_a, level_b):
    rows = []
    for col, name in DV_COLS.items():
        a = df.loc[df[group_col] == level_a, col].values
        b = df.loc[df[group_col] == level_b, col].values
        t, p = stats.ttest_ind(a, b, equal_var=True)
        rows.append({'subscale': name, 't': t, 'p': p,
                     f'mean_{level_a}': a.mean(), f'mean_{level_b}': b.mean()})
    return pd.DataFrame(rows)


print("=" * 70)
print("[ANOVA] 학년별 (year) 쓰기 태도 차이")
print("=" * 70)
anova_year = run_anova('year')
for _, r in anova_year.iterrows():
    sub = next(k for k, v in DV_COLS.items() if v == r['subscale'])
    pF, pp = PAPER['anova_year'][sub]
    flag = "*" if r['p'] < 0.05 else " "
    print(f"  {r['subscale']:8s}  sim F={r['F']:.3f} p={r['p']:.3f}{flag}  | paper F={pF:.3f} p={pp:.3f}")

print()
print("=" * 70)
print("[t-test] 성별 (gender) 쓰기 태도 차이")
print("=" * 70)
tt_gender = run_ttest('gender', 'M', 'F')
for _, r in tt_gender.iterrows():
    sub = next(k for k, v in DV_COLS.items() if v == r['subscale'])
    pt, pp = PAPER['ttest_gender'][sub]
    flag = "*" if r['p'] < 0.05 else " "
    print(f"  {r['subscale']:8s}  sim t={r['t']:.3f} p={r['p']:.3f}{flag}  | paper t={pt:.3f} p={pp:.3f}")

print()
print("=" * 70)
print("[ANOVA] TOPIK 숙달도별 쓰기 태도 차이")
print("=" * 70)
anova_topik = run_anova('topik')
for _, r in anova_topik.iterrows():
    sub = next(k for k, v in DV_COLS.items() if v == r['subscale'])
    pF, pp = PAPER['anova_topik'][sub]
    flag = "*" if r['p'] < 0.05 else " "
    print(f"  {r['subscale']:8s}  sim F={r['F']:.3f} p={r['p']:.3f}{flag}  | paper F={pF:.3f} p={pp:.3f}")

print()
print("=" * 70)
print("[t-test] 어학원 (institute) 경험 유무별 쓰기 태도 차이")
print("=" * 70)
tt_inst = run_ttest('inst', 'yes', 'no')
for _, r in tt_inst.iterrows():
    sub = next(k for k, v in DV_COLS.items() if v == r['subscale'])
    pt, pp = PAPER['ttest_inst'][sub]
    flag = "*" if r['p'] < 0.05 else " "
    print(f"  {r['subscale']:8s}  sim t={r['t']:.3f} p={r['p']:.3f}{flag}  | paper t={pt:.3f} p={pp:.3f}")


# ---------- 3) Pearson 상관 ----------
print()
print("=" * 70)
print("[Pearson] 하위범주 간 상관")
print("=" * 70)

PAPER_INTER = {
    ('P_score','R_score'): 0.583, ('P_score','A_score'): 0.281,
    ('P_score','T_score'): 0.708, ('R_score','A_score'): 0.508,
    ('R_score','T_score'): 0.930, ('A_score','T_score'): 0.731,
}

inter_rows = []
for (a, b), target in PAPER_INTER.items():
    r, p = stats.pearsonr(df[a], df[b])
    flag = "**" if p < 0.01 else ("*" if p < 0.05 else "  ")
    print(f"  {DV_COLS[a]:8s} - {DV_COLS[b]:8s}  sim r={r:.3f} p={p:.4f}{flag}  | paper r={target:.3f}**")
    inter_rows.append({'pair': f'{DV_COLS[a]}-{DV_COLS[b]}', 'r_sim': r, 'p_sim': p, 'r_paper': target})

# 학습자요인 vs 쓰기태도 상관
print()
print("=" * 70)
print("[Pearson] 학습자요인 vs 쓰기태도 하위범주")
print("=" * 70)

PAPER_LEARNER = {
    ('gender_num','P_score'): 0.290, ('year_num','A_score'): 0.328,
}

# 부호화
df['gender_num'] = df['gender'].map({'M': 1, 'F': 2})
df['year_num'] = df['year'].map({'1':1,'2':2,'3':3,'4':4,'EX':5})
df['topik_num'] = df['topik'].map({'none':0,'L3':3,'L4':4,'L5':5,'L6':6})
df['inst_num'] = df['inst'].map({'no':0,'yes':1})

learner_rows = []
for factor in ['gender_num','year_num','topik_num','inst_num']:
    for col in ['P_score','R_score','A_score','T_score']:
        r, p = stats.pearsonr(df[factor], df[col])
        learner_rows.append({'factor':factor, 'subscale':col, 'r':r, 'p':p})

learner = pd.DataFrame(learner_rows)
# 논문 표 15에 대응되는 r 값들 표시
print("  factor       subscale  sim_r   sim_p   significance")
for _, r in learner.iterrows():
    flag = "**" if r['p'] < 0.01 else ("*" if r['p'] < 0.05 else "  ")
    print(f"  {r['factor']:10s}  {r['subscale']:8s}  r={r['r']:+.3f}  p={r['p']:.3f}{flag}")


# ---------- 4) 결과 저장 ----------
all_results = {
    'anova_year': anova_year,
    'ttest_gender': tt_gender,
    'anova_topik': anova_topik,
    'ttest_inst': tt_inst,
    'inter_correlation': pd.DataFrame(inter_rows),
    'learner_correlation': learner,
}

with pd.ExcelWriter(os.path.join(HERE, 'gel_freq_results.xlsx')) as w:
    for sheet, d in all_results.items():
        d.to_excel(w, sheet_name=sheet, index=False)
print(f"\n저장: gel_freq_results.xlsx")

# 비교용 텍스트
with open(os.path.join(HERE, 'gel_freq_summary.txt'), 'w', encoding='utf-8') as f:
    f.write("주월랑(2023) 빈도주의 분석 재현 결과\n")
    f.write("=" * 70 + "\n")
    f.write("\n[ANOVA 학년 × 쓰기 태도]\n")
    for _, r in anova_year.iterrows():
        sub = next(k for k, v in DV_COLS.items() if v == r['subscale'])
        pF, pp = PAPER['anova_year'][sub]
        sig_sim = "유의" if r['p'] < 0.05 else "비유의"
        sig_paper = "유의" if pp < 0.05 else "비유의"
        f.write(f"  {r['subscale']:8s}  sim F={r['F']:.3f} p={r['p']:.3f} ({sig_sim})  "
                f"| paper F={pF:.3f} p={pp:.3f} ({sig_paper})\n")
    f.write("\n[t-test 성별 × 쓰기 태도]\n")
    for _, r in tt_gender.iterrows():
        sub = next(k for k, v in DV_COLS.items() if v == r['subscale'])
        pt, pp = PAPER['ttest_gender'][sub]
        sig_sim = "유의" if r['p'] < 0.05 else "비유의"
        sig_paper = "유의" if pp < 0.05 else "비유의"
        f.write(f"  {r['subscale']:8s}  sim t={r['t']:.3f} p={r['p']:.3f} ({sig_sim})  "
                f"| paper t={pt:.3f} p={pp:.3f} ({sig_paper})\n")
    f.write("\n[ANOVA TOPIK × 쓰기 태도]\n")
    for _, r in anova_topik.iterrows():
        sub = next(k for k, v in DV_COLS.items() if v == r['subscale'])
        pF, pp = PAPER['anova_topik'][sub]
        sig_sim = "유의" if r['p'] < 0.05 else "비유의"
        sig_paper = "유의" if pp < 0.05 else "비유의"
        f.write(f"  {r['subscale']:8s}  sim F={r['F']:.3f} p={r['p']:.3f} ({sig_sim})  "
                f"| paper F={pF:.3f} p={pp:.3f} ({sig_paper})\n")
    f.write("\n[t-test 어학원 × 쓰기 태도]\n")
    for _, r in tt_inst.iterrows():
        sub = next(k for k, v in DV_COLS.items() if v == r['subscale'])
        pt, pp = PAPER['ttest_inst'][sub]
        sig_sim = "유의" if r['p'] < 0.05 else "비유의"
        sig_paper = "유의" if pp < 0.05 else "비유의"
        f.write(f"  {r['subscale']:8s}  sim t={r['t']:.3f} p={r['p']:.3f} ({sig_sim})  "
                f"| paper t={pt:.3f} p={pp:.3f} ({sig_paper})\n")

print(f"저장: gel_freq_summary.txt")
