"""
gel_compare.py
빈도주의(주월랑 방식) 결과와 베이지안(Gelman 방식) 결과를 비교 대조한다.
"""
import os
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))

# 빈도주의 결과 로드
freq = pd.read_excel(os.path.join(HERE, 'gel_freq_results.xlsx'),
                     sheet_name=None)
# 베이지안 결과 로드
bayes = pd.read_excel(os.path.join(HERE, 'gel_bayes_results.xlsx'),
                      sheet_name=None)

shrink = pd.read_csv(os.path.join(HERE, 'gel_shrinkage.csv'))

# ---------- 비교표 작성 ----------
SUBSCALES = ['쓰기인식', '쓰기반응', '수행태도', '쓰기태도']

# 1) ANOVA(학년) vs 베이지안(sigma_year)
rows = []
anova_year = freq['anova_year']
for sub in SUBSCALES:
    fr = anova_year[anova_year['subscale'] == sub].iloc[0]
    sig = bayes['sigma_summary'].query("subscale==@sub & factor=='alpha'").iloc[0]
    rows.append({
        'subscale': sub,
        'freq_F': round(fr['F'], 3),
        'freq_p': round(fr['p'], 3),
        'freq_판정': '유의' if fr['p'] < 0.05 else '비유의',
        'bayes_sigma_year_mean': round(sig['posterior_mean_sigma'], 3),
        'bayes_sigma_year_CI': f"[{sig['ci_lo']:.3f},{sig['ci_hi']:.3f}]",
    })
year_cmp = pd.DataFrame(rows)
print("\n[학년 효과 비교]\n", year_cmp.to_string(index=False))

# 2) t-test(성별) vs 베이지안(F-M 차이)
rows = []
tg = freq['ttest_gender']
for sub in SUBSCALES:
    fr = tg[tg['subscale'] == sub].iloc[0]
    bs = bayes['pairwise_diffs'].query("subscale==@sub & factor=='beta'").iloc[0]
    rows.append({
        'subscale': sub,
        'freq_t': round(fr['t'], 3),
        'freq_p': round(fr['p'], 3),
        'freq_판정': '유의' if fr['p'] < 0.05 else '비유의',
        'bayes_F-M_mean': round(-bs['posterior_mean_diff'], 3),  # M-F 부호 반대
        'bayes_CI': f"[{-bs['ci_hi']:.3f},{-bs['ci_lo']:.3f}]",
        'bayes_판정': '유의' if bs['significant_95CI'] else '비유의',
    })
gender_cmp = pd.DataFrame(rows)
print("\n[성별 효과 비교]\n", gender_cmp.to_string(index=False))

# 3) ANOVA(TOPIK)
rows = []
at = freq['anova_topik']
for sub in SUBSCALES:
    fr = at[at['subscale'] == sub].iloc[0]
    sig = bayes['sigma_summary'].query("subscale==@sub & factor=='gamma'").iloc[0]
    rows.append({
        'subscale': sub,
        'freq_F': round(fr['F'], 3),
        'freq_p': round(fr['p'], 3),
        'freq_판정': '유의' if fr['p'] < 0.05 else '비유의',
        'bayes_sigma_topik_mean': round(sig['posterior_mean_sigma'], 3),
        'bayes_sigma_CI': f"[{sig['ci_lo']:.3f},{sig['ci_hi']:.3f}]",
    })
topik_cmp = pd.DataFrame(rows)
print("\n[TOPIK 효과 비교]\n", topik_cmp.to_string(index=False))

# 4) t-test(어학원)
rows = []
ti = freq['ttest_inst']
for sub in SUBSCALES:
    fr = ti[ti['subscale'] == sub].iloc[0]
    bs = bayes['pairwise_diffs'].query("subscale==@sub & factor=='delta'").iloc[0]
    rows.append({
        'subscale': sub,
        'freq_t': round(fr['t'], 3),
        'freq_p': round(fr['p'], 3),
        'freq_판정': '유의' if fr['p'] < 0.05 else '비유의',
        'bayes_diff_mean': round(bs['posterior_mean_diff'], 3),
        'bayes_CI': f"[{bs['ci_lo']:.3f},{bs['ci_hi']:.3f}]",
        'bayes_판정': '유의' if bs['significant_95CI'] else '비유의',
    })
inst_cmp = pd.DataFrame(rows)
print("\n[어학원 효과 비교]\n", inst_cmp.to_string(index=False))

# 5) Shrinkage 표
print("\n[부분 풀링(shrinkage) — 학년별 평균]")
print(shrink.round(3).to_string(index=False))

# 저장
with pd.ExcelWriter(os.path.join(HERE, 'gel_compare.xlsx')) as w:
    year_cmp.to_excel(w, sheet_name='year', index=False)
    gender_cmp.to_excel(w, sheet_name='gender', index=False)
    topik_cmp.to_excel(w, sheet_name='topik', index=False)
    inst_cmp.to_excel(w, sheet_name='inst', index=False)
    shrink.round(3).to_excel(w, sheet_name='shrinkage', index=False)
print("\n저장: gel_compare.xlsx")

# Markdown 텍스트로도 저장 (LaTeX 본문에 사용)
with open(os.path.join(HERE, 'gel_compare.txt'), 'w', encoding='utf-8') as f:
    f.write("[학년 효과 비교]\n")
    f.write(year_cmp.to_string(index=False) + "\n\n")
    f.write("[성별 효과 비교]\n")
    f.write(gender_cmp.to_string(index=False) + "\n\n")
    f.write("[TOPIK 효과 비교]\n")
    f.write(topik_cmp.to_string(index=False) + "\n\n")
    f.write("[어학원 효과 비교]\n")
    f.write(inst_cmp.to_string(index=False) + "\n\n")
    f.write("[Shrinkage]\n")
    f.write(shrink.round(3).to_string(index=False) + "\n")
print("저장: gel_compare.txt")
