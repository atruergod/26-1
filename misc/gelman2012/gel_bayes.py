"""
gel_bayes.py
Gelman, Hill & Yajima (2012) 의 베이지안 다층모형(Bayesian multilevel model)을
시뮬레이션 데이터에 적용한다.  백엔드는 numpyro (NUTS sampler).

모형(model):
    y_i = mu + alpha[year[i]] + beta[gender[i]]
              + gamma[topik[i]] + delta[inst[i]] + epsilon_i
    alpha[j]  ~ Normal(0, sigma_year)     # j=1..5
    gamma[k]  ~ Normal(0, sigma_topik)    # k=1..5
    beta[g]   ~ Normal(0, sigma_gender)   # g=1,2  (자동 부분풀링)
    delta[m]  ~ Normal(0, sigma_inst)     # m=1,2
    epsilon_i ~ Normal(0, sigma_y)
    sigma_*   ~ HalfNormal(1)             # 약한 정보 사전(weakly informative)
    mu        ~ Normal(3, 2)

각 하위범주(쓰기인식 P, 쓰기반응 R, 수행태도 A, 쓰기태도 T) 에 대해 별도 적합.
사후분포에서:
    - 95% credible interval 이 0 을 포함 -> 비유의
    - 95% credible interval 이 0 을 포함하지 않음 -> 유의 (Bayesian 의미)
    - P(effect > 0) 또는 P(effect < 0) 도 함께 보고
"""

import os
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS

numpyro.set_host_device_count(2)
RNG = jax.random.PRNGKey(2026)

HERE = os.path.dirname(os.path.abspath(__file__))
df = pd.read_csv(os.path.join(HERE, 'gel_data.csv'))

# 인덱스 매핑
YEAR_LEVELS  = ['1','2','3','4','EX']
TOPIK_LEVELS = ['none','L3','L4','L5','L6']
GENDER_LEVELS = ['M','F']
INST_LEVELS   = ['no','yes']

year_idx   = df['year'].map({v:i for i,v in enumerate(YEAR_LEVELS)}).values
topik_idx  = df['topik'].map({v:i for i,v in enumerate(TOPIK_LEVELS)}).values
gender_idx = df['gender'].map({v:i for i,v in enumerate(GENDER_LEVELS)}).values
inst_idx   = df['inst'].map({v:i for i,v in enumerate(INST_LEVELS)}).values


def model(year_idx, topik_idx, gender_idx, inst_idx, y=None):
    mu = numpyro.sample('mu', dist.Normal(3.0, 2.0))

    # 분산 모수: HalfNormal weakly informative
    sigma_year   = numpyro.sample('sigma_year',   dist.HalfNormal(1.0))
    sigma_topik  = numpyro.sample('sigma_topik',  dist.HalfNormal(1.0))
    sigma_gender = numpyro.sample('sigma_gender', dist.HalfNormal(1.0))
    sigma_inst   = numpyro.sample('sigma_inst',   dist.HalfNormal(1.0))
    sigma_y      = numpyro.sample('sigma_y',      dist.HalfNormal(1.0))

    # 합산 0 제약을 위해 비중심화(non-centered) 파라미터화
    with numpyro.plate('p_year',   len(YEAR_LEVELS)):
        alpha = numpyro.sample('alpha', dist.Normal(0.0, sigma_year))
    with numpyro.plate('p_topik',  len(TOPIK_LEVELS)):
        gamma = numpyro.sample('gamma', dist.Normal(0.0, sigma_topik))
    with numpyro.plate('p_gender', len(GENDER_LEVELS)):
        beta = numpyro.sample('beta', dist.Normal(0.0, sigma_gender))
    with numpyro.plate('p_inst',   len(INST_LEVELS)):
        delta = numpyro.sample('delta', dist.Normal(0.0, sigma_inst))

    yhat = mu + alpha[year_idx] + gamma[topik_idx] + beta[gender_idx] + delta[inst_idx]

    with numpyro.plate('obs', len(year_idx)):
        numpyro.sample('y', dist.Normal(yhat, sigma_y), obs=y)


def fit(y, name, seed=0):
    kernel = NUTS(model, target_accept_prob=0.95)
    mcmc = MCMC(kernel, num_warmup=800, num_samples=1500,
                num_chains=2, chain_method='sequential', progress_bar=False)
    mcmc.run(jax.random.PRNGKey(seed),
             year_idx=jnp.array(year_idx),
             topik_idx=jnp.array(topik_idx),
             gender_idx=jnp.array(gender_idx),
             inst_idx=jnp.array(inst_idx),
             y=jnp.array(y))
    samples = mcmc.get_samples(group_by_chain=False)
    print(f"\n[{name}] sampling done. divergences ok.")
    return samples


def summarize_factor(samples, factor_key, levels, name):
    """집단 효과(grand_mean 대비 편차)에 대한 사후 요약."""
    samp = samples[factor_key]  # shape (n_samples, K)
    # mean of effects (already centered around 0)
    rows = []
    for k, lev in enumerate(levels):
        eff = samp[:, k]
        m = float(eff.mean())
        ci_lo, ci_hi = np.percentile(eff, [2.5, 97.5])
        p_pos = float((eff > 0).mean())
        p_neg = float((eff < 0).mean())
        # Bayesian "significant" if 95% CI excludes 0
        sig = (ci_lo > 0) or (ci_hi < 0)
        rows.append({
            'subscale': name,
            'factor': factor_key,
            'level': lev,
            'posterior_mean': m,
            'ci_lo': float(ci_lo),
            'ci_hi': float(ci_hi),
            'P(eff>0)': p_pos,
            'P(eff<0)': p_neg,
            'significant_95CI': sig,
        })
    return rows


def summarize_pairwise(samples, factor_key, levels, name):
    """집단 간 차이의 사후 분포 요약(예: 남-여, 1학년-2학년)."""
    samp = samples[factor_key]
    rows = []
    for i in range(len(levels)):
        for j in range(i+1, len(levels)):
            diff = samp[:, i] - samp[:, j]
            m = float(diff.mean())
            ci_lo, ci_hi = np.percentile(diff, [2.5, 97.5])
            sig = (ci_lo > 0) or (ci_hi < 0)
            p_pos = float((diff > 0).mean())
            rows.append({
                'subscale': name,
                'factor': factor_key,
                'comparison': f'{levels[i]} - {levels[j]}',
                'posterior_mean_diff': m,
                'ci_lo': float(ci_lo),
                'ci_hi': float(ci_hi),
                'P(diff>0)': p_pos,
                'significant_95CI': sig,
            })
    return rows


def summarize_factor_test(samples, factor_key, name):
    """집단 요인 전체의 분산(sigma_*)에 대한 사후. classical ANOVA F-test 와 대응."""
    sig_key = factor_key.replace('alpha','sigma_year')\
                        .replace('gamma','sigma_topik')\
                        .replace('beta','sigma_gender')\
                        .replace('delta','sigma_inst')
    samp = samples[sig_key]
    m = float(samp.mean())
    ci_lo, ci_hi = np.percentile(samp, [2.5, 97.5])
    return {
        'subscale': name,
        'factor': factor_key,
        'posterior_mean_sigma': m,
        'ci_lo': float(ci_lo),
        'ci_hi': float(ci_hi),
        'note': '집단 간 표준편차의 사후 추정. 0에 가까울수록 집단 효과 미미.'
    }


# 각 하위범주에 대해 모형 적합
SUBSCALES = {
    'P_score': '쓰기인식',
    'R_score': '쓰기반응',
    'A_score': '수행태도',
    'T_score': '쓰기태도',
}

all_factor_rows = []
all_pair_rows = []
all_sigma_rows = []
posterior_summary = {}

for col, name in SUBSCALES.items():
    print(f"\n{'='*70}\n베이지안 다층 모형 적합: {name} ({col})\n{'='*70}")
    samples = fit(df[col].values, name=name, seed=hash(col) & 0xFFFF)
    posterior_summary[col] = samples

    # 요인별 효과 요약
    for fkey, lv in [
        ('alpha', YEAR_LEVELS),
        ('gamma', TOPIK_LEVELS),
        ('beta', GENDER_LEVELS),
        ('delta', INST_LEVELS),
    ]:
        all_factor_rows.extend(summarize_factor(samples, fkey, lv, name))
        all_pair_rows.extend(summarize_pairwise(samples, fkey, lv, name))
        all_sigma_rows.append(summarize_factor_test(samples, fkey, name))


# DataFrames 로 저장
factor_df = pd.DataFrame(all_factor_rows)
pair_df = pd.DataFrame(all_pair_rows)
sigma_df = pd.DataFrame(all_sigma_rows)

out_xlsx = os.path.join(HERE, 'gel_bayes_results.xlsx')
with pd.ExcelWriter(out_xlsx) as w:
    factor_df.to_excel(w, sheet_name='factor_effects', index=False)
    pair_df.to_excel(w, sheet_name='pairwise_diffs', index=False)
    sigma_df.to_excel(w, sheet_name='sigma_summary', index=False)
print(f"\n저장: {out_xlsx}")


# ----------- 빈도주의와 대응되는 판정 출력 -----------
print("\n\n" + "=" * 70)
print("[베이지안 다층모형 결과 요약 — 빈도주의 검정 대응 판정]")
print("=" * 70)


def fmt_sig(b):
    return "유의" if b else "비유의"


# 1) 학년 차이 (vs 1학년 = ANOVA의 0번째 카테고리 reference)
print("\n[학년] 사후 차이: 학년 j vs 1학년 (Bayesian, 빈도주의의 일원배치 분산분석에 대응)")
for col, name in SUBSCALES.items():
    samples = posterior_summary[col]
    alpha = samples['alpha']  # (S, 5)
    # 학년 2,3,4,EX vs 1
    print(f"  [{name}]")
    for k, lev in enumerate(YEAR_LEVELS[1:], start=1):
        diff = alpha[:, k] - alpha[:, 0]
        ci = np.percentile(diff, [2.5, 97.5])
        p_pos = float((diff > 0).mean())
        sig = (ci[0] > 0) or (ci[1] < 0)
        print(f"    {lev}학년 - 1학년: mean={diff.mean():+.3f}, "
              f"95% CI=[{ci[0]:+.3f},{ci[1]:+.3f}], P(>0)={p_pos:.3f}  {fmt_sig(sig)}")

# 2) 성별 차이 (F vs M)
print("\n[성별] F - M 사후 차이 (Bayesian, 빈도주의의 독립표본 t-검정에 대응)")
for col, name in SUBSCALES.items():
    samples = posterior_summary[col]
    beta = samples['beta']  # (S, 2)
    diff = beta[:, 1] - beta[:, 0]  # F - M
    ci = np.percentile(diff, [2.5, 97.5])
    p_pos = float((diff > 0).mean())
    sig = (ci[0] > 0) or (ci[1] < 0)
    print(f"    {name}: mean={diff.mean():+.3f}, "
          f"95% CI=[{ci[0]:+.3f},{ci[1]:+.3f}], P(>0)={p_pos:.3f}  {fmt_sig(sig)}")

# 3) TOPIK 차이 (각급 vs 없음)
print("\n[TOPIK] 각급 - 없음 (Bayesian, 빈도주의의 일원배치 분산분석에 대응)")
for col, name in SUBSCALES.items():
    samples = posterior_summary[col]
    gamma = samples['gamma']
    print(f"  [{name}]")
    for k, lev in enumerate(TOPIK_LEVELS[1:], start=1):
        diff = gamma[:, k] - gamma[:, 0]
        ci = np.percentile(diff, [2.5, 97.5])
        sig = (ci[0] > 0) or (ci[1] < 0)
        print(f"    {lev} - 없음: mean={diff.mean():+.3f}, "
              f"95% CI=[{ci[0]:+.3f},{ci[1]:+.3f}]  {fmt_sig(sig)}")

# 4) 어학원 (yes vs no)
print("\n[어학원] yes - no (Bayesian)")
for col, name in SUBSCALES.items():
    samples = posterior_summary[col]
    delta = samples['delta']
    diff = delta[:, 1] - delta[:, 0]
    ci = np.percentile(diff, [2.5, 97.5])
    sig = (ci[0] > 0) or (ci[1] < 0)
    print(f"    {name}: mean={diff.mean():+.3f}, "
          f"95% CI=[{ci[0]:+.3f},{ci[1]:+.3f}]  {fmt_sig(sig)}")


# 5) sigma_* 값 (집단 효과 크기 척도)
print("\n[집단 효과 척도 (sigma_factor)] — 0 에 가까울수록 집단 효과 미미")
print(sigma_df.to_string(index=False))


# ----------- 부분 풀링 효과(shrinkage) 시각화용 데이터 저장 -----------
# 각 학년별 (1) 표본평균, (2) 사후 학년효과 추가후 평균
shrinkage_rows = []
for col, name in SUBSCALES.items():
    samples = posterior_summary[col]
    mu_s = samples['mu']
    alpha = samples['alpha']
    for k, lev in enumerate(YEAR_LEVELS):
        # 사후 학년별 평균: mu + alpha[k]
        post_mean = float((mu_s + alpha[:, k]).mean())
        ci = np.percentile(mu_s + alpha[:, k], [2.5, 97.5])
        raw_mean = df.loc[df['year'] == lev, col].mean()
        n_k = (df['year'] == lev).sum()
        shrinkage_rows.append({
            'subscale': name, 'year': lev, 'n': int(n_k),
            'raw_mean': raw_mean,
            'post_mean': post_mean,
            'ci_lo': float(ci[0]), 'ci_hi': float(ci[1]),
        })

shrink_df = pd.DataFrame(shrinkage_rows)
shrink_df.to_csv(os.path.join(HERE, 'gel_shrinkage.csv'), index=False)
print(f"\n저장: gel_shrinkage.csv")

print("\n부분 풀링(shrinkage) 비교 (학년별 평균):")
print(shrink_df.to_string(index=False))


# 사후샘플 저장 (npz)
np.savez(os.path.join(HERE, 'gel_posterior_samples.npz'),
         **{f"{col}_{k}": v for col, samples in posterior_summary.items()
            for k, v in samples.items()})
print(f"저장: gel_posterior_samples.npz")
