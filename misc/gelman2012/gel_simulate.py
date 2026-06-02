"""
gel_simulate.py
주월랑(2023) 논문의 요약통계와 Cronbach's alpha = 0.854 를 이용하여
응답자 86명의 설문 데이터를 역시뮬레이션(reverse-simulation)한다.

[중요 설계 결정]
1) 설문조사 응답은 정수(integer) Likert 1-5 척도이다. 따라서 본 시뮬레이션은
   잠재 점수(latent score)를 다변량 정규분포에서 생성한 후, 각 문항(item)
   응답값을 사후 노이즈와 더해 [1,5] 정수로 반올림(round)하여 저장한다.
   이는 실제 설문 응답의 이산성(discreteness)을 보존한다.

2) 하위범주별 점수(P_score, R_score, A_score)는 해당 정수 문항들의 평균이며
   따라서 실수(real-valued)지만 이산값을 가진다. 이는 주월랑 논문 Table 5와 동일.

3) **T_score(쓰기태도, 전체)는 독립적인 잠재 하위범주가 아니라
   세 하위범주(P, R, A) 점수의 평균인 파생(derived) 합성점수(composite score)이다.**
   주월랑 논문도 이를 별도 종속변수로 취급하므로(Table 5-13), 본 코드도 동일하게
   다루지만, Bayesian 모형에서 T_score 결과는 본질적으로 P, R, A 의 공동
   사후분포에서 도출되는 결과의 함수임을 유의해야 한다.

[원논문의 용어 불일치 주의]
주월랑 논문의 초록(184쪽)은 하위범주를 "쓰기인식, 쓰기수행, 쓰기태도"로 표기하지만,
본문(190쪽)과 Table 5의 정확한 용어는 "쓰기인식, 쓰기반응, 수행태도"이며,
"쓰기태도"는 세 하위범주의 평균인 전체 종합점수(M=3.511)이다.
본 시뮬레이션은 본문/Table 의 정확한 용어를 따른다.
"""
import os
import numpy as np
import pandas as pd

RNG_SEED = 20230518
np.random.seed(RNG_SEED)

N = 86
K = 21
HERE = os.path.dirname(os.path.abspath(__file__))


def assign(values, counts):
    arr = np.concatenate([np.repeat(v, c) for v, c in zip(values, counts)])
    np.random.shuffle(arr)
    return arr


years   = assign(['1', '2', '3', '4', 'EX'], [54, 8, 9, 9, 6])
genders = assign(['M', 'F'], [27, 59])
topiks  = assign(['none', 'L3', 'L4', 'L5', 'L6'], [7, 14, 31, 23, 11])
insts   = assign(['yes', 'no'], [69, 17])

df = pd.DataFrame({
    'id': np.arange(N), 'year': years, 'gender': genders,
    'topik': topiks, 'inst': insts,
})

# Target means/SDs
P_BY_GENDER = {'M': (3.741, 0.602), 'F': (4.140, 0.625)}
R_OVERALL = 3.348
R_SD = 0.539
A_BY_YEAR = {
    '1': (3.046, 0.565), '2': (3.479, 0.523),
    '3': (3.444, 0.692), '4': (3.519, 0.475),
    'EX': (2.944, 0.430),
}
R_corr = np.array([
    [1.000, 0.583, 0.281],
    [0.583, 1.000, 0.508],
    [0.281, 0.508, 1.000],
])

mu = np.zeros((N, 3))
sd = np.zeros((N, 3))
for i in range(N):
    g = df.loc[i, 'gender']
    y = df.loc[i, 'year']
    mu[i, 0], sd[i, 0] = P_BY_GENDER[g]
    mu[i, 1], sd[i, 1] = R_OVERALL, R_SD
    mu[i, 2], sd[i, 2] = A_BY_YEAR[y]

L = np.linalg.cholesky(R_corr)
z = np.random.normal(size=(N, 3)) @ L.T
P_latent = mu[:, 0] + sd[:, 0] * z[:, 0]
R_latent = mu[:, 1] + sd[:, 1] * z[:, 1]
A_latent = mu[:, 2] + sd[:, 2] * z[:, 2]


def generate_items_integer(sigma, seed=None):
    """정수 Likert 1-5 문항 응답 생성. x_pj = clip(round(latent + N(0,sigma)), 1, 5)"""
    if seed is not None:
        np.random.seed(seed)
    items = np.zeros((N, K), dtype=int)
    for j in range(0, 4):
        x = P_latent + np.random.normal(0, sigma, N)
        items[:, j] = np.clip(np.round(x).astype(int), 1, 5)
    for j in range(4, 15):
        x = R_latent + np.random.normal(0, sigma, N)
        items[:, j] = np.clip(np.round(x).astype(int), 1, 5)
    for j in range(15, 21):
        x = A_latent + np.random.normal(0, sigma, N)
        items[:, j] = np.clip(np.round(x).astype(int), 1, 5)
    return items


def compute_alpha(items):
    n, k = items.shape
    item_vars = items.var(axis=0, ddof=1)
    total = items.sum(axis=1)
    total_var = total.var(ddof=1)
    if total_var <= 0:
        return np.nan
    return (k / (k - 1)) * (1 - item_vars.sum() / total_var)


# binary search: tune sigma to achieve Cronbach alpha ~ 0.854
TARGET_ALPHA = 0.854
lo, hi = 0.05, 1.5
for _ in range(40):
    mid = (lo + hi) / 2
    items_try = generate_items_integer(mid, seed=RNG_SEED + 999)
    a = compute_alpha(items_try)
    if a > TARGET_ALPHA:
        lo = mid
    else:
        hi = mid
SIGMA_ITEM = (lo + hi) / 2

items = generate_items_integer(SIGMA_ITEM, seed=RNG_SEED + 999)
final_alpha = compute_alpha(items)

# Subscale scores = mean of integer items.
# T_score is a DERIVED composite (mean of P,R,A) -- not an independent latent.
P_score = items[:, 0:4].mean(axis=1)
R_score = items[:, 4:15].mean(axis=1)
A_score = items[:, 15:21].mean(axis=1)
T_score = (P_score + R_score + A_score) / 3.0  # composite

df['P_score'] = P_score
df['R_score'] = R_score
df['A_score'] = A_score
df['T_score'] = T_score

for j in range(K):
    df[f'q{j+1:02d}'] = items[:, j]

# Diagnostics
print("=== Simulation diagnostics (Integer Likert) ===")
print(f"N = {N}, items = {K}")
print(f"SIGMA_ITEM tuned = {SIGMA_ITEM:.4f}")
print(f"Cronbach alpha   = {final_alpha:.4f} (target {TARGET_ALPHA})")
print(f"Item dtype = {items.dtype}, range = [{items.min()}, {items.max()}]")
print(f"Item unique vals = {sorted(np.unique(items).tolist())}")

print("\n=== Subscale means/SDs (computed from integer items) ===")
for col, label, tM, tSD in [
    ('P_score', 'WP', 4.015, 0.642),
    ('R_score', 'WR', 3.348, 0.539),
    ('A_score', 'PA', 3.171, 0.585),
    ('T_score', 'TOT', 3.511, 0.469),
]:
    m, s = df[col].mean(), df[col].std(ddof=1)
    print(f"  {label:5s}: sim M={m:.3f} SD={s:.3f}  |  paper M={tM:.3f} SD={tSD:.3f}")

print("\n=== Inter-subscale correlations ===")
print(df[['P_score','R_score','A_score']].corr().round(3))

print("\n=== year counts ===")
print(df['year'].value_counts())

print("\n=== Sample item response distributions (q01, q05, q16) ===")
for q in ['q01', 'q05', 'q16']:
    vc = df[q].value_counts().sort_index().to_dict()
    print(f"  {q}: {vc}")

df.to_csv(os.path.join(HERE, 'gel_data.csv'), index=False)
print(f"\nSaved: gel_data.csv")

with open(os.path.join(HERE, 'gel_meta.txt'), 'w') as f:
    f.write(f"N = {N}\n")
    f.write(f"K = {K}\n")
    f.write(f"seed = {RNG_SEED}\n")
    f.write(f"sigma_item = {SIGMA_ITEM}\n")
    f.write(f"cronbach_alpha = {final_alpha}\n")
    f.write("item_type = integer Likert 1-5\n")
    f.write("subscale_type = real-valued (mean of integer items)\n")
    f.write("T_score_note = derived composite = mean(P_score, R_score, A_score)\n")
