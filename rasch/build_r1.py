import json, os

def mk_md(src):
    lines = src.split('\n')
    return {"cell_type":"markdown","metadata":{},"source":lines}

def mk_code(lines):
    if isinstance(lines, str):
        lines = lines.split('\n')
    src = [l+'\n' for l in lines[:-1]] + [lines[-1]]
    return {"cell_type":"code","execution_count":None,"metadata":{},"outputs":[],"source":src}

cells = []

# ── Cell 0: Title & Overview ──
cells.append(mk_md(
"""# Rasch Model (1PL IRT) — Bayesian Estimation with Stan

## 학습 목표 (Learning Objectives)

이 노트북은 **라쉬 모델(Rasch Model)**을 배우는 한국어교육학과 · 글로벌한국학 대학원생을 위한 학습 자료입니다.

1. 라쉬 모델의 수식과 의미를 설명할 수 있다.
2. Stan + CmdStanPy로 MAP 추정과 완전 베이지안 MCMC 추정을 수행할 수 있다.
3. MCMC 수렴 진단 (Rhat, ESS, trace plot)을 해석할 수 있다.
4. Wright Map을 그리고 해석할 수 있다.
5. 사후 예측 검사(PPC)로 모델 타당성을 평가할 수 있다.

---

## 1. 라쉬 모델이란? (What is the Rasch Model?)

라쉬 모델(Rasch, 1960)은 문항 반응 이론(Item Response Theory, IRT)의 가장 기초적인 모델입니다.
이분형(dichotomous) 응답(맞음/틀림, 예/아니오)에 적용됩니다.

$$
P(X_{nj} = 1 \\mid \\theta_n, \\beta_j) = \\frac{\\exp(\\theta_n - \\beta_j)}{1 + \\exp(\\theta_n - \\beta_j)}
$$

| 기호 | 의미 |
|---|---|
| $\\theta_n$ | 피험자 $n$의 **능력(Ability)** (로짓 척도) |
| $\\beta_j$  | 문항 $j$의 **난이도(Item Difficulty)** (로짓 척도) |
| $X_{nj}$   | 피험자 $n$의 문항 $j$ 응답 (0 또는 1) |

**핵심 직관**: $\\theta_n > \\beta_j$이면 정답 확률 > 0.5

---

## 2. 식별 가능성 조건 (Identifiability Condition)

$\\theta_n$과 $\\beta_j$를 모두 같은 상수 $c$만큼 이동해도 확률이 동일합니다.
따라서 척도 원점을 고정해야 합니다: $\\sum_{j=1}^{J} \\beta_j = 0$

Stan 구현: `beta = append_row(beta_free, -sum(beta_free));`

---

## 3. 베이지안 추정의 장점

- **불확실성 정량화**: 전체 사후 분포 획득
- **소표본 안정성**: Prior가 MLE의 무한 추정치 문제 방지
- **PPC**: 모델 타당성의 시각적 검증"""))

# ── Cell 1: imports ──
cells.append(mk_code(
"""import sys, os, warnings
import numpy as np
import matplotlib as mpl
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

warnings.filterwarnings('ignore')

def setup_korean_font():
    candidates = {
        'win32':  [('C:/Windows/Fonts/malgun.ttf', 'Malgun Gothic'),
                   ('C:/Windows/Fonts/gulim.ttc',  'Gulim')],
        'darwin': [('/System/Library/Fonts/AppleSDGothicNeo.ttc', 'Apple SD Gothic Neo'),
                   ('/Library/Fonts/NanumGothic.ttf', 'NanumGothic')],
        'linux':  [('/usr/share/fonts/truetype/nanum/NanumGothic.ttf', 'NanumGothic'),
                   ('/usr/share/fonts/truetype/droid/DroidSansFallback.ttf', 'Droid Sans Fallback')],
    }
    for path, name in candidates.get(sys.platform, candidates.get('linux', [])):
        if os.path.exists(path):
            fm.fontManager.addfont(path)
            mpl.rcParams['font.family'] = [name, 'DejaVu Sans']
            print(f"[Font] {name}")
            return
    mpl.rcParams['font.family'] = ['DejaVu Sans']
    print('[Font] DejaVu Sans (no Korean font found)')

setup_korean_font()
mpl.rcParams['axes.unicode_minus'] = False
mpl.rcParams['figure.dpi'] = 72   # 출력 이미지 크기 절감 (저장 파일 용량 감소)

try:
    import cmdstanpy
    from scipy import stats
    import pandas as pd
    print("All packages imported successfully.")
    print(f"CmdStanPy version: {cmdstanpy.__version__}")
except ImportError as e:
    print(f"Warning: {e}")
    print("Install: pip install cmdstanpy scipy pandas")
    print("Then:    python -m cmdstanpy.install_cmdstan")"""))

# ── Cell 2: Simulation explanation ──
cells.append(mk_md(
"""## 4. 시뮬레이션 데이터 생성 (Simulation Study)

| 설정 | 값 |
|---|---|
| 피험자 수 (N) | 100 |
| 문항 수 (J) | 20 |
| 능력 분포 | $\\theta_n \\sim \\mathcal{N}(0, 1)$ |
| 난이도 분포 | $\\beta_j \\sim \\mathcal{N}(0, 1)$ 후 합=0 정규화 |

시뮬레이션으로 **참 파라미터(true parameters)**를 알고 있는 상황에서
추정 파라미터와 비교해 모델의 **회복 성능(Parameter Recovery)**을 평가합니다."""))

# ── Cell 3: Data generation ──
cells.append(mk_code(
"""import numpy as np

np.random.seed(5101)

N = 100   # 피험자 수
J = 20    # 문항 수

# 참 파라미터 생성
theta_true = np.random.normal(0, 1, N)
beta_raw   = np.random.normal(0, 1, J)
beta_true  = beta_raw - beta_raw.mean()   # sum-to-zero 제약

# 이분형 응답 생성 (Bernoulli sampling)
def rasch_prob(theta, beta):
    return 1.0 / (1.0 + np.exp(-(theta[:, None] - beta[None, :])))

P = rasch_prob(theta_true, beta_true)
Y = (np.random.uniform(size=P.shape) < P).astype(int)

print(f"피험자 수: {N},  문항 수: {J}")
print(f"theta_true: mean={theta_true.mean():.3f}, SD={theta_true.std():.3f}")
print(f"beta_true:  mean={beta_true.mean():.6f}, range=[{beta_true.min():.2f}, {beta_true.max():.2f}]")
print(f"응답 행렬 Y: {Y.shape},  전체 정답률: {Y.mean():.3f}")

# 응답 행렬 + 난이도-정답률 산점도
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

im = axes[0].imshow(Y, aspect='auto', cmap='RdYlGn', interpolation='nearest')
axes[0].set_xlabel('문항 번호')
axes[0].set_ylabel('피험자 번호')
axes[0].set_title('응답 행렬 Y (초록=정답, 빨강=오답)')
plt.colorbar(im, ax=axes[0])

axes[1].scatter(beta_true, Y.mean(axis=0), color='steelblue', s=60, zorder=3)
axes[1].set_xlabel('문항 난이도 (beta_true)')
axes[1].set_ylabel('정답률')
axes[1].set_title('문항 난이도 vs 정답률')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show(); plt.close()
print("데이터 생성 완료")"""))

# ── Cell 4: Stan model explanation ──
cells.append(mk_md(
"""## 5. Stan 모델 정의 (Stan Model Definition)

### Sum-to-zero 식별 조건

Stan에서는 `beta_free` (J-1개)를 파라미터로 두고, 마지막 값을 자동 계산합니다:

```stan
vector[J] beta;
beta = append_row(beta_free, -sum(beta_free));  // sum(beta) = 0 보장
```

### 사전 분포

- $\\theta_n \\sim \\mathcal{N}(0, \\sigma_{\\theta})$  (계층적 사전분포)
- $\\sigma_{\\theta} \\sim \\text{Exponential}(1)$
- $\\beta_j \\sim \\mathcal{N}(0, 2)$  (약한 정보 사전분포)"""))

# ── Cell 5: Stan model code ──
# Using list of lines to avoid triple-quote nesting issue
stan_lines = [
    "# Stan 모델 코드",
    "STAN_CODE = (",
    "    'data {\\n'",
    "    '  int<lower=1> N;\\n'",
    "    '  int<lower=1> J;\\n'",
    "    '  int<lower=1> Nobs;\\n'",
    "    '  array[Nobs] int<lower=1,upper=N> nn;\\n'",
    "    '  array[Nobs] int<lower=1,upper=J> jj;\\n'",
    "    '  array[Nobs] int<lower=0,upper=1> y;\\n'",
    "    '}\\n'",
    "    'parameters {\\n'",
    "    '  vector[N] theta;\\n'",
    "    '  real<lower=0> sigma_theta;\\n'",
    "    '  vector[J-1] beta_free;\\n'",
    "    '}\\n'",
    "    'transformed parameters {\\n'",
    "    '  vector[J] beta;\\n'",
    "    '  beta = append_row(beta_free, -sum(beta_free));\\n'",
    "    '}\\n'",
    "    'model {\\n'",
    "    '  sigma_theta ~ exponential(1);\\n'",
    "    '  theta ~ normal(0, sigma_theta);\\n'",
    "    '  beta_free ~ normal(0, 2);\\n'",
    "    '  for (obs in 1:Nobs)\\n'",
    "    '    y[obs] ~ bernoulli_logit(theta[nn[obs]] - beta[jj[obs]]);\\n'",
    "    '}\\n'",
    "    'generated quantities {\\n'",
    "    '  array[Nobs] int y_rep;\\n'",
    "    '  vector[Nobs] log_lik;\\n'",
    "    '  for (obs in 1:Nobs) {\\n'",
    "    '    real lp = theta[nn[obs]] - beta[jj[obs]];\\n'",
    "    '    y_rep[obs]   = bernoulli_logit_rng(lp);\\n'",
    "    '    log_lik[obs] = bernoulli_logit_lpmf(y[obs] | lp);\\n'",
    "    '  }\\n'",
    "    '}\\n'",
    ")",
    "",
    "# Stan 데이터 딕셔너리",
    "nn_idx, jj_idx, y_flat = [], [], []",
    "for n in range(N):",
    "    for j in range(J):",
    "        nn_idx.append(n + 1)",
    "        jj_idx.append(j + 1)",
    "        y_flat.append(int(Y[n, j]))",
    "",
    "stan_data = {'N': N, 'J': J, 'Nobs': N*J,",
    "             'nn': nn_idx, 'jj': jj_idx, 'y': y_flat}",
    "",
    "stan_file = 'rasch_model.stan'",
    "with open(stan_file, 'w') as f:",
    "    f.write(STAN_CODE)",
    "print(f'Stan 파일 저장: {stan_file}')",
    "print(f'데이터: N={N}, J={J}, Nobs={N*J}')",
]
cells.append(mk_code(stan_lines))

# ── Cell 6: MAP explanation ──
cells.append(mk_md(
"""## 6. MAP 추정 + Laplace 근사

**MAP (Maximum A Posteriori)**: 사후 분포의 최빈값.
Laplace 근사: MAP 주변을 정규 분포로 근사 → 96% 신뢰구간 계산.

| 방법 | 속도 | 불확실성 표현 |
|---|---|---|
| MAP + Laplace | 빠름 | 근사적 (정규분포 가정) |
| Full MCMC | 느림 | 정확 (전체 사후 분포) |"""))

# ── Cell 7: MAP estimation ──
cells.append(mk_code(
"""MAP_SUCCESS = False
try:
    import cmdstanpy
    model = cmdstanpy.CmdStanModel(stan_file=stan_file)
    print("Stan 모델 컴파일 완료")

    map_fit = model.optimize(data=stan_data, algorithm='lbfgs',
                             jacobian=True, seed=5101)

    all_vars = map_fit.optimized_params_dict
    theta_map = np.array([all_vars[f'theta[{n+1}]'] for n in range(N)])
    beta_map  = np.array([all_vars[f'beta[{j+1}]'] for j in range(J)])

    print(f"MAP 완료")
    print(f"theta_map: mean={theta_map.mean():.3f}, SD={theta_map.std():.3f}")
    print(f"beta_map:  range=[{beta_map.min():.2f}, {beta_map.max():.2f}]")
    MAP_SUCCESS = True

except Exception as e:
    print(f"Stan MAP 오류: {e}")
    print("데모 모드: 참값 + 소음으로 대체 (실제 추정이 아님)")
    theta_map = theta_true + np.random.normal(0, 0.15, N)
    beta_map  = beta_true  + np.random.normal(0, 0.15, J)
    beta_map -= beta_map.mean()"""))

# ── Cell 8: MAP scatter plot ──
cells.append(mk_code(
"""# Fisher 정보행렬 기반 SE (Laplace 근사용)
def rasch_se_theta(theta_est, beta_est):
    se = []
    for n in range(len(theta_est)):
        p = 1 / (1 + np.exp(-(theta_est[n] - beta_est)))
        se.append(1.0 / np.sqrt((p*(1-p)).sum() + 1e-10))
    return np.array(se)

def rasch_se_beta(theta_est, beta_est):
    se = []
    for j in range(len(beta_est)):
        p = 1 / (1 + np.exp(-(theta_est - beta_est[j])))
        se.append(1.0 / np.sqrt((p*(1-p)).sum() + 1e-10))
    return np.array(se)

se_theta = rasch_se_theta(theta_map, beta_map)
se_beta  = rasch_se_beta(theta_map, beta_map)
z96 = 2.054  # 96% CI 계수

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
for ax, true_v, est_v, se, lbl, col in [
    (axes[0], theta_true, theta_map, se_theta, 'theta (능력)',  'steelblue'),
    (axes[1], beta_true,  beta_map,  se_beta,  'beta (난이도)', 'coral')]:

    ax.errorbar(true_v, est_v, yerr=z96*se, fmt='o', ms=4, color=col, alpha=0.6,
                ecolor='lightgray', elinewidth=1, capsize=2, label='MAP ± 96% CI')
    lims = [min(true_v.min(), est_v.min())-0.3, max(true_v.max(), est_v.max())+0.3]
    ax.plot(lims, lims, 'r--', lw=1.5, label='완벽 회복선')
    r = np.corrcoef(true_v, est_v)[0,1]
    ax.text(0.05, 0.93, f'r = {r:.3f}', transform=ax.transAxes, fontsize=11,
            color='darkred', bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow'))
    ax.set_xlabel(f'참 {lbl}'); ax.set_ylabel(f'MAP 추정 {lbl}')
    ax.set_title(f'{lbl} 회복 (MAP + 96% CI)')
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

plt.suptitle('MAP 추정: 참 파라미터 vs MAP 추정값', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.show(); plt.close()
print(f"Theta r = {np.corrcoef(theta_true, theta_map)[0,1]:.4f}")
print(f"Beta  r = {np.corrcoef(beta_true, beta_map)[0,1]:.4f}")"""))

# ── Cell 9: MCMC explanation ──
cells.append(mk_md(
"""## 7. 완전 베이지안 MCMC 추정 (Full Bayesian MCMC)

Stan의 **NUTS (No-U-Turn Sampler)** 는 해밀토니안 역학을 이용해
고차원 공간을 효율적으로 탐색합니다.

| 설정 | 값 |
|---|---|
| 체인 수 | 4 |
| 웜업 | 1,000 iterations |
| 샘플링 | 1,000 iterations |
| 총 샘플 | 4,000 |"""))

# ── Cell 10: MCMC ──
cells.append(mk_code(
"""MCMC_SUCCESS = False
try:
    import cmdstanpy
    if 'model' not in dir():
        model = cmdstanpy.CmdStanModel(stan_file=stan_file)

    print("MCMC 샘플링 시작 (1-5분 소요)...")
    mcmc_fit = model.sample(
        data=stan_data, chains=4,
        iter_warmup=1000, iter_sampling=1000,
        seed=5101, show_progress=True, show_console=False
    )
    print("MCMC 완료!")

    theta_samples = mcmc_fit.stan_variable('theta')   # (4000, N)
    beta_samples  = mcmc_fit.stan_variable('beta')    # (4000, J)
    y_rep_samples = mcmc_fit.stan_variable('y_rep')   # (4000, N*J)
    theta_post_mean = theta_samples.mean(axis=0)
    beta_post_mean  = beta_samples.mean(axis=0)
    MCMC_SUCCESS = True
    print(f"theta 사후 평균: mean={theta_post_mean.mean():.3f}")

except Exception as e:
    print(f"Stan MCMC 오류: {e}")
    print("데모 모드: 근사 사후 샘플 (실제 MCMC 아님)")
    n_s = 4000
    theta_post_mean = theta_true + np.random.normal(0, 0.12, N)
    beta_post_mean  = beta_true  + np.random.normal(0, 0.12, J)
    beta_post_mean -= beta_post_mean.mean()
    theta_samples = theta_post_mean + np.random.normal(0, 0.08, (n_s, N))
    beta_samples  = beta_post_mean  + np.random.normal(0, 0.10, (n_s, J))
    beta_samples -= beta_samples.mean(axis=1, keepdims=True)
    p_r = 1/(1+np.exp(-(theta_samples[:,:,None]-beta_samples[:,None,:])))
    y_rep_samples = (np.random.uniform(size=p_r.shape)<p_r).astype(int).reshape(n_s, N*J)"""))

# ── Cell 11: Convergence ──
cells.append(mk_md(
"""## 8. MCMC 수렴 진단 (Convergence Diagnostics)

| 지표 | 기준값 | 의미 |
|---|---|---|
| **R-hat** | < 1.01 | 체인 간 분산 비율 (1에 가까울수록 좋음) |
| **ESS_bulk** | > 400 | 분포 중앙부의 유효 샘플 수 |
| **ESS_tail** | > 400 | 분포 꼬리의 유효 샘플 수 |
| **발산 횟수** | = 0 | NUTS 수치적 불안정 |

**Trace Plot**: 4개 체인이 같은 범위를 안정적으로 탐색하면 수렴된 것입니다."""))

cells.append(mk_code(
"""if MCMC_SUCCESS:
    try:
        diag = mcmc_fit.diagnose()
        print("진단:", diag[:400])
        summary_df = mcmc_fit.summary()
        print("\\nR-hat (theta[1..5]):")
        idx_cols = ['Mean', 'StdDev', 'R_hat', 'ESS_bulk']
        print(summary_df.loc[[f'theta[{i}]' for i in range(1,6)], idx_cols])
    except Exception as e:
        print(f"진단 오류: {e}")

# Trace plot
fig, axes = plt.subplots(2, 3, figsize=(14, 6))
for i in range(3):
    ax = axes[0, i]
    if MCMC_SUCCESS:
        ch_data = mcmc_fit.stan_variable('theta').reshape(4, 1000, N)
        for ch in range(4):
            ax.plot(ch_data[ch, :, i], alpha=0.7, lw=0.8, label=f'Ch{ch+1}')
    else:
        for ch in range(4):
            ax.plot(theta_samples[ch*1000:(ch+1)*1000, i], alpha=0.7, lw=0.8, label=f'Ch{ch+1}')
    ax.set_title(f'theta[{i+1}] Trace'); ax.legend(fontsize=6, loc='upper right')
    ax.grid(True, alpha=0.2); ax.set_xlabel('Iteration')

for j in range(3):
    ax = axes[1, j]
    if MCMC_SUCCESS:
        ch_data = mcmc_fit.stan_variable('beta').reshape(4, 1000, J)
        for ch in range(4):
            ax.plot(ch_data[ch, :, j], alpha=0.7, lw=0.8, label=f'Ch{ch+1}')
    else:
        for ch in range(4):
            ax.plot(beta_samples[ch*1000:(ch+1)*1000, j], alpha=0.7, lw=0.8, label=f'Ch{ch+1}')
    ax.set_title(f'beta[{j+1}] Trace'); ax.legend(fontsize=6, loc='upper right')
    ax.grid(True, alpha=0.2); ax.set_xlabel('Iteration')

plt.suptitle('MCMC Trace Plots (체인 혼합 확인)', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.show(); plt.close()
print("안정적 Trace = 수렴 완료.")"""))

# ── Cell 12: PPC ──
cells.append(mk_md(
"""## 9. 사후 예측 검사 (Posterior Predictive Check, PPC)

사후 분포에서 복제 데이터 $y^{rep}$를 생성해 실제 데이터 $y$와 비교합니다.
잘 일치하면 모델이 데이터를 잘 설명합니다."""))

cells.append(mk_code(
"""n_ppc = min(200, y_rep_samples.shape[0])
y_rep_sub  = y_rep_samples[:n_ppc]
y_obs_flat = Y.flatten()

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# 전체 정답률
ax = axes[0]
rep_means = y_rep_sub.mean(axis=1)
obs_mean  = y_obs_flat.mean()
ax.hist(rep_means, bins=30, color='steelblue', alpha=0.7, edgecolor='white', label='y_rep 정답률')
ax.axvline(obs_mean, color='red', lw=2, ls='--', label=f'y 관측={obs_mean:.3f}')
ax.set_xlabel('정답률'); ax.set_ylabel('빈도'); ax.set_title('PPC: 전체 정답률')
ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

# 피험자 총점
ax = axes[1]
rep_ps = y_rep_sub.reshape(n_ppc, N, J).sum(axis=2)
obs_ps = Y.sum(axis=1)
ax.hist(rep_ps.flatten(), bins=range(J+2), alpha=0.5, color='steelblue',
        density=True, label='y_rep', edgecolor='white')
ax.hist(obs_ps, bins=range(J+2), alpha=0.7, color='coral',
        density=True, label='y 관측', edgecolor='gray')
ax.set_xlabel('피험자 총점'); ax.set_ylabel('밀도'); ax.set_title('PPC: 피험자 총점')
ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

# 문항별 정답률
ax = axes[2]
rep_it = y_rep_sub.reshape(n_ppc, N, J).mean(axis=1)
obs_it = Y.mean(axis=0)
ax.errorbar(range(J), rep_it.mean(axis=0), yerr=rep_it.std(axis=0)*2,
            fmt='o', color='steelblue', capsize=3, alpha=0.7, label='y_rep 평균±2SD')
ax.scatter(range(J), obs_it, color='red', s=50, zorder=5, marker='x', label='y 관측')
ax.set_xlabel('문항 번호'); ax.set_ylabel('정답률'); ax.set_title('PPC: 문항별 정답률')
ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

plt.suptitle('사후 예측 검사 (PPC)', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.show(); plt.close()
print("파란색(복제)이 빨간색(관측)을 잘 감싸면 모델 적합합니다.")"""))

# ── Cell 13: Scatter+Boxplot ──
cells.append(mk_md(
"""## 10. 참 파라미터 vs 사후 분포 비교

- **산점도**: 참값 vs 사후 평균 (96% CI)
- **박스플롯**: 사후 분포와 참값 포함 여부
- **Coverage**: 96% CI 포함률 (목표: ~96%)"""))

cells.append(mk_code(
"""fig = plt.figure(figsize=(16, 10))
gs = gridspec.GridSpec(2, 3, figure=fig)

t_mean = theta_samples.mean(axis=0)
b_mean = beta_samples.mean(axis=0)
t_lo   = np.percentile(theta_samples,  2, axis=0)
t_hi   = np.percentile(theta_samples, 98, axis=0)
b_lo   = np.percentile(beta_samples,   2, axis=0)
b_hi   = np.percentile(beta_samples,  98, axis=0)

# Theta 산점도
ax1 = fig.add_subplot(gs[0, 0])
ax1.errorbar(theta_true, t_mean,
             yerr=[t_mean-t_lo, t_hi-t_mean],
             fmt='o', ms=4, color='steelblue', alpha=0.6,
             ecolor='lightblue', elinewidth=1, capsize=2)
lims = [theta_true.min()-0.3, theta_true.max()+0.3]
ax1.plot(lims, lims, 'r--', lw=1.5)
r = np.corrcoef(theta_true, t_mean)[0,1]
ax1.set_title(f'Theta: 참값 vs 사후 평균 (r={r:.3f})')
ax1.set_xlabel('참 theta'); ax1.set_ylabel('사후 평균'); ax1.grid(True, alpha=0.3)

# Beta 산점도
ax2 = fig.add_subplot(gs[0, 1])
ax2.errorbar(beta_true, b_mean,
             yerr=[b_mean-b_lo, b_hi-b_mean],
             fmt='s', ms=5, color='coral', alpha=0.7,
             ecolor='lightsalmon', elinewidth=1, capsize=2)
lims = [beta_true.min()-0.3, beta_true.max()+0.3]
ax2.plot(lims, lims, 'r--', lw=1.5)
r = np.corrcoef(beta_true, b_mean)[0,1]
ax2.set_title(f'Beta: 참값 vs 사후 평균 (r={r:.3f})')
ax2.set_xlabel('참 beta'); ax2.set_ylabel('사후 평균'); ax2.grid(True, alpha=0.3)

# Coverage
ax3 = fig.add_subplot(gs[0, 2])
cov_t = ((theta_true>=t_lo)&(theta_true<=t_hi)).mean()*100
cov_b = ((beta_true >=b_lo)&(beta_true <=b_hi)).mean()*100
ax3.bar(['Theta', 'Beta'], [cov_t, cov_b],
        color=['steelblue','coral'], alpha=0.8, edgecolor='white')
ax3.axhline(96, color='red', ls='--', lw=1.5, label='96% 목표')
ax3.set_ylabel('Coverage (%)'); ax3.set_title('96% CI Coverage')
ax3.set_ylim(0, 110); ax3.legend(fontsize=9); ax3.grid(True, alpha=0.3, axis='y')
for i, v in enumerate([cov_t, cov_b]):
    ax3.text(i, v+2, f'{v:.1f}%', ha='center', fontweight='bold', fontsize=11)

# Theta 박스플롯
ax4 = fig.add_subplot(gs[1, :2])
idx_t = np.argsort(theta_true)[:20]
bp = ax4.boxplot([theta_samples[:,i] for i in idx_t],
                  patch_artist=True, medianprops={'color':'red','lw':2})
for p in bp['boxes']:
    p.set_facecolor('lightblue'); p.set_alpha(0.7)
ax4.scatter(range(1,21), theta_true[idx_t], color='red', s=40,
            zorder=5, marker='D', label='참값')
ax4.set_xlabel('피험자 (능력 순)'); ax4.set_ylabel('theta')
ax4.set_title('Theta 사후 분포 박스플롯 (처음 20명)'); ax4.legend(fontsize=9)
ax4.grid(True, alpha=0.3)

# Beta 박스플롯
ax5 = fig.add_subplot(gs[1, 2])
idx_b = np.argsort(beta_true)
bp2 = ax5.boxplot([beta_samples[:,j] for j in idx_b],
                   patch_artist=True, medianprops={'color':'red','lw':2})
for p in bp2['boxes']:
    p.set_facecolor('lightsalmon'); p.set_alpha(0.7)
ax5.scatter(range(1,J+1), beta_true[idx_b], color='darkred', s=40,
            zorder=5, marker='D', label='참값')
ax5.set_xlabel('문항 (난이도 순)'); ax5.set_ylabel('beta')
ax5.set_title('Beta 사후 분포 박스플롯'); ax5.legend(fontsize=9)
ax5.grid(True, alpha=0.3)

plt.suptitle('참 파라미터 vs 사후 분포 비교', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.show(); plt.close()
print(f"Theta Coverage: {cov_t:.1f}%, Beta Coverage: {cov_b:.1f}%")"""))

# ── Cell 14: Wright Map ──
cells.append(mk_md(
"""## 11. Wright Map (라이트 맵)

**Wright Map**은 피험자 능력과 문항 난이도를 **같은 로짓 척도**에서 비교합니다.

- **왼쪽**: 피험자 능력($\\theta$) 분포 히스토그램 (가로)
- **오른쪽**: 문항 난이도($\\beta$) 위치 (96% CI 포함)

이상적인 검사: 두 분포가 잘 겹쳐야 변별력이 높습니다."""))

cells.append(mk_code(
"""fig, axes = plt.subplots(1, 2, figsize=(10, 8),
                         gridspec_kw={'width_ratios': [3, 1]})

all_v  = np.concatenate([theta_samples.mean(axis=0), beta_samples.mean(axis=0)])
y_min, y_max = all_v.min()-0.5, all_v.max()+0.5

# 왼쪽: 피험자 능력 히스토그램
ax_l = axes[0]
counts, edges = np.histogram(theta_samples.mean(axis=0), bins=20, range=(y_min,y_max))
centers = (edges[:-1]+edges[1:])/2; h = edges[1]-edges[0]
ax_l.barh(centers, counts, height=h*0.85, color='steelblue', alpha=0.7, edgecolor='white')
ax_l.set_ylim(y_min, y_max); ax_l.invert_xaxis()
ax_l.set_xlabel('피험자 수'); ax_l.set_ylabel('로짓 척도 (Logit)')
ax_l.set_title('피험자 능력 분포')
ax_l.axhline(0, color='gray', ls=':', lw=1); ax_l.grid(True, alpha=0.3, axis='x')

# 오른쪽: 문항 난이도
ax_r = axes[1]
bm = beta_samples.mean(axis=0)
bl = np.percentile(beta_samples,  2, axis=0)
bh = np.percentile(beta_samples, 98, axis=0)
for j in range(J):
    ax_r.errorbar(0.3, bm[j], yerr=[[bm[j]-bl[j]], [bh[j]-bm[j]]],
                  fmt='s', ms=6, color='coral', alpha=0.8,
                  ecolor='lightsalmon', elinewidth=1.5, capsize=3)
    ax_r.text(0.55, bm[j], f'B{j+1}', va='center', ha='left', fontsize=7)
ax_r.set_xlim(0, 1.2); ax_r.set_ylim(y_min, y_max)
ax_r.set_title('문항 난이도'); ax_r.set_xticks([])
ax_r.axhline(0, color='gray', ls=':', lw=1); ax_r.grid(True, alpha=0.3, axis='y')

plt.suptitle('Wright Map: 피험자 능력 vs 문항 난이도', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.show(); plt.close()
print("Wright Map 완성!")"""))

# ── Cell 15: Uncertainty ──
cells.append(mk_md(
"""## 12. 불확실성 분석: 96% 신뢰구간

**베이지안 96% CrI 해석**: "이 파라미터가 [2%, 98%] 구간에 있을 확률이 96%"

이는 빈도주의 95% CI와 달리 파라미터에 대한 직접적인 확률 진술입니다."""))

cells.append(mk_code(
"""fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Theta CI
ax = axes[0]
ord_t = np.argsort(t_mean)
tm_s = t_mean[ord_t]; tl_s = t_lo[ord_t]; th_s = t_hi[ord_t]
x = np.arange(N)
ax.fill_between(x, tl_s, th_s, alpha=0.3, color='steelblue', label='96% CI')
ax.plot(x, tm_s, 'b-', lw=1.5, label='사후 평균')
ax.scatter(x, theta_true[ord_t], color='red', s=15, alpha=0.7, zorder=5, label='참값')
ax.set_xlabel('피험자 (능력 순)'); ax.set_ylabel('theta (로짓)')
ax.set_title('피험자 능력의 96% 신뢰구간'); ax.legend(fontsize=9); ax.grid(True, alpha=0.2)

# Beta CI
ax = axes[1]
ord_b = np.argsort(b_mean)
bm_s = b_mean[ord_b]; bl_s = b_lo[ord_b]; bh_s = b_hi[ord_b]
x_b = np.arange(J)
ax.fill_between(x_b, bl_s, bh_s, alpha=0.3, color='coral', label='96% CI')
ax.plot(x_b, bm_s, 'r-', lw=1.5, label='사후 평균')
ax.scatter(x_b, beta_true[ord_b], color='darkred', s=40, zorder=5, label='참값')
ax.set_xlabel('문항 (난이도 순)'); ax.set_ylabel('beta (로짓)')
ax.set_title('문항 난이도의 96% 신뢰구간'); ax.legend(fontsize=9); ax.grid(True, alpha=0.2)

plt.suptitle('96% 베이지안 신뢰구간 (Credible Interval)', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.show(); plt.close()
ci_w_t = (th_s - tl_s).mean()
ci_w_b = (bh_s - bl_s).mean()
print(f"평균 96% CI 폭 — Theta: {ci_w_t:.3f} logits, Beta: {ci_w_b:.3f} logits")"""))

# ── Cell 16: References ──
cells.append(mk_md(
"""## 13. References / 참고 문헌

### 국제 학술지 (확인된 논문)

1. **Bürkner, P.-C. (2021)**. Bayesian Item Response Modeling in R with brms and Stan. *Journal of Statistical Software, 100*(5), 1–54. https://doi.org/10.18637/jss.v100.i05

2. **Luo, Y., & Jiao, H. (2018)**. Using the Stan Program for Bayesian Item Response Theory. *Educational and Psychological Measurement, 78*(3), 384–408. https://doi.org/10.1177/0013164417693666

3. **Vehtari, A., Gelman, A., Simpson, D., Carpenter, B., & Bürkner, P.-C. (2021)**. Rank-Normalization, Folding, and Localization: An Improved R-hat for Assessing Convergence of MCMC. *Bayesian Analysis, 16*(2), 667–718. https://doi.org/10.1214/20-BA1221

4. **Gelman, A., Vehtari, A., Simpson, D., Margossian, C. C., Carpenter, B., Yao, Y., Kennedy, L., Gabry, J., Bürkner, P.-C., & Modrák, M. (2020)**. Bayesian Workflow. *arXiv:2011.01808*.

5. **Masters, G. N. (1982)**. A Rasch model for partial credit scoring. *Psychometrika, 47*(2), 149–174.

---

### KCI 등재 학술지

다음 논문들은 KCI 포털(www.kci.go.kr)에서 직접 검색하여 전문을 확인할 수 있습니다.

6. **한국어 학습자의 쓰기 수행 평가 신뢰도 분석 — 다국면 라쉬 모형을 사용하여** (KCI 논문번호: ART002387352)

7. **문항반응이론을 활용한 학문 목적 한국어 말하기 평가 문항 및 채점자 특성 분석** (KCI 논문번호: ART002419170)

8. **Rasch 모형을 적용한 문항분석 및 차별기능문항 탐색 — 2021학년도 물리인증제** (KCI 논문번호: ART002889793)

---

> **KCI 검색 방법**: www.kci.go.kr → 논문검색
> 추천 검색어: `라쉬 모형`, `문항반응이론`, `베이지안 문항분석`, `한국어 능력 평가`
"""))

# Build notebook JSON
nb = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10.0"}
    },
    "cells": cells
}

out = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'IRT_R1_Rasch_Model.ipynb')
with open(out, 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

with open(out) as f:
    test = json.load(f)
print(f"IRT_R1_Rasch_Model.ipynb rebuilt: {len(test['cells'])} cells")
print(f"File size: {os.path.getsize(out):,} bytes")
print("JSON valid: OK")
 open(out) as f:
    test = json.load(f)
print(f"IRT_R1_Rasch_Model.ipynb rebuilt: {len(test['cells'])} cells")
print(f"File size: {os.path.getsize(out):,} bytes")
print("JSON valid: OK")
