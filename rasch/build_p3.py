"""Rebuild IRT_P3_MFRM_Model.ipynb with N_Eff fix."""
import json, os

BASE = os.path.dirname(os.path.abspath(__file__))

def mk_md(src):
    return {"cell_type": "markdown", "metadata": {}, "source": src.split('\n')}

def mk_code(lines):
    if isinstance(lines, str):
        lines = lines.split('\n')
    src = [l + '\n' for l in lines[:-1]] + [lines[-1]]
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": src}

cells = []

# ── Cell 0: Title ──
cells.append(mk_md(
"""# Many-Facet Rasch Model (MFRM) — Bayesian Estimation with Stan

## 학습 목표 (Learning Objectives)

이 노트북은 **다국면 라쉬 모형(MFRM: Many-Facet Rasch Model)**을 배우는
한국어교육학과 · 글로벌한국학 대학원생을 위한 학습 자료입니다.

1. MFRM의 구조와 적용 맥락(채점자가 있는 수행 평가)을 설명할 수 있다.
2. Stan + CmdStanPy로 MFRM의 MAP 및 MCMC 추정을 수행할 수 있다.
3. 채점자 엄격성(Rater Severity) 파라미터를 해석할 수 있다.
4. 범주 확률 곡선(Category Probability Curves)을 그리고 해석할 수 있다.
5. Wright Map을 통해 피험자·문항·채점자를 동일 척도에서 비교할 수 있다.

---

## 1. MFRM이란? (What is the Many-Facet Rasch Model?)

Linacre(1989)가 제안한 MFRM은 기본 라쉬 모델을 확장하여
**채점자(Rater)**와 같은 추가 "국면(Facet)"을 모형에 포함합니다.

$$
\\log \\frac{P(X_{nir} = k)}{P(X_{nir} = k-1)} = \\theta_n - \\beta_i - \\rho_r - \\tau_k
$$

| 파라미터 | 의미 |
|---|---|
| $\\theta_n$ | 피험자 $n$의 **능력** |
| $\\beta_i$  | 문항 $i$의 **난이도** |
| $\\rho_r$   | 채점자 $r$의 **엄격성** (양수=엄격, 음수=관대) |
| $\\tau_k$   | 점수 $k$로의 전환 **단계 난이도** (PCM 구조) |

---

## 2. 시뮬레이션 설계

| 설정 | 값 |
|---|---|
| 피험자 수 (N) | 100 |
| 문항 수 (I) | 4 |
| 채점자 수 (R) | 5 |
| 리커트 점수 (K) | 0, 1, 2 (단계 파라미터 2개) |
| 관측치 수 | N × I × R = 2,000 |

**문항 1**: 잘 설계된 문항 — 단계 난이도 순서가 올바름 ($\\tau_1 < \\tau_2$)
**교육적 의미**: 채점자 엄격성을 통제하지 않으면 피험자 능력 추정이 왜곡됩니다.

---

## 3. 식별 가능성 조건

세 파라미터 그룹 모두 합=0 제약:
$$\\sum_i \\beta_i = 0, \\quad \\sum_r \\rho_r = 0, \\quad \\sum_k \\tau_k = 0$$"""))

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
                   ('C:/Windows/Fonts/gulim.ttc', 'Gulim')],
        'darwin': [('/System/Library/Fonts/AppleSDGothicNeo.ttc', 'Apple SD Gothic Neo')],
        'linux':  [('/usr/share/fonts/truetype/nanum/NanumGothic.ttf', 'NanumGothic')],
    }
    for path, name in candidates.get(sys.platform, candidates.get('linux', [])):
        if os.path.exists(path):
            fm.fontManager.addfont(path)
            mpl.rcParams['font.family'] = [name, 'DejaVu Sans']
            print(f"[Font] {name}")
            return
    mpl.rcParams['font.family'] = ['DejaVu Sans']
    print('[Font] DejaVu Sans')

setup_korean_font()
mpl.rcParams['axes.unicode_minus'] = False

try:
    import cmdstanpy, pandas as pd
    print(f"CmdStanPy {cmdstanpy.__version__} ready")
except ImportError as e:
    print(f"Warning: {e}. Install: pip install cmdstanpy")"""))

# ── Cell 2: Simulation ──
cells.append(mk_code(
"""import numpy as np
np.random.seed(5101)

N, I, R, K = 100, 4, 5, 3   # persons, items, raters, score levels (0..K-1)

# True parameters (sum-to-zero)
theta_true = np.random.normal(0, 1, N)

beta_raw  = np.array([-1.0, -0.3, 0.3, 1.0])
beta_true = beta_raw - beta_raw.mean()

rho_raw  = np.array([-0.8, -0.4, 0.0, 0.4, 0.8])
rho_true = rho_raw - rho_raw.mean()

# Shared step difficulties (K-1 = 2 params, sum-to-zero)
# Item 1 designed to have ordered steps (tau[0] < tau[1])
tau_raw   = np.array([-0.6, 0.6])
tau_true  = tau_raw - tau_raw.mean()

print("True parameters:")
print(f"  theta: mean={theta_true.mean():.3f}, SD={theta_true.std():.3f}")
print(f"  beta:  {np.round(beta_true, 3)}  (sum={beta_true.sum():.6f})")
print(f"  rho:   {np.round(rho_true, 3)}  (sum={rho_true.sum():.6f})")
print(f"  tau:   {np.round(tau_true, 3)}  (sum={tau_true.sum():.6f})")

# Generate PCM responses
def pcm_probs(logit_cumulative):
    \"\"\"Convert cumulative logits to category probabilities (PCM).\"\"\"
    K_loc = len(logit_cumulative) + 1
    unnorm = np.zeros(K_loc)
    unnorm[0] = 0.0
    for k in range(1, K_loc):
        unnorm[k] = unnorm[k-1] + logit_cumulative[k-1]
    unnorm -= unnorm.max()
    exp_u = np.exp(unnorm)
    return exp_u / exp_u.sum()

Y_obs = np.zeros((N, I, R), dtype=int)
for n in range(N):
    for i in range(I):
        for r in range(R):
            cum_logits = [theta_true[n] - beta_true[i] - rho_true[r] - tau_true[k]
                          for k in range(K-1)]
            probs = pcm_probs(cum_logits)
            Y_obs[n, i, r] = np.random.choice(K, p=probs)

print(f"\\nY_obs shape: {Y_obs.shape}, mean score: {Y_obs.mean():.3f}")"""))

# ── Cell 3: Visualize data ──
cells.append(mk_code(
"""fig, axes = plt.subplots(1, 3, figsize=(14, 4))

# Score distribution
ax = axes[0]
scores, counts = np.unique(Y_obs, return_counts=True)
ax.bar(scores, counts/(N*I*R), color=['#e74c3c','#f39c12','#2ecc71'], edgecolor='white')
ax.set_xlabel('점수'); ax.set_ylabel('비율')
ax.set_title('전체 점수 분포')
ax.grid(True, alpha=0.3, axis='y')

# Person mean scores
ax = axes[1]
person_means = Y_obs.mean(axis=(1, 2))
ax.hist(person_means, bins=20, color='steelblue', edgecolor='white', alpha=0.8)
ax.set_xlabel('피험자 평균 점수'); ax.set_ylabel('빈도')
ax.set_title('피험자별 평균 점수 분포')
ax.grid(True, alpha=0.3)

# Rater means (severity check)
ax = axes[2]
rater_means = Y_obs.mean(axis=(0, 1))
ax.bar(range(1, R+1), rater_means, color='coral', edgecolor='white', alpha=0.8)
ax.axhline(Y_obs.mean(), color='gray', ls='--', lw=1.5, label=f'전체 평균={Y_obs.mean():.2f}')
ax.set_xlabel('채점자'); ax.set_ylabel('평균 부여 점수')
ax.set_title('채점자별 평균 점수\\n(낮을수록 엄격한 채점자)')
ax.legend(fontsize=9); ax.grid(True, alpha=0.3, axis='y')

plt.suptitle('MFRM 시뮬레이션 데이터 요약', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.show()
print("데이터 탐색 완료")"""))

# ── Cell 4: Stan model ──
stan_lines = [
    "# MFRM Stan 모델 코드",
    "MFRM_STAN = (",
    "    'data {\\n'",
    "    '  int<lower=1> N; int<lower=1> I; int<lower=1> R; int<lower=2> K;\\n'",
    "    '  int<lower=1> Nobs;\\n'",
    "    '  array[Nobs] int<lower=1,upper=N> nn;\\n'",
    "    '  array[Nobs] int<lower=1,upper=I> ii;\\n'",
    "    '  array[Nobs] int<lower=1,upper=R> rr;\\n'",
    "    '  array[Nobs] int<lower=0,upper=K-1> y;\\n'",
    "    '}\\n'",
    "    'parameters {\\n'",
    "    '  vector[N] theta;\\n'",
    "    '  real<lower=0> sigma_theta;\\n'",
    "    '  vector[I-1] beta_free;\\n'",
    "    '  vector[R-1] rho_free;\\n'",
    "    '  vector[K-2] tau_free;\\n'",
    "    '}\\n'",
    "    'transformed parameters {\\n'",
    "    '  vector[I] beta = append_row(beta_free, -sum(beta_free));\\n'",
    "    '  vector[R] rho  = append_row(rho_free,  -sum(rho_free));\\n'",
    "    '  vector[K-1] tau = append_row(tau_free, -sum(tau_free));\\n'",
    "    '}\\n'",
    "    'model {\\n'",
    "    '  sigma_theta ~ exponential(1);\\n'",
    "    '  theta ~ normal(0, sigma_theta);\\n'",
    "    '  beta_free ~ normal(0, 2);\\n'",
    "    '  rho_free  ~ normal(0, 2);\\n'",
    "    '  tau_free  ~ normal(0, 2);\\n'",
    "    '  for (obs in 1:Nobs) {\\n'",
    "    '    vector[K] log_p;\\n'",
    "    '    log_p[1] = 0;\\n'",
    "    '    for (k in 2:K)\\n'",
    "    '      log_p[k] = log_p[k-1] + (theta[nn[obs]] - beta[ii[obs]] - rho[rr[obs]] - tau[k-1]);\\n'",
    "    '    y[obs] ~ categorical_logit(log_p);\\n'",
    "    '  }\\n'",
    "    '}\\n'",
    "    'generated quantities {\\n'",
    "    '  array[Nobs] int y_rep;\\n'",
    "    '  for (obs in 1:Nobs) {\\n'",
    "    '    vector[K] log_p;\\n'",
    "    '    log_p[1] = 0;\\n'",
    "    '    for (k in 2:K)\\n'",
    "    '      log_p[k] = log_p[k-1] + (theta[nn[obs]] - beta[ii[obs]] - rho[rr[obs]] - tau[k-1]);\\n'",
    "    '    y_rep[obs] = categorical_logit_rng(log_p) - 1;\\n'",
    "    '  }\\n'",
    "    '}\\n'",
    ")",
    "",
    "# Stan 데이터 딕셔너리",
    "nn_list, ii_list, rr_list, y_list = [], [], [], []",
    "for n in range(N):",
    "    for i in range(I):",
    "        for r in range(R):",
    "            nn_list.append(n+1); ii_list.append(i+1)",
    "            rr_list.append(r+1); y_list.append(int(Y_obs[n,i,r]))",
    "",
    "stan_data = {'N':N,'I':I,'R':R,'K':K,'Nobs':N*I*R,",
    "             'nn':nn_list,'ii':ii_list,'rr':rr_list,'y':y_list}",
    "",
    "stan_file = 'mfrm_model.stan'",
    "with open(stan_file, 'w') as f: f.write(MFRM_STAN)",
    "print(f'Stan 파일 저장: {stan_file}, Nobs={N*I*R}')",
]
cells.append(mk_code(stan_lines))

# ── Cell 5: MAP ──
cells.append(mk_md(
"""## 4. MAP 추정 (MAP Estimation)

`optimize()` 로 빠른 MAP 추정을 수행합니다."""))

cells.append(mk_code(
"""MAP_SUCCESS = False
try:
    import cmdstanpy
    model = cmdstanpy.CmdStanModel(stan_file=stan_file)
    print("Stan 컴파일 완료")
    map_fit = model.optimize(data=stan_data, algorithm='lbfgs', jacobian=True, seed=5101)
    v = map_fit.optimized_params_dict
    theta_map = np.array([v[f'theta[{n+1}]'] for n in range(N)])
    beta_map   = np.array([v[f'beta[{i+1}]']  for i in range(I)])
    rho_map    = np.array([v[f'rho[{r+1}]']   for r in range(R)])
    tau_map    = np.array([v[f'tau[{k+1}]']   for k in range(K-1)])
    print(f"MAP 완료  |  beta: {np.round(beta_map,3)}  |  rho: {np.round(rho_map,3)}")
    MAP_SUCCESS = True
except Exception as e:
    print(f"Stan MAP 오류: {e} — 데모 모드 사용")
    theta_map = theta_true + np.random.normal(0, 0.15, N)
    beta_map  = beta_true  + np.random.normal(0, 0.15, I); beta_map -= beta_map.mean()
    rho_map   = rho_true   + np.random.normal(0, 0.10, R); rho_map  -= rho_map.mean()
    tau_map   = tau_true   + np.random.normal(0, 0.10, K-1); tau_map -= tau_map.mean()"""))

# ── Cell 6: MAP scatter ──
cells.append(mk_code(
"""# MAP 산점도 (참값 vs MAP 추정값)
fig, axes = plt.subplots(1, 3, figsize=(13, 4))
z96 = 2.054

for ax, true_v, est_v, lbl, col in [
    (axes[0], theta_true, theta_map, 'theta (능력)',   'steelblue'),
    (axes[1], beta_true,  beta_map,  'beta (문항)',    'coral'),
    (axes[2], rho_true,   rho_map,   'rho (채점자)',   'green')]:

    ax.scatter(true_v, est_v, color=col, alpha=0.7, s=40)
    lims = [min(true_v.min(), est_v.min())-0.3, max(true_v.max(), est_v.max())+0.3]
    ax.plot(lims, lims, 'r--', lw=1.5, label='완벽 회복선')
    r = np.corrcoef(true_v, est_v)[0,1]
    ax.text(0.05, 0.93, f'r = {r:.3f}', transform=ax.transAxes, fontsize=11,
            color='darkred', bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow'))
    ax.set_xlabel(f'참 {lbl}'); ax.set_ylabel(f'MAP 추정')
    ax.set_title(f'{lbl} 회복'); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

plt.suptitle('MAP 추정: 참 파라미터 vs MAP 추정값', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.show()"""))

# ── Cell 7: MCMC ──
cells.append(mk_md(
"""## 5. 완전 MCMC 추정 (Full Bayesian Sampling)

| 설정 | 값 |
|---|---|
| 체인 | 4 |
| 웜업 | 1,000 |
| 샘플링 | 1,000 |
| 총 샘플 | 4,000 |"""))

cells.append(mk_code(
"""MCMC_SUCCESS = False
try:
    import cmdstanpy
    if 'model' not in dir():
        model = cmdstanpy.CmdStanModel(stan_file=stan_file)
    print("MCMC 시작 (2-5분 소요)...")
    fit = model.sample(data=stan_data, chains=4,
                       iter_warmup=1000, iter_sampling=1000,
                       seed=5101, show_progress=True, show_console=False)
    print("MCMC 완료!")
    theta_samples = fit.stan_variable('theta')   # (4000, N)
    beta_samples  = fit.stan_variable('beta')    # (4000, I)
    rho_samples   = fit.stan_variable('rho')     # (4000, R)
    tau_samples   = fit.stan_variable('tau')     # (4000, K-1)
    y_rep_samples = fit.stan_variable('y_rep')   # (4000, N*I*R)
    MCMC_SUCCESS = True
    print(f"rho (채점자 엄격성) 사후 평균: {rho_samples.mean(axis=0).round(3)}")
except Exception as e:
    print(f"Stan MCMC 오류: {e} — 데모 모드")
    n_s = 4000
    theta_samples = theta_true + np.random.normal(0, 0.12, (n_s, N))
    beta_samples  = beta_true  + np.random.normal(0, 0.15, (n_s, I))
    beta_samples -= beta_samples.mean(axis=1, keepdims=True)
    rho_samples   = rho_true   + np.random.normal(0, 0.10, (n_s, R))
    rho_samples  -= rho_samples.mean(axis=1, keepdims=True)
    tau_samples   = tau_true   + np.random.normal(0, 0.10, (n_s, K-1))
    tau_samples  -= tau_samples.mean(axis=1, keepdims=True)
    # Fake y_rep
    y_rep_samples = np.zeros((n_s, N*I*R), dtype=int)"""))

# ── Cell 8: Convergence (N_Eff fixed) ──
cells.append(mk_md(
"""## 6. MCMC 수렴 진단 (Convergence Diagnostics)

R-hat < 1.01, ESS > 400이면 수렴으로 간주합니다."""))

cells.append(mk_code(
"""if MCMC_SUCCESS:
    try:
        summary = fit.summary()
        # cmdstanpy 버전에 따라 열 이름이 다를 수 있음
        ess_col = 'N_Eff' if 'N_Eff' in summary.columns else 'ESS_bulk'

        theta_s = summary.loc[[f'theta[{i}]' for i in range(1, min(6,N+1))]]
        beta_s  = summary.loc[[f'beta[{i}]'  for i in range(1, I+1)]]
        rho_s   = summary.loc[[f'rho[{i}]'   for i in range(1, R+1)]]
        tau_s   = summary.loc[[f'tau[{i}]'   for i in range(1, K)]]

        print("Convergence summary:")
        for name, df in [('theta', theta_s), ('beta', beta_s),
                         ('rho', rho_s), ('tau', tau_s)]:
            print(f"  {name:5s}: Rhat max={df['R_hat'].max():.4f}, "
                  f"ESS min={df[ess_col].min():.0f}")
    except Exception as e:
        print(f"Summary error: {e}")
        try:
            print(fit.diagnose())
        except:
            pass

# Trace plots
fig, axes = plt.subplots(2, 3, figsize=(14, 6))
titles = ['theta[1]','theta[2]','theta[3]','beta[1]','rho[1]','tau[1]']
for ax, title in zip(axes.flat, titles):
    param_name = title.replace('[','[').replace(']',']')
    # Extract parameter name and index
    base = title[:title.index('[')]
    idx  = int(title[title.index('[')+1:title.index(']')]) - 1
    param_map = {'theta': theta_samples, 'beta': beta_samples,
                 'rho': rho_samples, 'tau': tau_samples}
    samples_arr = param_map[base]
    for ch in range(4):
        ax.plot(samples_arr[ch*1000:(ch+1)*1000, idx], alpha=0.7, lw=0.8, label=f'Ch{ch+1}')
    ax.set_title(f'{title} Trace'); ax.legend(fontsize=6, loc='upper right')
    ax.grid(True, alpha=0.2); ax.set_xlabel('Iteration')

plt.suptitle('MCMC Trace Plots', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.show()"""))

# ── Cell 9: PPC ──
cells.append(mk_md(
"""## 7. 사후 예측 검사 (PPC)

복제 데이터의 점수 분포가 실제 데이터와 얼마나 일치하는지 확인합니다."""))

cells.append(mk_code(
"""n_ppc = min(200, y_rep_samples.shape[0])
y_rep_sub = y_rep_samples[:n_ppc]
y_obs_flat = Y_obs.flatten()

fig, axes = plt.subplots(1, 3, figsize=(14, 4))

# Score distribution
ax = axes[0]
for score in range(K):
    rep_rate = (y_rep_sub == score).mean(axis=1)
    obs_rate = (y_obs_flat == score).mean()
    ax.hist(rep_rate, bins=25, alpha=0.4, label=f'y_rep 점수={score}', edgecolor='white')
    ax.axvline(obs_rate, lw=2, ls='--', label=f'y_obs 점수={score}: {obs_rate:.3f}')
ax.set_xlabel('비율'); ax.set_title('PPC: 점수별 비율'); ax.legend(fontsize=7)
ax.grid(True, alpha=0.3)

# Person total
ax = axes[1]
rep_pts = y_rep_sub.reshape(n_ppc, N, I*R).sum(axis=2)
obs_pts = Y_obs.reshape(N, I*R).sum(axis=1)
ax.hist(rep_pts.flatten(), bins=range(I*R*(K-1)+2), alpha=0.5, color='steelblue',
        density=True, label='y_rep')
ax.hist(obs_pts, bins=range(I*R*(K-1)+2), alpha=0.7, color='coral',
        density=True, label='y_obs', edgecolor='gray')
ax.set_xlabel('총점'); ax.set_title('PPC: 피험자 총점'); ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# Rater means
ax = axes[2]
rep_rater = y_rep_sub.reshape(n_ppc, N, I, R).mean(axis=(1, 2))
obs_rater = Y_obs.mean(axis=(0, 1))
ax.errorbar(range(R), rep_rater.mean(axis=0), yerr=rep_rater.std(axis=0)*2,
            fmt='o', color='steelblue', capsize=3, alpha=0.7, label='y_rep 평균±2SD')
ax.scatter(range(R), obs_rater, color='red', s=60, zorder=5, marker='x', label='y_obs')
ax.set_xlabel('채점자'); ax.set_title('PPC: 채점자별 평균 점수'); ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

plt.suptitle('사후 예측 검사 (PPC)', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.show()"""))

# ── Cell 10: Scatter+Boxplot ──
cells.append(mk_md(
"""## 8. 참 파라미터 vs 사후 분포

산점도 (참값 vs 사후 평균)와 96% CI 커버리지를 확인합니다."""))

cells.append(mk_code(
"""fig, axes = plt.subplots(2, 3, figsize=(15, 8))

params = [
    (theta_true, theta_samples, 'theta (N=100)', 'steelblue', axes[0,0]),
    (beta_true,  beta_samples,  'beta (I=4)',    'coral',     axes[0,1]),
    (rho_true,   rho_samples,   'rho (R=5)',     'green',     axes[0,2]),
    (tau_true,   tau_samples,   'tau (K-1=2)',   'purple',    axes[1,0]),
]

for true_v, samps, lbl, col, ax in params:
    post_m = samps.mean(axis=0)
    post_lo = np.percentile(samps,  2, axis=0)
    post_hi = np.percentile(samps, 98, axis=0)
    ax.errorbar(true_v, post_m, yerr=[post_m-post_lo, post_hi-post_m],
                fmt='o', ms=5, color=col, alpha=0.7,
                ecolor='lightgray', elinewidth=1, capsize=3)
    lims = [min(true_v.min(),post_m.min())-0.3, max(true_v.max(),post_m.max())+0.3]
    ax.plot(lims, lims, 'r--', lw=1.5)
    r = np.corrcoef(true_v, post_m)[0,1]
    cov = ((true_v>=post_lo)&(true_v<=post_hi)).mean()*100
    ax.set_title(f'{lbl}: r={r:.3f}, cov={cov:.0f}%')
    ax.set_xlabel('참값'); ax.set_ylabel('사후 평균 ± 96% CI')
    ax.grid(True, alpha=0.3)

# Rho 박스플롯 (채점자 엄격성)
ax = axes[1, 1]
bp = ax.boxplot([rho_samples[:, r] for r in range(R)],
                patch_artist=True, medianprops={'color':'red','lw':2})
for p in bp['boxes']:
    p.set_facecolor('lightgreen'); p.set_alpha(0.7)
ax.scatter(range(1, R+1), rho_true, color='red', s=50, zorder=5, marker='D', label='참값')
ax.axhline(0, color='gray', ls=':', lw=1)
ax.set_xlabel('채점자'); ax.set_ylabel('rho (로짓)'); ax.set_title('채점자 엄격성 분포')
ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

# Summary table
ax = axes[1, 2]
ax.axis('off')
summary_data = [['파라미터', '참값 범위', '사후 평균 r', '96% Coverage']]
for true_v, samps, lbl, _, _ in params:
    pm = samps.mean(axis=0)
    pl = np.percentile(samps, 2, axis=0)
    ph = np.percentile(samps, 98, axis=0)
    r = np.corrcoef(true_v, pm)[0,1]
    cov = ((true_v>=pl)&(true_v<=ph)).mean()*100
    summary_data.append([lbl, f'[{true_v.min():.2f}, {true_v.max():.2f}]',
                         f'{r:.3f}', f'{cov:.1f}%'])
tbl = ax.table(cellText=summary_data[1:], colLabels=summary_data[0],
               loc='center', cellLoc='center')
tbl.auto_set_font_size(False); tbl.set_fontsize(10)
tbl.scale(1, 2)
ax.set_title('파라미터 회복 요약', fontsize=11)

plt.suptitle('MFRM: 참 파라미터 vs 사후 분포', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.show()"""))

# ── Cell 11: Category Probability Curves ──
cells.append(mk_md(
"""## 9. 범주 확률 곡선 (Category Probability Curves)

각 문항-채점자 조합에서 점수 k를 받을 확률을 로짓 척도의 함수로 시각화합니다.
**문항 1**은 순서가 올바른 단계 난이도($\\tau_1 < \\tau_2$)를 가집니다."""))

cells.append(mk_code(
"""beta_pm = beta_samples.mean(axis=0)
tau_pm  = tau_samples.mean(axis=0)
# Use rho=0 (average rater) for display

fig, axes = plt.subplots(1, I, figsize=(14, 4))
logit_range = np.linspace(-4, 4, 200)
colors = ['#e74c3c', '#f39c12', '#2ecc71']

for i in range(I):
    ax = axes[i]
    for theta_val in logit_range:
        cum_logits = [theta_val - beta_pm[i] - tau_pm[k] for k in range(K-1)]
        unnorm = np.zeros(K); unnorm[0] = 0.0
        for k in range(1, K):
            unnorm[k] = unnorm[k-1] + cum_logits[k-1]
        unnorm -= unnorm.max()
        exp_u = np.exp(unnorm)
        probs = exp_u / exp_u.sum()
        for k in range(K):
            ax.plot(theta_val, probs[k], 'o', color=colors[k], ms=1, alpha=0.5)

    # Draw smooth curves
    all_probs = np.zeros((len(logit_range), K))
    for ti, theta_val in enumerate(logit_range):
        cum_logits = [theta_val - beta_pm[i] - tau_pm[k] for k in range(K-1)]
        unnorm = np.zeros(K); unnorm[0] = 0.0
        for k in range(1, K):
            unnorm[k] = unnorm[k-1] + cum_logits[k-1]
        unnorm -= unnorm.max()
        exp_u = np.exp(unnorm)
        all_probs[ti] = exp_u / exp_u.sum()

    for k in range(K):
        ax.plot(logit_range, all_probs[:, k], color=colors[k], lw=2,
                label=f'P(X={k})')

    ax.axvline(beta_pm[i], color='gray', ls='--', lw=1)
    ax.set_xlabel('능력 (theta, 로짓)')
    ax.set_ylabel('확률') if i == 0 else None
    ordered = "순서 정렬" if i == 0 else "공유 단계"
    ax.set_title(f'문항 {i+1} (beta={beta_pm[i]:.2f})\\n[{ordered}]')
    ax.legend(fontsize=8, loc='center')
    ax.grid(True, alpha=0.3)

plt.suptitle('MFRM 범주 확률 곡선 (Category Probability Curves)', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.show()"""))

# ── Cell 12: Wright Map ──
cells.append(mk_md(
"""## 10. Wright Map (3-Panel)

피험자 능력, 문항 단계 위치, 채점자 엄격성을 동일 로짓 척도에서 표시합니다."""))

cells.append(mk_code(
"""fig, axes = plt.subplots(1, 3, figsize=(12, 9),
                         gridspec_kw={'width_ratios': [3, 1, 1]})

theta_pm = theta_samples.mean(axis=0)
all_v = np.concatenate([theta_pm, beta_pm + tau_pm.max(), rho_samples.mean(axis=0)])
y_min, y_max = all_v.min()-0.5, all_v.max()+0.5

# Panel 1: Person ability
ax = axes[0]
counts, edges = np.histogram(theta_pm, bins=20, range=(y_min, y_max))
centers = (edges[:-1]+edges[1:])/2; h = edges[1]-edges[0]
ax.barh(centers, counts, height=h*0.85, color='steelblue', alpha=0.7, edgecolor='white')
ax.set_ylim(y_min, y_max); ax.invert_xaxis()
ax.set_xlabel('피험자 수'); ax.set_ylabel('로짓 척도')
ax.set_title('피험자 능력'); ax.axhline(0, color='gray', ls=':', lw=1)
ax.grid(True, alpha=0.3, axis='x')

# Panel 2: Item + step locations
ax = axes[1]
for i in range(I):
    for k in range(K-1):
        loc = beta_pm[i] + tau_pm[k]
        ax.plot(0.3, loc, 's', ms=8, color='coral', alpha=0.8)
        ax.text(0.55, loc, f'I{i+1}.T{k+1}', va='center', ha='left', fontsize=7)
ax.set_xlim(0, 1.5); ax.set_ylim(y_min, y_max); ax.set_xticks([])
ax.set_title('문항+단계'); ax.axhline(0, color='gray', ls=':', lw=1)
ax.grid(True, alpha=0.3, axis='y')

# Panel 3: Rater severity
ax = axes[2]
rho_pm = rho_samples.mean(axis=0)
rho_lo  = np.percentile(rho_samples,  2, axis=0)
rho_hi  = np.percentile(rho_samples, 98, axis=0)
for r in range(R):
    ax.errorbar(0.3, rho_pm[r], yerr=[[rho_pm[r]-rho_lo[r]], [rho_hi[r]-rho_pm[r]]],
                fmt='^', ms=7, color='green', alpha=0.8,
                ecolor='lightgreen', elinewidth=2, capsize=4)
    lbl = '엄격' if rho_pm[r] > 0 else '관대'
    ax.text(0.55, rho_pm[r], f'R{r+1} ({lbl})', va='center', ha='left', fontsize=7)
ax.set_xlim(0, 1.5); ax.set_ylim(y_min, y_max); ax.set_xticks([])
ax.set_title('채점자 엄격성'); ax.axhline(0, color='gray', ls=':', lw=1)
ax.grid(True, alpha=0.3, axis='y')

plt.suptitle('Wright Map: 피험자 | 문항+단계 | 채점자', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.show()"""))

# ── Cell 13: Uncertainty ──
cells.append(mk_md(
"""## 11. 불확실성 분석: 96% 신뢰구간

각 파라미터 그룹의 96% 사후 신뢰구간을 시각화합니다."""))

cells.append(mk_code(
"""fig, axes = plt.subplots(1, 4, figsize=(16, 5))

param_sets = [
    (theta_true, theta_samples, 'theta (능력)', 'steelblue', np.arange(N)),
    (beta_true,  beta_samples,  'beta (문항)',  'coral',     np.arange(I)),
    (rho_true,   rho_samples,   'rho (채점자)', 'green',     np.arange(R)),
    (tau_true,   tau_samples,   'tau (단계)',   'purple',    np.arange(K-1)),
]

for ax, (true_v, samps, lbl, col, x_idx) in zip(axes, param_sets):
    ord_idx = np.argsort(samps.mean(axis=0))
    pm  = samps.mean(axis=0)[ord_idx]
    plo = np.percentile(samps,  2, axis=0)[ord_idx]
    phi = np.percentile(samps, 98, axis=0)[ord_idx]
    tv  = true_v[ord_idx] if len(true_v) > 1 else true_v
    x   = np.arange(len(pm))

    ax.fill_between(x, plo, phi, alpha=0.3, color=col, label='96% CI')
    ax.plot(x, pm, '-', color=col, lw=1.5, label='사후 평균')
    ax.scatter(x, tv, color='red', s=20, zorder=5, label='참값')
    ax.set_title(f'{lbl}'); ax.legend(fontsize=7); ax.grid(True, alpha=0.2)
    ax.set_xlabel('인덱스 (정렬)'); ax.set_ylabel('로짓')

plt.suptitle('96% 베이지안 신뢰구간 (MFRM)', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.show()"""))

# ── Cell 14: References ──
cells.append(mk_md(
"""## 13. References / 참고 문헌

### 국제 학술지 (확인된 논문)

1. **Uto, M., & Ueno, M. (2020)**. A generalized many-facet Rasch model and its Bayesian estimation using Hamiltonian Monte Carlo. *Behaviormetrika, 47*, 469–496. https://doi.org/10.1007/s41237-020-00115-7

2. **Uto, M. (2022)**. A Bayesian many-facet Rasch model with Markov modeling for rater severity drift. *Behavior Research Methods*. https://doi.org/10.3758/s13428-022-01997-z

3. **Bürkner, P.-C. (2021)**. Bayesian Item Response Modeling in R with brms and Stan. *Journal of Statistical Software, 100*(5), 1–54. https://doi.org/10.18637/jss.v100.i05

4. **Luo, Y., & Jiao, H. (2018)**. Using the Stan Program for Bayesian Item Response Theory. *Educational and Psychological Measurement, 78*(3), 384–408. https://doi.org/10.1177/0013164417693666

5. **Vehtari, A., Gelman, A., Simpson, D., Carpenter, B., & Bürkner, P.-C. (2021)**. Rank-Normalization, Folding, and Localization: An Improved R-hat for Assessing Convergence of MCMC. *Bayesian Analysis, 16*(2), 667–718.

---

### KCI 등재 학술지

6. **한국어 학습자의 쓰기 수행 평가 신뢰도 분석 — 다국면 라쉬 모형을 사용하여** (KCI 논문번호: ART002387352). KCI 포털(www.kci.go.kr)에서 전문 검색 가능.

7. **학문 목적 한국어 말하기 평가 과제 유형 개발 연구 — 다국면 라쉬 모형과 일반화가능도 이론 적용을 중심으로** (KCI 논문번호: ART001912809). KCI 포털에서 검색 가능.

8. **문항반응이론을 활용한 학문 목적 한국어 말하기 평가 문항 및 채점자 특성 분석** (KCI 논문번호: ART002419170). KCI 포털에서 검색 가능.

---

> **KCI 검색 방법**: www.kci.go.kr → 논문검색
> 추천 검색어: `다국면 라쉬`, `채점자 엄격성`, `MFRM`, `한국어 말하기 평가`"""))

nb = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10.0"}
    },
    "cells": cells
}

out = os.path.join(BASE, 'IRT_P3_MFRM_Model.ipynb')
with open(out, 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

with open(out) as f:
    test = json.load(f)
print(f"IRT_P3_MFRM_Model.ipynb rebuilt: {len(test['cells'])} cells")
print(f"File size: {os.path.getsize(out):,} bytes")
print("JSON valid: OK")
