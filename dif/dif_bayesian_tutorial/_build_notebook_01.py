"""Notebook 01 빌더 — 소표본·희소집단에서의 안정성 (학습 설명 보강판)."""
import json
from pathlib import Path

NB_PATH = Path(__file__).parent / "notebooks" / "01_small_sample_stability.ipynb"


def md(text): return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}
def code(text): return {"cell_type": "code", "metadata": {}, "source": text.splitlines(keepends=True),
                        "outputs": [], "execution_count": None}


cells = []

# ============================================================
# 0. 제목 + 학습 목표
# ============================================================
cells.append(md("""# Notebook 01 — 소표본·희소집단에서의 안정성

> **장점 #1**: 베이지안(Bayesian) 추론은 사전분포(prior)를 통한 정규화(regularization)로
> 표본이 작거나 집단 크기가 불균형(sparse focal group)할 때에도 안정적인 추정을 제공합니다.

### 학습 목표 (Learning Objectives)

- 표본 크기와 집단 균형이 DIF 검출에 미치는 영향을 시뮬레이션으로 확인한다.
- **빈도주의 MH**와 **베이지안 1PL DIF**의 추정 안정성을 비교한다.
- **RMSE, bias, 표집 SD**의 차이를 이해하고 **편향-분산 분해(bias-variance decomposition)**를 직접 본다.
- **명목 수준의 coverage(nominal coverage)** 개념을 이해하고 신용구간을 빈도주의적으로 점검한다.
- 점추정 SD를 관찰하는 **세 가지 이유**(MH 비교, 베이지안 절차의 빈도주의 평가, RMSE 분해)를 안다.

### 학습 전제

Notebook 00의 §1~§13을 먼저 학습하셨다고 가정합니다.
특히 §10(편향 vs 영향), §11(첫 베이지안 적합), §12(MH 비교)의 결과를 이해하고 있어야 합니다.
"""))

# ============================================================
# 1. 점추정 SD를 보는 이유 — 학습의 출발점
# ============================================================
cells.append(md("""## 1. 본 노트북에서 점추정 SD를 관찰하는 이유

베이지안 추론은 본래 **사후분포 전체**를 제공합니다.
사후 평균과 95% 신용구간이 자연스러운 보고 단위입니다.
그런데 본 노트북에서는 **시뮬레이션 반복 간 점추정치의 표준편차**(표집 SD, sampling SD)를 명시적으로 봅니다.
이유는 다음 **세 가지**입니다.

### 이유 1 — MH와의 직접 비교 (공통 척도 필요)

빈도주의 Mantel-Haenszel(MH) 절차는 본 자료의 구현에서 within-sample 신뢰구간을 산출하지 않습니다.
따라서 두 방법을 *공통 척도*로 비교하려면 양쪽 모두에 적용 가능한
**반복 시뮬레이션 기반 평가**(RMSE, 표집 SD)가 필요합니다.

### 이유 2 — 베이지안 절차의 빈도주의적 평가 (frequentist evaluation)

베이지안 절차도 **빈도주의 성질(frequentist properties)** 로 평가될 수 있고 평가되어야 합니다.

- 점추정의 평균이 진짜 값에 가까운가? → **bias**
- 자료마다 추정이 얼마나 흔들리는가? → **표집 SD**
- 95% 신용구간이 진짜를 95% 비율로 포함하는가? → **명목 수준의 coverage**

이것이 *"베이지안이지만 빈도주의적 보증도 갖는가"* 를 묻는 표준 점검이며, simulation study의 핵심 도구입니다.

### 이유 3 — RMSE의 편향-분산 분해 (Bias-Variance Decomposition)

통계학의 기본 항등식:

$$
\\mathrm{RMSE}^2 = \\mathrm{Bias}^2 + \\mathrm{Variance}_{\\text{across-rep}}
$$

여기서 분산이 곧 **표집 SD의 제곱**입니다. RMSE 한 숫자만 보면 "왜 그 값인지" 모릅니다.
**bias와 SD를 함께 봐야** 방법 차이의 원인이 드러납니다.

| 측정량 | 의미 | 본 노트북 컬럼명 |
|---|---|---|
| **Bias** | "평균적으로 진짜에서 얼마나 벗어나 있는가" (정확성) | `bias = mean_est − truth` |
| **표집 SD** | "추정이 자료마다 얼마나 흔들리는가" (안정성) | `sd` |
| **RMSE** | 둘을 합친 종합 오차 | `rmse` |

베이지안의 prior shrinkage는 **약간의 bias를 도입하는 대신 SD를 크게 줄여** 전체 RMSE를 낮춥니다.
이 메커니즘을 시각화하는 것이 본 노트북의 핵심입니다.

### 두 종류의 SD 구분 (중요)

베이지안 분석에는 **서로 다른 두 가지 SD**가 있습니다.

| 명칭 | 정의 | 측정 대상 |
|---|---|---|
| **사후 SD (posterior SD)** | 한 번의 적합 안에서 사후분포의 표준편차 | 이 자료를 본 후의 모수 불확실성 |
| **표집 SD (sampling SD)** | N_REPS회 반복으로 얻은 점추정치들의 표준편차 | 절차의 빈도주의적 안정성 |

본 노트북의 `sd` 컬럼은 **표집 SD**입니다.
**잘 보정된(well-calibrated)** 베이지안 절차에서는 두 SD가 거의 일치합니다 (캘리브레이션).
이 점검은 §11의 명목 coverage 분석과 본질적으로 같은 작업입니다.
"""))

# ============================================================
# 2. 시뮬레이션 시나리오
# ============================================================
cells.append(md("""## 2. 시뮬레이션 시나리오 설정

다음 4개 시나리오를 반복 시뮬레이션(Monte Carlo replication)하여 두 방법의 추정 안정성을 비교합니다.

| 시나리오 | n_ref | n_focal | 의도 |
|---|---|---|---|
| **A — 균형 충분** | 300 | 300 | 표준, 두 방법이 유사할 것으로 예상 |
| **B — 균형 소표본** | 80  | 80  | 표본 적음, 베이지안 우위 시작 |
| **C — 희소 focal** | 300 | 50  | 집단 불균형, 베이지안 우위 확대 |
| **D — 극단 희소** | 400 | 25  | 매우 극단적, MH 불안정성 극대화 |

자료생성과정(data-generating process, DGP)은 Notebook 00과 동일:
- 10문항, 문항 5에 $\\Delta b = +0.8$ (강한 DIF), 문항 8에 $\\Delta b = -0.4$ (약한 DIF).
- 두 집단의 능력 평균 동일 (impact 없음).

각 시나리오마다 **N_REPS = 30** 회 반복합니다 (학습 목적상 빠른 실행; 실제 연구는 100~1000회 권장).

**예상되는 결과**:
- A에서는 두 방법이 비슷한 RMSE.
- B → C → D로 갈수록 **MH의 RMSE가 베이지안보다 빠르게 악화**.
- 베이지안은 prior 정규화로 **분산이 작아져** RMSE가 잘 보존됨.
- 단, 베이지안에는 약간의 bias(0 쪽 shrinkage)가 도입됨 — 편향-분산 trade-off.
"""))

# ============================================================
# 3. 확률모델 명시 (Explicit Probabilistic Model)
# ============================================================
cells.append(md("""## 3. 베이즈 확률모델 — 명시적 기술 (Explicit Probabilistic Model)

본 노트북에서 사용하는 베이지안 적합 함수 `models.fit_rasch_dif()` 는
다음 **Rasch 1PL + uniform DIF** 확률모델을 그대로 구현합니다.
앞 절들의 정성적 서술은 모두 이 모형과 그 사전분포(prior distribution)에서 도출되는 결과입니다 — 여기서는 **수식 단위로** 모형을 명시합니다.
이 절을 통해 *"베이즈가 무엇을 입력으로 받고 무엇을 출력하는지"* 가 명확해야 하며,
이후 §6~§12의 모든 결과는 이 모형의 사후분포를 시각화한 것입니다.

### 3.1. 우도 (Likelihood)

응답자 $i$ ($i = 1, \\ldots, N$) 가 문항 $j$ ($j = 1, \\ldots, J = 10$) 에 답한 이진 응답
$Y_{ij} \\in \\{0, 1\\}$ 은 다음 Bernoulli 분포를 따른다고 가정합니다.

$$
Y_{ij} \\mid \\theta_i,\\, b_j,\\, \\delta_j,\\, \\mu_F,\\, g_i \\;\\sim\\; \\mathrm{Bernoulli}(p_{ij})
$$

$$
\\mathrm{logit}(p_{ij}) \\;=\\; \\underbrace{(\\theta_i + g_i \\cdot \\mu_F)}_{\\text{응답자측 (능력 + 집단 시프트)}}
                       \\;-\\; \\underbrace{(b_j + g_i \\cdot \\delta_j)}_{\\text{문항측 (기본 난이도 + DIF 시프트)}}
$$

여기서 $g_i \\in \\{0, 1\\}$ 은 집단 지시변수 (0 = reference, 1 = focal).
관측을 *응답 단위* $k = 1, \\ldots, K$ 로 펴면 (long format):

$$
\\eta_k = (\\theta_{ii[k]} + g_k\\,\\mu_F) - (b_{jj[k]} + g_k\\,\\delta_{jj[k]}), \\qquad
Y_k \\sim \\mathrm{Bernoulli}(\\sigma(\\eta_k))
$$

이 식이 `rasch_dif.stan` 의 `y ~ bernoulli_logit(eta)` 한 줄에 해당합니다 ($\\sigma$ = 로지스틱).

### 3.2. 파라메터 (Parameters) — 추정 대상

| 기호 | 차원 | 의미 |
|---|---|---|
| $\\theta_i$ | $N$ (응답자 수) | 응답자 $i$ 의 잠재 능력 (latent ability) |
| $b_j$ | $J = 10$ | 문항 $j$ 의 기본 난이도 (reference 집단 기준) |
| $\\delta_j$ | $J = 10$ | 문항 $j$ 의 **DIF 효과 크기** — focal 집단에서의 난이도 변화 |
| $\\mu_F$ | 1 | focal 집단의 평균 능력 시프트 (group impact 항) |

본 노트북의 **주된 추정 대상**은 $\\delta_j$ 입니다. 나머지는 보조 모수(nuisance parameters)로,
사후 적분으로 자연스럽게 주변화(marginalize)됩니다.

### 3.3. 사전분포 (Prior Distributions) — 형태와 의미

`rasch_dif.stan` 의 `model { ... }` 블록은 다음 네 개의 사전을 부과합니다.

| 모수 | 사전분포 | 의미 / 정당화 |
|---|---|---|
| $\\theta_i$ | $\\mathrm{N}(0,\\, 1^2)$ | 능력 분포의 척도를 1 SD로 고정 → **척도 식별성(scale identifiability)** 확보. 표준 IRT 관습. 이 고정이 없으면 $(\\theta_i, b_j)$ 가 함께 임의 상수만큼 이동해도 우도가 불변(비식별). |
| $b_j$ | $\\mathrm{N}(0,\\, 2^2)$ | 문항 난이도가 일반적으로 $\\theta$ SD의 2~3배 이내라는 **약하게 정보적(weakly informative)** 사전. $\\|b\\| > 6$ 같은 극단값에 사후질량 거의 0. |
| $\\delta_j$ | $\\mathrm{N}(0,\\, \\sigma_\\delta^2),\\ \\sigma_\\delta = 1.0$ | **"DIF는 보통 작거나 0"** 이라는 약한 사전 신념. logit 스케일에서 $\\sigma_\\delta = 1.0$ 은 매우 약한 정규화 — $\\|\\delta\\| > 2$ 도 허용하지만 0 근처에 더 무게를 둠. 본 절의 안정성 효과는 이 사전에서 직접 기인. |
| $\\mu_F$ | $\\mathrm{N}(0,\\, 1^2)$ | 두 집단의 평균 능력 차이도 1 SD 이내가 자연스럽다는 약한 사전 (impact 추정 정규화). |

> **"약하게 정보적인(weakly informative)"** 의 정확한 의미:
> 사전이 "no DIF"라는 결론을 *강요*하지 않습니다.
> 자료가 강한 DIF 신호를 보이면 사후는 자료를 따르고, 자료가 빈약하면 사전이 $\\delta_j \\approx 0$ 쪽으로 *약하게* 끌어당깁니다.
> 이것이 곧 **정규화(regularization)** 입니다 — Tikhonov(ridge) 회귀의 베이지안 대응물.

> **참고**: 본 노트북에서는 $\\sigma_\\delta = 1.0$ 으로 고정합니다 (`prior_sigma_delta=1.0`).
> Notebook 03 의 위계 사전(hierarchical prior)은 $\\sigma_\\delta$ 자체를 자료로 추정하여
> 풀링 강도(shrinkage strength)를 데이터 주도로 결정합니다.

### 3.4. 사후분포 (Posterior)

베이즈 정리에 의해 사후는 우도와 사전의 곱에 비례합니다:

$$
p(\\theta, b, \\delta, \\mu_F \\mid Y) \\;\\propto\\;
\\underbrace{\\prod_{k=1}^{K} \\mathrm{Bernoulli}\\!\\left(y_k \\mid \\sigma(\\eta_k)\\right)}_{\\text{우도}}
\\;\\times\\;
\\underbrace{\\prod_i \\phi(\\theta_i; 0, 1) \\;\\prod_j \\phi(b_j; 0, 2)
\\;\\prod_j \\phi(\\delta_j; 0, \\sigma_\\delta)\\; \\phi(\\mu_F; 0, 1)}_{\\text{사전의 곱}}
$$

$\\phi(\\cdot; 0, \\sigma)$ 는 평균 0, 표준편차 $\\sigma$ 인 정규분포 밀도.
**NUTS(No-U-Turn Sampler) / HMC(Hamiltonian Monte Carlo)** 가 이 사후로부터 표본을 추출하며,
본 노트북에서 `samples = fit["samples"]["delta"]` 로 받는 행렬이 곧 $\\delta_j$ 의 사후표본입니다.

### 3.5. 왜 이 모형이 소표본·희소집단에서 안정적인가 — 다섯 가지 근거

**근거 1. 사전이 추정량을 유한 영역으로 제약 (proper posterior 보장)**

- MH 의 $\\log(\\alpha_{MH})$ 는 어느 분위(stratum)에 0 또는 무한대 셀이 있으면 *발산하거나 정의되지 않음* (희소표본의 0-cell 문제).
- $\\delta_j \\sim \\mathrm{N}(0, 1)$ 사전은 사후가 *항상* 유한한 평균·분산을 갖도록 보장 (proper posterior).
  자료에 정보가 거의 0 이어도 사후는 사전과 유사할 뿐 — **추정 실패가 발생하지 않습니다**.

**근거 2. 자료 정밀도와 사전 정밀도의 자동 균형 (precision pooling)**

정규-정규 결합에서 사후 정밀도 = 우도 정밀도 + 사전 정밀도:

$$
\\frac{1}{\\sigma_{\\text{post}}^2} \\;=\\; \\frac{1}{\\sigma_{\\text{data}}^2} + \\frac{1}{\\sigma_{\\text{prior}}^2}
$$

- 표본이 적으면 $\\sigma_{\\text{data}}^2$ 이 커서(= 우도 정밀도가 작아서) **사전이 자동으로 더 큰 비중** → $\\delta_j$ 가 0 쪽으로 끌림 (= 약간의 음의 bias).
- 표본이 충분하면 우도가 압도 → 자료가 결론을 결정 (bias 소멸).
- 이 비율 조정은 **추가 설정 없이** 베이즈 공식이 자동 수행합니다.

**근거 3. 부분 풀링(partial pooling)에 의한 분산 감소**

- $\\delta_j \\sim \\mathrm{N}(0, \\sigma_\\delta)$ 가 모든 문항에 *공통 사전 평균 0* 을 부과 → 강한 자료가 없는 문항은 자연히 0 쪽으로 끌림.
- 결과: **표집 SD 감소**, 약간의 음의 bias 도입 — 편향-분산 trade-off (§10 시각화에서 직접 확인).
- (Notebook 03 의 **위계 사전**은 $\\sigma_\\delta$ 자체를 자료로 추정하여 풀링 강도를 데이터 주도로 결정.)

**근거 4. 분위(stratum) 셀이 비어도 정보 손실 없음**

- MH 는 분위별 $2 \\times 2$ 표 합산 → 분위 셀이 비면 그 분위의 기여가 통째로 사라짐.
- 베이지안은 모든 응답을 **하나의 joint likelihood** 로 묶음. 응답자 능력 $\\theta_i$ 가 잠재 모수로 *연속*적으로 추정되므로 분위 분할 자체가 불필요.
- 결과적으로 자료의 **모든 비트(every bit)** 가 모수 추정에 기여 — 작은 자료에서 효율 격차가 더 큼.

**근거 5. 불확실성을 사후분포로 정직하게 표현**

- 자료가 부족하면 사후가 *넓어질 뿐*, 추정이 "실패"하지 않음.
- 95% 신용구간이 자동으로 확장 → "모른다"를 정량화.
- 잘 보정된 절차에서는 이 구간이 **명목 수준의 coverage** 까지 달성 (§11, §12에서 시뮬레이션으로 검증).

**핵심 요약**

> 사전분포 $\\delta_j \\sim \\mathrm{N}(0, 1)$ 은 *"DIF는 작거나 없다"는 약한 가정*을 도입함으로써
> 자료가 빈약한 영역에서 추정량이 발산하지 않게 막고, 자료가 풍부해지면 자료가 결론을 결정하도록 길을 비킵니다.
> 이것이 베이즈 추론이 **소표본·희소집단에서 안정적**일 수 있는 근본 메커니즘이며,
> 본 노트북의 시뮬레이션(§7~§10)이 이 메커니즘을 RMSE·bias·SD 수치로 가시화합니다.
"""))

# ============================================================
# 4. 백엔드 + import
# ============================================================
cells.append(md("""## 4. 백엔드 선택 및 모듈 import

Notebook 00과 동일한 방식. Windows에서 UTF-8 오류가 발생하면 Notebook 00 §5의 안내를 따르세요.
"""))

cells.append(code("""# 백엔드 선택
BACKEND = "stan"
import platform, importlib.util, warnings
def _resolve(req):
    req = req.lower()
    if req == "numpyro":
        ok = (importlib.util.find_spec("jax") is not None
              and importlib.util.find_spec("numpyro") is not None)
        if not ok:
            warnings.warn("numpyro unavailable -> Stan fallback")
            return "stan"
    return "stan" if req != "numpyro" else "numpyro"
BACKEND = _resolve(BACKEND)
print(f"Active backend: {BACKEND}")
"""))

cells.append(code("""import sys
from pathlib import Path
PROJECT_ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from difbayes import simulate, visualize, frequentist, diagnostics, models
plt.rcParams.update({"figure.dpi": 110, "axes.spines.top": False, "axes.spines.right": False, "font.size": 10})
print("Modules loaded.")
"""))

# ============================================================
# 4. MH 출력의 척도 — log-odds 정확한 용어
# ============================================================
cells.append(md("""## 5. 빈도주의 MH 출력의 척도 — log-odds (로그-오즈) 정리

본격 시뮬레이션 전에 **두 방법의 추정량을 같은 척도로 표현**하는 변환을 정리합니다.
이 절은 **표준 용어**를 정확히 사용하여 혼란을 없애는 것이 목적입니다.

### 5.1. 정확한 용어

| 영어 표준 | 한국어 표준 | 정의 |
|---|---|---|
| **logit** = **log-odds** | **로짓** = **로그-오즈** | $\\log[p/(1-p)]$ — 단일 확률의 로그-오즈 변환 |
| **log odds ratio (log OR)** | **로그-오즈비** | $\\log(\\alpha) = \\log(\\text{odds}_1 / \\text{odds}_2)$ — 두 오즈의 비에 로그 |
| **log-odds scale** | **로그-오즈 스케일** | 위 두 양이 공통으로 존재하는 좌표축 |

**"logit-OR"는 비표준 축약**입니다. 정확히는 **로그-오즈비(log odds ratio)** 이며,
그 값이 존재하는 좌표축은 **로짓 스케일 = 로그-오즈 스케일**입니다.

### 5.2. 두 방법의 자연 출력 비교

| 방법 | 자연 출력 | 단위 |
|---|---|---|
| **MH** | 공통 승산비 $\\alpha_{MH}$ 와 그 로그 $\\log(\\alpha_{MH})$ | **로그-오즈 스케일** |
| **Bayes** | 사후 평균 $\\hat{\\Delta b}$ | **로짓 = 로그-오즈 스케일** |

두 값은 *같은 단위(로그-오즈)*에 살고 있어 직접 비교 가능합니다.

### 5.3. Rasch 1PL과의 직접 연결

Rasch 1PL의 정의에서, 같은 능력 $\\theta$ 의 응답자가 두 집단에 있을 때:

$$
\\log\\frac{\\mathrm{odds}_{ref}}{\\mathrm{odds}_{focal}} = (\\theta - b_{ref}) - (\\theta - b_{focal}) = b_{focal} - b_{ref} = \\Delta b
$$

즉:

$$
\\boxed{\\;\\log(\\alpha_{MH}) \\approx \\Delta b\\;}
$$

로그-오즈비는 두 로짓 값의 *차이*이므로 단위가 logit이 됩니다.
따라서 MH 출력을 $\\log(\\alpha_{MH})$ 로 두면 베이지안 사후 평균 $\\Delta b$ 와 **직접 비교 가능**합니다.

### 5.4. ETS의 Delta_MH 와의 관계

`frequentist.py` 의 출력 중 `delta_mh` 는 다음 정의를 따릅니다 (ETS 관습).

$$
\\Delta_{MH} = -2.35 \\times \\log(\\alpha_{MH})
$$

`-2.35` 는 logit을 probit-like delta scale로 환산하기 위한 **근사 상수**일 뿐 정확한 등식이 아닙니다.
따라서 본 노트북에서는 ETS scale로 환산하지 않고 **로그-오즈 스케일을 유지**해 비교합니다.

다음 코드의 변환 한 줄이 이 결정을 구현합니다:

```python
log_odds_ratio = -m.delta_mh / 2.35    # equivalent to log(alpha_mh)
```
"""))

cells.append(md("""## 6. 시뮬레이션 모수 설정

진짜 모수와 반복 횟수를 정의합니다.

> 시간 안내: 4개 시나리오 x 30회 x 2개 방법 = 240회 적합.
> Stan 컴파일은 첫 1회만, 이후 캐시 사용. 총 5~15분 소요 예상.
> 시간이 부족하면 N_REPS = 10 으로 줄여 실행해보세요.
"""))

cells.append(code("""N_REPS = 30   # 학습용 빠른 실행. 정밀 비교는 100~1000 권장.

SCENARIOS = [
    dict(name="A: balanced (300/300)", n_ref=300, n_focal=300),
    dict(name="B: small balanced (80/80)", n_ref=80,  n_focal=80),
    dict(name="C: sparse focal (300/50)", n_ref=300, n_focal=50),
    dict(name="D: extreme sparse (400/25)", n_ref=400, n_focal=25),
]

b_true = np.linspace(-2.0, 2.0, 10)
delta_b_true = np.zeros(10)
delta_b_true[4] = 0.8
delta_b_true[7] = -0.4
TRUE_J = len(b_true)
print(f"N_REPS = {N_REPS}, J = {TRUE_J}")
print(f"True Delta b: {delta_b_true.round(2)}")
"""))

cells.append(md("""## 7. 반복 시뮬레이션 실행

각 시나리오 x 반복마다:
1. 새 자료 생성 (seed 변경)
2. MH 적용 -> 점추정 (로그-오즈 스케일)
3. 베이지안 적합 -> 사후 평균 + 95% 신용구간

각 결과를 `results` 리스트에 dict 형태로 누적합니다.

> 주의 — MH 결과의 lo, hi 는 NaN: `frequentist.py` 가 현재 표준오차(Robins-Breslow-Greenland)를
> 산출하지 않기 때문입니다. 본 노트북의 핵심 비교(RMSE, 표집 SD)는 점추정만으로 가능하므로 영향 없습니다.

### Quiet mode 안내

`fit_rasch_dif()` 는 기본값 `verbose=False` 로 호출됩니다. 이 모드에서는
**cmdstanpy 의 INFO 로그**, **subprocess stdout/stderr**, **컴파일 메시지**가 모두 흡수되어
노트북이 깔끔하게 유지됩니다 (WARNING/ERROR 메시지는 여전히 표시).
첫 컴파일만 별도로 `verbose=True` 로 1회 호출해 진행 상황을 확인합니다.
"""))

cells.append(code("""# 첫 컴파일 — verbose=True 로 1회만 실행해 진행 상황 확인
print("Pre-compiling Stan model (verbose, ~1 minute on first run)...")
_pre = models.fit_rasch_dif(
    Y=simulate.simulate_rasch_dif(
        n_ref=20, n_focal=20, b_true=b_true, delta_b_true=delta_b_true, seed=0,
    ).Y,
    group=simulate.simulate_rasch_dif(
        n_ref=20, n_focal=20, b_true=b_true, delta_b_true=delta_b_true, seed=0,
    ).group,
    backend=BACKEND,
    n_chains=1, n_warmup=50, n_samples=50,
    prior_sigma_delta=1.0, seed=0, verbose=True,
)
del _pre
print("Pre-compile done. Subsequent fits will be silent.\\n")
"""))

cells.append(code("""results = []

for sc in SCENARIOS:
    print(f"=== Scenario {sc['name']} ===")
    for r in range(N_REPS):
        seed = 1000 * (1 + SCENARIOS.index(sc)) + r
        data = simulate.simulate_rasch_dif(
            n_ref=sc["n_ref"], n_focal=sc["n_focal"],
            b_true=b_true, delta_b_true=delta_b_true, seed=seed,
        )

        # MH on log-odds scale (matches Bayesian Delta b unit)
        # log(alpha_MH) = -delta_MH / 2.35
        mh = frequentist.mantel_haenszel_all(data.Y, data.group, n_strata=4)
        for j, m in enumerate(mh):
            log_odds_ratio = (-m.delta_mh / 2.35) if np.isfinite(m.delta_mh) else np.nan
            results.append(dict(
                scenario=sc["name"], rep=r, method="MH (log-odds)",
                item=j+1, est=log_odds_ratio, lo=np.nan, hi=np.nan,
                truth=delta_b_true[j],
            ))

        # Bayesian non-hierarchical — verbose=False (default) → quiet
        fit = models.fit_rasch_dif(
            Y=data.Y, group=data.group, backend=BACKEND,
            n_chains=2, n_warmup=300, n_samples=500,
            prior_sigma_delta=1.0, seed=seed,
            verbose=False,
        )
        samples = fit["samples"]["delta"].reshape(-1, TRUE_J)
        for j in range(TRUE_J):
            s = samples[:, j]
            results.append(dict(
                scenario=sc["name"], rep=r, method="Bayes (weak prior)",
                item=j+1, est=s.mean(),
                lo=np.quantile(s, 0.025), hi=np.quantile(s, 0.975),
                truth=delta_b_true[j],
            ))
        if (r + 1) % 10 == 0:
            print(f"  rep {r+1}/{N_REPS}")

results_df = pd.DataFrame(results)
print(f"\\nTotal rows: {len(results_df)}")
results_df.head()
"""))

cells.append(md("""## 8. 결과 집계 — RMSE, bias, 표집 SD

각 (시나리오, 방법, 문항) 그룹에 대해 N_REPS회의 점추정치들을 모아 다음을 계산합니다.

| 컬럼 | 정의 | 의미 |
|---|---|---|
| `mean_est` | 점추정치들의 평균 | 추정의 *중심* |
| `truth` | 진짜 모수 | 비교 기준 |
| `bias = mean_est - truth` | 평균 오차 | 정확성 |
| `sd` | 점추정치들의 표준편차 | **표집 SD**, 안정성 |
| `rmse` | sqrt(평균 제곱 오차) | 종합 오차 |

이론적으로 RMSE^2 = Bias^2 + SD^2 가 성립합니다 (편향-분산 분해).
표 마지막에 bias2_plus_sd2 = sqrt(bias^2 + sd^2) 를 추가하여 직접 점검합니다.
"""))

cells.append(code("""agg = (results_df
       .assign(error=lambda d: d["est"] - d["truth"])
       .groupby(["scenario", "method", "item"])
       .agg(rmse=("error", lambda x: np.sqrt(np.nanmean(x**2))),
            sd=("est", "std"),
            mean_est=("est", "mean"),
            truth=("truth", "first"))
       .reset_index())
agg["bias"] = agg["mean_est"] - agg["truth"]
agg["bias2_plus_sd2"] = np.sqrt(agg["bias"]**2 + agg["sd"]**2)
print("First 8 rows:")
print(agg.head(8).round(3).to_string(index=False))
print()
print("Decomposition check: max |rmse - sqrt(bias^2 + sd^2)| =",
      np.abs(agg['rmse'] - agg['bias2_plus_sd2']).max().round(4))
"""))

cells.append(md("""**점검**: `bias2_plus_sd2` 와 `rmse` 가 거의 같아야 합니다 (편향-분산 분해의 수치적 확인).
미세 차이는 N_REPS=30 의 Monte Carlo 변동이 원인입니다.
"""))

cells.append(md("""## 9. 시각화 1 — RMSE 시나리오별 비교

진짜 DIF 문항(5번, 8번)에 대한 RMSE 변화를 시각화합니다.
시나리오 A -> D 로 갈수록 두 방법의 RMSE 격차가 어떻게 변하는지가 핵심입니다.
"""))

cells.append(code("""fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for k, item in enumerate([5, 8]):
    sub = agg[agg["item"] == item]
    scenarios = [s["name"] for s in SCENARIOS]
    x = np.arange(len(scenarios))
    width = 0.35
    mh   = sub[sub["method"] == "MH (log-odds)"].set_index("scenario").loc[scenarios, "rmse"]
    bay  = sub[sub["method"] == "Bayes (weak prior)"].set_index("scenario").loc[scenarios, "rmse"]
    axes[k].bar(x - width/2, mh, width, label="MH (log-odds)", color="#1f77b4")
    axes[k].bar(x + width/2, bay, width, label="Bayes (weak prior)", color="#2ca02c")
    axes[k].set_xticks(x)
    axes[k].set_xticklabels([s.split(":")[0] for s in scenarios])
    axes[k].set_title(f"Item {item}  (true Delta b = {delta_b_true[item-1]:+.2f})")
    axes[k].set_ylabel("RMSE")
    axes[k].legend(fontsize=9)
    axes[k].grid(axis="y", alpha=0.3)
fig.suptitle("RMSE of DIF estimates across scenarios", y=1.02, fontsize=12)
fig.tight_layout()
fig.savefig("../outputs/01_rmse_comparison.png", dpi=120, bbox_inches="tight")
plt.show()
"""))

cells.append(md("""**해석 (1) — RMSE**

- 시나리오 A에서 두 방법은 비슷한 RMSE.
- B, C, D로 갈수록 **MH의 RMSE가 베이지안보다 빠르게 악화**.
- 왜? 다음 시각화(편향-분산 분해)에서 원인을 분석합니다.
"""))

cells.append(md("""## 10. 시각화 2 — 편향-분산 분해 (Bias-Variance Decomposition)

RMSE는 한 숫자라 **무엇 때문에** 커지는지(bias인지, SD인지) 알려주지 못합니다.
문항 5에 대해 bias와 SD를 시나리오별로 나란히 보면 메커니즘이 드러납니다.
"""))

cells.append(code("""fig, axes = plt.subplots(1, 2, figsize=(13, 4))

sub = agg[agg["item"] == 5]
scenarios = [s["name"] for s in SCENARIOS]
x = np.arange(len(scenarios))
width = 0.35

mh_b  = sub[sub["method"] == "MH (log-odds)"].set_index("scenario").loc[scenarios, "bias"]
bay_b = sub[sub["method"] == "Bayes (weak prior)"].set_index("scenario").loc[scenarios, "bias"]
axes[0].bar(x - width/2, mh_b, width, label="MH", color="#1f77b4")
axes[0].bar(x + width/2, bay_b, width, label="Bayes", color="#2ca02c")
axes[0].axhline(0, color="gray", lw=0.5)
axes[0].set_xticks(x)
axes[0].set_xticklabels([s.split(":")[0] for s in scenarios])
axes[0].set_ylabel("Bias (mean_est - truth)")
axes[0].set_title("Bias of Delta b estimates (Item 5)")
axes[0].legend(fontsize=9)
axes[0].grid(axis="y", alpha=0.3)

mh_s  = sub[sub["method"] == "MH (log-odds)"].set_index("scenario").loc[scenarios, "sd"]
bay_s = sub[sub["method"] == "Bayes (weak prior)"].set_index("scenario").loc[scenarios, "sd"]
axes[1].bar(x - width/2, mh_s, width, label="MH", color="#1f77b4")
axes[1].bar(x + width/2, bay_s, width, label="Bayes", color="#2ca02c")
axes[1].set_xticks(x)
axes[1].set_xticklabels([s.split(":")[0] for s in scenarios])
axes[1].set_ylabel("Sampling SD (across reps)")
axes[1].set_title("Sampling SD of Delta b estimates (Item 5)")
axes[1].legend(fontsize=9)
axes[1].grid(axis="y", alpha=0.3)

fig.tight_layout()
fig.savefig("../outputs/01_bias_variance.png", dpi=120, bbox_inches="tight")
plt.show()
"""))

cells.append(md("""**해석 (2) — 편향-분산 트레이드오프**

- **Bias 그림**: 베이지안은 prior가 진짜 +0.8을 약간 0 쪽으로 끌어당겨 **음의 bias**가 보임.
  특히 표본이 적을수록 prior 영향이 커서 bias 절댓값이 커짐.
- **SD 그림**: MH의 SD가 시나리오 D로 갈수록 폭증. 베이지안은 prior 정규화로 SD가 잘 억제됨.
- **종합**: 베이지안은 **약간의 bias를 도입**(나쁨)하지만 **SD를 크게 줄여**(좋음) 전체 RMSE를 낮춤.
  이것이 **편향-분산 트레이드오프(bias-variance tradeoff)**의 교과서적 예시입니다.

> Tip: "베이지안이 무조건 우월하다"가 아니라 **"베이지안은 약간의 체계적 편향을 감수하고
> 분산을 크게 줄이는 다른 trade-off를 채택한다"** 가 정확한 진술입니다.
"""))

cells.append(md("""## 11. 신용구간 Coverage — \"명목 수준(nominal level)\"의 의미

신용구간을 시뮬레이션으로 점검하기 전에, **명목 수준의 coverage** 개념을 정확히 정리합니다.

### 11.1. 정의

- **명목 수준 (nominal level)**: 우리가 *선언한* 신뢰도. "95% 신용구간"이라 부르면 명목 수준은 0.95.
  - "명목(nominal)"은 라틴어 nomen(이름)에서 왔으며, **"이름붙인 값"**, **"선언된 값"**을 뜻합니다.
- **실제 coverage (actual coverage)**: 반복 시뮬레이션에서 신용구간이 진짜 값을 *실제로* 포함한 비율.
- **명목 수준의 coverage**: 선언한 수준과 실제 coverage가 **일치**하는 상태.
  - 예: 0.95 선언 -> 실제 0.95 포함.

### 11.2. 세 가지 가능한 상태

| 실제 coverage | 명칭 | 진단 |
|---|---|---|
| ~ 0.95 | **Nominal (보정됨)** | 사후 SD ~ 표집 SD, 캘리브레이션 양호 |
| > 0.95 | **Conservative (보수적)** | 사후 SD > 표집 SD, 구간이 너무 넓음 (underconfident) |
| < 0.95 | **Anti-conservative (반보수적)** | 사후 SD < 표집 SD, 구간이 너무 좁음 (overconfident) |

### 11.3. 왜 점검하는가

베이지안 신용구간은 본래 *"이 자료를 봤을 때 모수가 이 구간에 있을 사후확률이 0.95"* 라는 **사후확률 진술**입니다.
빈도주의적 "장기 빈도 보장"은 자동으로 따라오지 않습니다 — prior와 모형이 적절할 때만 성립합니다.
따라서 시뮬레이션으로 **명목 coverage가 달성되는지 점검**하면 베이지안 절차의 캘리브레이션을 검증할 수 있습니다.

명목 coverage를 달성하면, 베이지안 신용구간은 **두 가지 보증**을 동시에 갖습니다:
- (a) "이 자료를 본 후 모수가 구간에 있을 사후확률 0.95"
- (b) "이 절차를 반복 적용하면 0.95 비율로 진짜를 포함"

### 11.4. 캘리브레이션과의 연결

§1에서 본 사후 SD와 표집 SD의 관계가 다시 등장합니다:

> 사후 SD ~ 표집 SD  ==>  실제 coverage ~ 명목 수준

두 SD의 일치를 점검하는 것이 곧 명목 coverage를 점검하는 것과 **본질적으로 같은 작업**입니다.
이는 베이지안의 **빈도주의적 평가(이유 2)** 의 실질적 도구입니다.

### 11.5. 어원적 직관

"명목(nominal)"은 통계학에서 **"이름붙인 값 vs 실제로 측정된 값"**의 대비를 표시할 때 씁니다.

| 명목 값 | 실제 값 | 일치하는가? |
|---|---|---|
| 명목 직경 1인치 나사 | 실측 0.997인치 | 거의 일치 (보정됨) |
| 명목 5% 유의수준 검정 | 실제 type-I error rate | 모형이 옳다면 일치 |
| **명목 95% 신용구간** | **실제 coverage** | **시뮬레이션으로 점검 가능** |
"""))

cells.append(md("""## 12. 시뮬레이션 기반 Coverage 점검

각 (시나리오, 문항) 그룹에서 베이지안 95% 신용구간이 진짜 값을 포함한 비율을 계산합니다.
이상적으로 0.95에 가까워야 합니다 (명목 수준의 coverage).
"""))

cells.append(code("""bay = results_df[results_df["method"] == "Bayes (weak prior)"].copy()
bay["covered"] = (bay["lo"] <= bay["truth"]) & (bay["truth"] <= bay["hi"])
coverage = bay.groupby(["scenario", "item"])["covered"].mean().reset_index()
print("Bayesian 95% credible interval coverage:")
cov_table = coverage.pivot(index="item", columns="scenario", values="covered").round(2)
print(cov_table)
"""))

cells.append(code("""# Scenario-level mean coverage with nominal line
fig, ax = plt.subplots(figsize=(8, 4))
scenarios = [s["name"] for s in SCENARIOS]
mean_cov = [coverage[coverage["scenario"]==s]["covered"].mean() for s in scenarios]
bars = ax.bar([s.split(":")[0] for s in scenarios], mean_cov, color="#2ca02c", alpha=0.7)
ax.axhline(0.95, color="red", ls="--", label="Nominal level (0.95)")
ax.set_ylim(0, 1.05)
ax.set_ylabel("Mean coverage (across items)")
ax.set_title("Bayesian 95% CI Coverage vs Nominal Level")
ax.legend()
ax.grid(axis="y", alpha=0.3)
for bar, val in zip(bars, mean_cov):
    ax.text(bar.get_x() + bar.get_width()/2, val + 0.02, f"{val:.2f}",
            ha="center", fontsize=10)
fig.savefig("../outputs/01_coverage.png", dpi=120, bbox_inches="tight")
plt.show()
"""))

cells.append(md("""**해석 (3) — Coverage 점검**

- 대부분의 시나리오에서 coverage가 **0.95 근처에 머무는 것**은 베이지안 절차가
  **명목 수준의 coverage**를 달성한다는 증거입니다.
- 시나리오 D(극단 희소)에서는 표본이 너무 작아 N_REPS=30 으로는 Monte Carlo 변동이 커서
  coverage가 다소 흔들릴 수 있습니다.
- coverage 평균이 0.95에서 멀리 떨어지면 **prior 민감도 분석** 또는 **위계 모형**(Notebook 03)을 고려해야 합니다.

> 베이지안의 신용구간이 빈도주의 신뢰구간으로서도 명목 수준을 달성한다는 것은
> *"이 베이지안 절차는 두 가지 보증을 동시에 갖는다"* 는 뜻입니다.
> 즉 (a) 사후확률 0.95 진술, 그리고 (b) 반복 적용 시 0.95 비율로 진짜 포함.
> 잘 보정된 베이지안 절차에서 이 둘이 일치합니다.
"""))

cells.append(md("""## 13. 요약 (Summary)

### 13.1. 개념적 정리

1. **점추정 SD를 보는 세 가지 이유**:
   - (a) MH와의 공통 척도 비교
   - (b) 베이지안 절차의 **빈도주의적 평가**
   - (c) RMSE의 **편향-분산 분해**

2. **두 종류의 SD 구분**:
   - **사후 SD** (within-sample): 한 번 적합 안의 사후분포 폭.
   - **표집 SD** (across-replication): 자료를 새로 뽑을 때마다 점추정 변동.

3. **로그-오즈 척도의 표준 용어**:
   - **로짓 (logit) = 로그-오즈 (log-odds)** — 단일 확률의 log-odds 변환.
   - **로그-오즈비 (log odds ratio)** — 두 오즈의 비에 로그.
   - **로그-오즈 스케일** — 위 둘이 공통으로 사는 좌표축.
   - "logit-OR"는 비표준 축약 — 정확히 표기하려면 "log odds ratio" 사용.

4. **명목 수준의 coverage (nominal coverage)**:
   - "선언한 신뢰도와 실제 포함률의 일치".
   - 잘 보정된 베이지안 절차의 핵심 지표.
   - 사후 SD ~ 표집 SD 이면 자동으로 달성.

5. **베이즈 확률모델의 명시적 구조 (§3)**:
   - 우도: Bernoulli-logit Rasch 1PL + uniform DIF.
   - 사전: $\\theta \\sim N(0,1)$, $b \\sim N(0,2)$, $\\delta \\sim N(0,1)$, $\\mu_F \\sim N(0,1)$.
   - 사전이 추정량을 유한 영역으로 제약 → proper posterior 보장.
   - 자료 정밀도와 사전 정밀도의 자동 균형 → 소표본에서 자동 정규화.

### 13.2. 시뮬레이션 관찰

6. **편향-분산 트레이드오프**:
   - 베이지안은 **약간의 bias**를 도입하고 **SD를 크게 줄여** RMSE 우위 달성.
   - 소표본·희소집단으로 갈수록 효과 극대화.

### 13.3. 핵심 메시지

> 베이지안의 우위는 *마법이 아니라* **편향-분산 trade-off의 다른 선택**입니다.
> Prior가 약간의 bias를 도입하지만 분산을 크게 줄여, **소표본·희소집단 상황에서 RMSE 측면에서 더 안정**합니다.
> 동시에 신용구간이 **명목 수준의 coverage**를 달성하면 빈도주의적 보증도 함께 갖춥니다.

### 13.4. 다음 노트북
- **Notebook 02** — 사후확률 기반 풍부한 의사결정 (확률 진술, ROPE).
- **Notebook 03** — 위계 사전으로 다중검정 문제 자동 완화 (shrinkage).
- **Notebook 04** — Sparsity prior로 anchor-free DIF 검출.

### 13.5. 더 깊이 학습하려면

- **Simulation-Based Calibration (SBC)** — Talts et al. (2018), 표준 캘리브레이션 점검법.
- **Posterior Predictive Checks** — 모형이 자료를 재현하는지 확인.
- **Prior Sensitivity Analysis** — prior 폭을 0.5, 1.0, 2.0 등으로 변경해 결과 안정성 점검.
- **Robins-Breslow-Greenland 분산 추정량** — MH 신뢰구간 추가하여 coverage 양방향 비교 가능.
"""))


notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3 (ipykernel)", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11"},
    },
    "nbformat": 4, "nbformat_minor": 5,
}
NB_PATH.parent.mkdir(parents=True, exist_ok=True)
NB_PATH.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"Notebook saved: {NB_PATH}")
print(f"Total cells: {len(cells)}")
