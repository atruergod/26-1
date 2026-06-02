"""Notebook 00 빌더 — 개념 정립 + ICC 시각화.

이 스크립트는 .ipynb 파일을 JSON 형태로 생성한다.
실행:  python _build_notebook_00.py
"""
import json
from pathlib import Path

NB_PATH = Path(__file__).parent / "notebooks" / "00_intro_rasch_dif_icc.ipynb"


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}


def code(text: str) -> dict:
    return {"cell_type": "code", "metadata": {},
            "source": text.splitlines(keepends=True),
            "outputs": [], "execution_count": None}


cells = []

# ============================================================
# 0. 제목
# ============================================================
cells.append(md("""# Notebook 00 — Rasch 1PL DIF 검출: 개념 정립과 ICC 시각화

> 본 노트북은 **차별기능문항(Differential Item Functioning, DIF) 분석**의 개념적 토대를 다지고,
> Rasch 1PL 모형의 문항특성곡선(Item Characteristic Curve, ICC)을 통해 DIF의 본질을
> 시각적으로 이해한 후, 베이지안(Bayesian) 추정을 통해 DIF를 검출하는 첫 경험을 제공합니다.

학습 목표 (Learning Objectives)
- DIF 분석의 두 가지 주요 목적(검출, 원인·맥락 해석)을 이해한다.
- DIF 연구의 역사적 발전 과정과 핵심 인물·업적을 안다.
- Rasch 1PL/2PL의 베이지안 검출 방법의 장단점을 설명할 수 있다.
- 시뮬레이션 자료에서 ICC를 통해 DIF를 시각적으로 식별한다.
- 첫 베이지안 1PL DIF 모형을 적합하고 사후분포(posterior distribution)를 해석한다.
"""))

# ============================================================
# 1. DIF 분석의 주요 목적
# ============================================================
cells.append(md("""## 1. DIF 분석의 주요 목적 (Purposes of DIF Analysis)

차별기능문항(Differential Item Functioning, DIF) 분석은 동일한 잠재특성(latent trait, 보통 θ로 표기되는 능력)을
가진 서로 다른 집단의 응답자들이 특정 문항에서 체계적으로 다르게 반응하는 현상을 다룹니다.
DIF 분석의 목적은 단순히 통계적 차이를 찾는 데 있지 않고, 그 차이가 측정의 공정성(fairness)에
어떤 함의를 갖는지를 해석하는 데까지 이어집니다. 표준적으로 DIF 분석은 다음 두 단계로 구성됩니다.

### 목적 1. 검출 (Detection)

첫 번째 목적은 동일한 능력 수준의 두(또는 그 이상의) 집단—**참조집단(reference group)**과
**초점집단(focal group)**—사이에서 어떤 문항이 다르게 작동하는지를 통계적으로 식별하는 것입니다.
대표적 방법은 다음과 같습니다.

- **Mantel-Haenszel(MH) 통계량**: 능력 매칭 후 2×2 분할표 기반 검정
- **로지스틱 회귀(logistic regression) DIF 검정**: 능력 점수 + 집단 + 상호작용
- **IRT 기반 방법**: Lord's chi-square, Raju's area measures
- **SIBTEST**: 다차원성을 고려한 비모수 방법

DIF는 크게 두 형태로 구분됩니다.

- **균일 DIF (Uniform DIF)**: 능력 수준과 무관하게 한 집단에 일관되게 불리한 형태.
  Rasch 1PL에서는 두 집단의 난이도 차이 $\\Delta b_j = b_j^{\\text{focal}} - b_j^{\\text{ref}}$ 로 표현됨.
- **비균일 DIF (Non-uniform DIF)**: 능력 수준에 따라 방향이 달라지는 형태.
  두 집단의 변별도(discrimination, $a_j$) 차이로 인해 발생하며, 2PL 이상에서만 모형화 가능.

검출 단계의 산출물은 **"어떤 문항에서, 어느 정도 크기로, 어떤 방향의 DIF가 있는가"** 입니다.

### 목적 2. 원인 탐색과 맥락적 해석 (Source Identification and Contextual Interpretation)

두 번째 목적은 검출된 DIF의 의미를 해석하는 것입니다.
**통계적으로 DIF가 발견되었다고 해서 그 문항이 곧 편향(bias)된 것은 아닙니다.**
다음 세 가지 가능성을 구분해야 합니다.

1. **편향 (Item bias)** — 문항의 어휘, 맥락, 문화적 배경, 표현 방식이 특정 집단에 부당하게 불리하게 작용.
2. **구성개념 무관 분산 (Construct-irrelevant variance)** — 측정하려는 구성개념과 무관한 부차적 차원의 개입.
3. **영향 (Item impact)** — 두 집단 간 실제적·정당한 차이를 그대로 반영. **편향이 아님**.

이 단계에서는 통계 외에 다음과 같은 **질적 방법(qualitative methods)**이 함께 사용됩니다.

- 내용 전문가 검토 (content review)
- 인지 면담 (cognitive interview)
- 교육과정·문화적 맥락 분석

> ⚠️ **핵심 원칙**: DIF 분석의 두 번째 목적은 **편향(bias)과 영향(impact)을 구분하여
> 측정의 공정성(measurement fairness)을 평가**하는 것입니다.
> 본 자료는 첫 번째 목적인 "검출"을 베이지안 관점에서 다루지만, 검출된 결과를 해석할 때는
> 항상 두 번째 목적을 염두에 두어야 합니다.
"""))

# ============================================================
# 2. DIF 연구의 발전사 (Zumbo 3세대 프레임워크)
# ============================================================
cells.append(md("""## 2. DIF 연구의 발전사와 주요 업적 (Historical Development of DIF Research)

DIF 분석은 단순한 통계 기법이 아니라, **시험의 공정성(test fairness)**이라는 사회적·교육적
문제의식에서 출발해 60여 년에 걸쳐 발전해 온 연구 전통입니다.

### Zumbo의 3세대 프레임워크 (Three Generations of DIF Analyses)

**Zumbo (2007)** *"Three generations of DIF analyses: Considering where it has been, where it is
now, and where it is going"* (*Language Assessment Quarterly*, 4(2), 223–233)는 DIF 연구사를
다음 세 세대(generation)로 구분하였으며, 이후 DIF 종설의 표준 참고 틀로 자리잡았습니다.

- **1세대 (First Generation, ~1980년대 중반)**: "문항편향(item bias)" 시기.
  능력 매칭(ability matching)이 미숙하여 영향(impact)과 편향(bias)을 혼동.
- **2세대 (Second Generation, 1980년대 후반~2000년대 초)**: MH·로지스틱 회귀·IRT 기반 검출의 시기.
  "bias"가 "DIF"로 개명되며 통계적 정교화 달성.
- **3세대 (Third Generation, 2000년대 이후)**: 검출에서 **원인 탐색·맥락 해석**으로의 전환.
  측정 공정성(measurement fairness)과 타당도(validity) 이론에 통합.

본 자료는 Zumbo의 프레임워크를 따르되, 한국어 독자의 편의를 위해 더 세분화된 5단계로 제시하며,
각 단계 제목에 해당 세대 표기를 괄호로 부기합니다.

### 2.1. 태동기 — 시험 편향 개념의 출현  *(Zumbo 1세대 초기, 1960s~1970s)*

미국의 시민권 운동(Civil Rights Movement)과 1964년 민권법(Civil Rights Act of 1964) 이후,
표준화 시험이 인종·성별·사회경제적 집단에 공정한가에 대한 의문이 제기되면서 시험 편향(test bias) 연구가
본격화되었습니다.

- **Cleary (1968)** — 회귀(regression) 기반 예측편향(predictive bias) 정의.
- **Angoff & Ford (1973)** — 델타 도표(delta plot) 방법, 두 집단의 문항 난이도 차이 시각화.
- **Scheuneman (1979)** — 카이제곱(chi-square) 기반 집단 간 비교.

이 시기의 핵심 한계는 **능력(ability) 통제 없이 단순 정답률만을 비교**했다는 점이며,
이는 영향(impact)과 편향(bias)을 혼동하게 만들었습니다.

### 2.2. 방법론적 정립기 — MH와 IRT의 등장  *(Zumbo 1세대 → 2세대 전환점, 1980s)*

능력 수준을 통제한 상태에서 집단 차이를 평가해야 한다는 인식이 자리잡으면서, 현대적 DIF 검출
방법론이 정립되었습니다.

- **Lord (1980)** — IRT(Item Response Theory) 기반 검출 방법(Lord's chi-square) 제시.
- **Mellenbergh (1982)** — 로지스틱 회귀(logistic regression) DIF 검정의 초기 형태.
- **Holland & Thayer (1988)** — Mantel-Haenszel(MH) 절차를 DIF에 적용. **사실상 표준** 정립.
  ETS(Educational Testing Service)의 운영 절차에 채택됨.

> 🔑 이 시기의 결정적 사건은 Holland와 Thayer가 "bias"라는 가치판단적 용어 대신 **"Differential Item
> Functioning(DIF)"**이라는 중립적 용어를 도입한 것입니다. 이는 통계적 차이의 검출과 그에 대한
> 해석을 개념적으로 분리해야 함을 분명히 한 **DIF 연구사 최대의 용어적·개념적 전환**입니다.
> Zumbo가 1세대에서 2세대로의 전환점으로 지목한 사건이기도 합니다.

### 2.3. 방법론의 다양화와 정교화  *(Zumbo 2세대 성숙기, 1990s)*

1990년대는 검출 방법이 폭발적으로 다양해진 시기입니다.

- **Raju (1988, 1990)** — 두 집단 ICC 사이 면적을 측정하는 area measures.
- **Swaminathan & Rogers (1990)** — 로지스틱 회귀로 uniform과 non-uniform DIF 동시 검정.
- **Thissen, Steinberg & Wainer (1988, 1993)** — IRT 우도비 검정(IRT likelihood-ratio test).
- **Shealy & Stout (1993)** — SIBTEST, 다차원성을 명시적으로 고려.
- **ETS A/B/C 분류 기준** 표준화 — MH 효과크기 기반 실무적 의사결정 도구.

### 2.4. 해석적 전환 — 편향과 영향의 구분  *(Zumbo 2세대 → 3세대 진입, 1990s 후반~2000s)*

검출 기법이 풍부해지면서 연구의 초점은 "어떻게 더 잘 검출할 것인가"에서 **"검출된 DIF를 어떻게
해석할 것인가"**로 이동했습니다.

- **Camilli & Shepard (1994)** — 저서 *Methods for Identifying Biased Test Items*에서
  통계적 DIF가 곧 편향이 아니라는 점을 강조하며 내용 분석과 인지 면담의 중요성 부각.
- **Roussos & Stout (1996)** — DIF 원인 분석 프레임워크. 보조 차원(secondary dimension)이
  DIF의 근원이 될 수 있음을 이론적으로 정리.
- **Zumbo (1999, 2007)** — DIF 분석을 **실질적 탐구(substantive inquiry)**로 자리매김.
  검출-원인탐색-해석으로 이어지는 다단계 절차 정착. 본인의 2007년 논문에서 3세대 프레임을 정식화.

이 시기 이후 **"편향(bias) vs. 영향(impact)"의 구분**은 DIF 분석의 핵심 원칙이 됩니다.

### 2.5. 현대적 발전 — 위계·다차원·베이지안 접근  *(Zumbo 3세대 심화 + "4세대" 논의, 2000s~현재)*

21세기에 들어 컴퓨터 성능 향상과 함께 모형이 복잡해졌습니다.

- **위계 IRT (hierarchical IRT)** — 학생-학교 등 위계 구조에서의 DIF.
- **혼합 IRT (mixture IRT)** — 잠재계층(latent class) 수준의 DIF 식별.
- **베이지안(Bayesian) 접근의 본격적 도입**:
  - **Fox (2010)** *Bayesian Item Response Modeling*
  - **Soares, Gonçalves & Gamerman (2009)** — 위계 사전(hierarchical prior)을 통한 자동 축소(shrinkage)와 사후확률(posterior probability) 기반 판정.
  - **Frederickx 외** — hierarchical Bayesian DIF.
  - **Regularized DIF / Bayesian Lasso DIF** — sparsity 유도.
  - **Spike-and-slab prior**, **Horseshoe prior** — anchor-free 검출.

이러한 흐름은 다중검정(multiple testing), 소표본(small sample), anchor item 선택 문제 등
전통적 방법의 한계를 동시에 해결하려는 시도이며, **본 학습자료의 주제이기도 합니다**.

> 📝 **4세대(4th generation) 논의**: 일부 최근 연구(2015 이후)는 베이지안·정규화(regularization)·
> 머신러닝 기반 DIF 검출을 비공식적으로 "4세대"라고 부르기도 합니다. 다만 이 명명은 아직 정착된
> 표준이 아니며, **Zumbo의 3세대 안에서 통계 방법론 측면의 심화**로 보는 시각이 더 일반적입니다.
> 본 자료에서 다루는 베이지안 1PL/2PL DIF는 Zumbo의 3세대 흐름의 통계 방법론적 첨단에 위치합니다.

### 요약 (Summary)

> **Zumbo(2007)의 3세대 프레임워크에 본 자료의 5단계를 매핑하면**:
> - **1세대** (사회적 문제의식 → 통계적 시도): 2.1 + 2.2 초반
> - **2세대** (MH/IRT 기반 통계적 정교화): 2.2 후반 + 2.3
> - **3세대** (편향과 영향의 구분 + 베이지안 등 현대 방법론): 2.4 + 2.5

> ### 주요 참고문헌 (Key References for History)
> - Zumbo, B. D. (2007). Three generations of DIF analyses. *Language Assessment Quarterly*, 4(2), 223–233.
> - Holland, P. W., & Thayer, D. T. (1988). Differential item performance and the Mantel-Haenszel procedure.
> - Camilli, G., & Shepard, L. A. (1994). *Methods for Identifying Biased Test Items*.
> - Penfield, R. D., & Camilli, G. (2007). Differential item functioning and item bias. *Handbook of Statistics* 26.
> - Sireci, S. G., & Rios, J. A. (2013). Decisions that make a difference in detecting DIF. *Educational Research and Evaluation*.
"""))

# ============================================================
# 3. 베이지안 Rasch 1PL/2PL DIF의 장단점
# ============================================================
cells.append(md("""## 3. Rasch 1PL / 2PL의 베이지안 추정·검출의 장단점

### 3.1. 장점 (Advantages)

#### (1) 소표본·희소집단에서의 안정성 (Stability under small / sparse samples)
빈도주의(frequentist) MH나 최대가능도(MLE) 기반 IRT는 표본이 작거나 집단 간 크기 불균형이
심할 때 추정치가 불안정해지고 표준오차(standard error)가 과대해집니다.
베이지안 접근은 **사전분포(prior)를 통한 정규화(regularization) 효과**로 소표본에서도
사후분포가 수렴하고 합리적인 추정을 제공합니다.
특히 minority group의 DIF 검출에서 강점이 두드러집니다.

#### (2) 불확실성의 자연스러운 정량화 (Natural quantification of uncertainty)
DIF 모수(예: 집단 간 난이도 차이 $\\Delta b$)의 **사후분포(posterior distribution) 전체**를
얻을 수 있어, 점추정치와 신뢰구간(confidence interval)을 넘어 **사후확률(posterior probability)**에
기반한 직접적 진술이 가능합니다.

> 예: *"이 문항이 집단 A에 불리하게 작동할 확률이 0.96이다."*

또한 **ROPE(Region of Practical Equivalence)** 기반 판정과 **신용구간(credible interval)** 해석이
빈도주의 p-value 해석의 난점을 극복합니다.

#### (3) 다중검정 문제의 완화 (Mitigation of multiple testing)
DIF 분석은 본질적으로 수십~수백 개 문항을 동시에 검정하는 **다중비교(multiple comparison)** 상황입니다.
빈도주의에서는 Bonferroni, Benjamini-Hochberg 등 보정이 필요하지만,
베이지안 **위계모형(hierarchical model)**은 문항별 DIF 모수를 공통 분포 $\\Delta b_j \\sim N(0, \\tau^2)$에서
추출되는 것으로 모델링하여 **자동적인 축소(shrinkage)**를 통해 거짓양성(false positive)을 줄입니다.

#### (4) 사전정보의 통합 (Integration of prior information)
선행 연구, 전문가 판단, 과거 시행 자료 등을 사전분포로 반영할 수 있습니다.
"대부분의 문항은 DIF가 없거나 작을 것"이라는 가정을 **spike-and-slab prior**나 **horseshoe prior**로
표현해 sparsity를 유도할 수 있으며, 이는 **변수 선택(variable selection)** 관점의 현대적 DIF 탐색으로
이어집니다.

#### (5) 복잡한 모형 확장의 용이성 (Easy extension to complex models)
다집단(multi-group), 다국면(multi-faceted), 다층(multilevel; 학생-학교), 종단(longitudinal) 구조,
누락자료(missingness) 등을 **하나의 통일된 틀**에서 모델링하기가 빈도주의보다 훨씬 자연스럽습니다.
Stan, NumPyro, JAGS, brms, NIMBLE 등으로 구현 가능합니다.

#### (6) Anchor item 선정 문제의 완화 (Anchor-free DIF detection)
전통적 DIF 검출은 "DIF가 없는 anchor item"을 사전에 선택해야 하지만, **이 선택이 결과를 좌우**합니다.
베이지안 접근에서는 모든 문항의 DIF 모수에 **sparsity 유도 prior**를 부여하여 anchor 선정 자체를
데이터 기반으로 처리할 수 있습니다 (Bayesian Lasso DIF, regularized DIF).

### 3.2. 단점 및 고려사항 (Disadvantages and Considerations)

#### (1) 계산 비용 (Computational cost)
MCMC(Markov Chain Monte Carlo) 수렴에 시간이 걸리고,
**사후 진단(R-hat, ESS, trace plot, divergence)** 이 필요합니다.
변분추론(variational inference, VI)이나 INLA로 일부 완화 가능합니다.

#### (2) 사전분포 민감성 (Sensitivity to prior choice)
사전 선택이 결과에 영향을 줄 수 있어 **민감도 분석(sensitivity analysis)**이 권장됩니다.
특히 소표본일수록 사전의 영향이 큽니다.

#### (3) 해석·보고의 학습곡선 (Learning curve for interpretation)
응용 연구자와 실무자에게 사후분포 기반 해석이 익숙하지 않을 수 있으며,
학술지 심사자가 빈도주의 결과를 추가 요구하는 경우도 있습니다.

#### (4) 표준화된 효과크기 기준의 부재 (Lack of standardized effect-size criteria)
빈도주의 MH에는 ETS의 A/B/C 분류 같은 합의된 효과크기 기준이 있지만,
베이지안에서는 아직 표준화가 덜 되어 있습니다.
사후확률 기반 판정 기준(예: $P(|\\Delta b| > 0.5 \\mid \\text{data}) > 0.95$)을 연구자가 직접 정해야 합니다.

### 3.3. 종합 (Summary)

> 베이지안 1PL/2PL DIF는 **소표본·다문항·다집단·위계구조** 상황에서, 그리고 **불확실성의 정직한
> 표현**과 **anchor-free 접근**이 중요한 연구에서 분명한 이점을 가집니다.
> 반면 대규모 표본의 단순한 두 집단 비교에서는 MH 같은 전통 기법이 여전히 간결하고 충분할 수 있습니다.
> **"무조건 우월"이 아니라 연구 맥락에 따른 비교 우위**로 이해하는 것이 적절합니다.
"""))

# ============================================================
# 4. 본 노트북의 학습 흐름
# ============================================================
cells.append(md("""## 4. 본 노트북의 학습 흐름

본 노트북은 다음 순서로 진행됩니다.

1. **백엔드 설정** — Stan(기본) 또는 NumPyro(Mac/Linux 선택) 선택, Windows UTF-8 안전장치.
2. **Rasch 1PL 모형 수식 정리** — 식별성(identifiability), 가정.
3. **시뮬레이션 자료 생성** — 10문항, 두 집단 각 300명.
   - 문항 5: 강한 uniform DIF ($\\Delta b = +0.8$)
   - 문항 8: 약한 uniform DIF ($\\Delta b = -0.4$)
   - 나머지 8문항: DIF 없음
4. **ICC 시각화** — 모든 문항의 두 집단 ICC를 격자로 표시.
5. **능력 분포 비교** — 두 집단의 진짜 능력 분포.
6. **개념 정리** — bias vs impact, uniform vs non-uniform DIF.
7. **첫 베이지안 적합** — Stan(또는 NumPyro)으로 1PL DIF 모형 적합.
8. **사후분포 시각화** — Forest plot, posterior density, ROPE.
9. **MH 비교** — 빈도주의 결과와의 대조.
10. **요약 및 다음 노트북 안내**.
"""))

# ============================================================
# 5. 백엔드 선택
# ============================================================
cells.append(md("""## 5. 백엔드 선택 (Backend Selection)

> ⚙️ **사용자 설정** — 아래 셀에서 백엔드를 선택하세요.
>
> - `"stan"`: cmdstanpy 사용. **모든 OS 지원, 기본값**.
> - `"numpyro"`: NumPyro + jax 사용. **Mac/Linux 권장, Windows 비권장**
>   (Windows에는 jax 공식 빌드가 제공되지 않음).
>
> 잘못된 선택을 해도 자동 fallback이 작동하므로 노트북이 멈추지 않습니다.

### 5.1. Windows 사용자를 위한 UTF-8 안내 (Important for Windows users)

Windows 환경에서 cmdstanpy(Stan 백엔드)를 사용할 때 종종 다음과 같은 인코딩 오류가 발생합니다.

```
UnicodeDecodeError: 'cp949' codec can't decode byte 0x... in position ...
```

이는 Stan 컴파일러의 출력이 UTF-8인데 Windows의 기본 인코딩이 cp949(한국어) 또는 cp1252인
환경에서 발생하는 문제입니다. **해결 방법**은 다음 셋 중 하나입니다.

1. **(가장 안정적)** Jupyter를 UTF-8 모드로 시작하세요. 명령 프롬프트에서:
   `python -X utf8 -m jupyter notebook`
   또는 환경 변수를 미리 지정:
   `set PYTHONUTF8=1` 후 `jupyter notebook` 실행.
2. **(시스템 전역)** 시스템 환경 변수 `PYTHONUTF8=1`을 추가. 재로그인 후 모든 Python 세션에서 적용됨.
3. **(미봉책)** 아래 셀이 자동으로 인코딩을 UTF-8로 재설정 시도. 이미 실행 중인 Jupyter에서는
   완전히 동작하지 않을 수 있으므로 1번 방법을 권장합니다.

Mac/Linux 사용자는 보통 기본이 UTF-8이므로 이 작업이 필요하지 않습니다.
"""))

cells.append(code("""# ============================================================
# Windows UTF-8 defensive setup
# (Jupyter가 이미 cp949/cp1252 환경에서 시작된 경우의 인코딩 오류 방어)
# ============================================================
# cmdstanpy 가 .stan 파일이나 컴파일러 출력을 읽을 때 발생하는
# UnicodeDecodeError 를 막기 위해 builtins.open / subprocess.Popen 을 patch 한다.
# Mac / Linux 에서는 아무 효과 없음.
import os, sys, platform, builtins, subprocess, io

if platform.system() == "Windows":
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    os.environ.setdefault("PYTHONUTF8", "1")

    # 1) sys.stdout/stderr 재설정
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, Exception):
        pass

    # 2) builtins.open 패치 — 텍스트 모드일 때 encoding 기본값을 UTF-8 로
    if not getattr(builtins, "_difbayes_open_patched", False):
        _orig_open = builtins.open
        def _utf8_open(file, mode="r", buffering=-1, encoding=None,
                       errors=None, newline=None, closefd=True, opener=None):
            if "b" not in mode and encoding is None:
                encoding = "utf-8"
                if errors is None:
                    errors = "replace"
            return _orig_open(file, mode, buffering, encoding, errors,
                              newline, closefd, opener)
        builtins.open = _utf8_open
        builtins._difbayes_open_patched = True

    # 3) subprocess.Popen 패치 — text=True 일 때 UTF-8 강제
    if not getattr(subprocess.Popen, "_difbayes_patched", False):
        _orig_popen_init = subprocess.Popen.__init__
        def _utf8_popen_init(self, *args, **kwargs):
            if kwargs.get("text") or kwargs.get("universal_newlines"):
                kwargs.setdefault("encoding", "utf-8")
                kwargs.setdefault("errors", "replace")
            return _orig_popen_init(self, *args, **kwargs)
        subprocess.Popen.__init__ = _utf8_popen_init
        subprocess.Popen._difbayes_patched = True

    import locale
    print("[Windows UTF-8 check]")
    print(f"  PYTHONUTF8       = {os.environ.get('PYTHONUTF8', '(not set)')}")
    print(f"  PYTHONIOENCODING = {os.environ.get('PYTHONIOENCODING', '(not set)')}")
    print(f"  locale.getpreferredencoding() = {locale.getpreferredencoding()}")
    print(f"  sys.stdout.encoding          = {sys.stdout.encoding}")
    print("  builtins.open patched        = True")
    print("  subprocess.Popen patched     = True")
    if locale.getpreferredencoding().lower() not in ("utf-8", "utf8", "cp65001"):
        print()
        print("  Note: locale은 여전히 cp949/cp1252 이지만,")
        print("        open/Popen 이 모두 UTF-8 로 패치되었으므로 정상 동작해야 합니다.")
        print("        그래도 오류가 계속되면 'python -X utf8 -m jupyter notebook' 으로 재시작.")
else:
    print(f"[OS = {platform.system()}]  No UTF-8 patching needed.")
"""))

cells.append(code("""# ============================================================
# Backend selection (백엔드 선택)
# ============================================================
# 변경하려면 아래 값만 수정하세요.
BACKEND = "stan"          # "stan" (default) or "numpyro"

# Auto-fallback safeguard (자동 안전장치)
import platform, importlib.util, warnings
def _resolve(req):
    req = req.lower()
    if req == "numpyro":
        ok_jax = importlib.util.find_spec("jax") is not None
        ok_np  = importlib.util.find_spec("numpyro") is not None
        if not (ok_jax and ok_np):
            warnings.warn("[notebook] numpyro/jax unavailable -> falling back to Stan.")
            return "stan"
    return "stan" if req != "numpyro" else "numpyro"

BACKEND = _resolve(BACKEND)
print(f"Operating system : {platform.system()}")
print(f"Active backend   : {BACKEND}")
"""))

# ============================================================
# 6. 모듈 import
# ============================================================
cells.append(md("""## 6. 모듈 import

`difbayes` 공통 모듈을 import하고, matplotlib 기본 스타일을 설정합니다.
"""))

cells.append(code("""import sys
from pathlib import Path

# 상위 디렉터리에 있는 difbayes 패키지를 import 경로에 추가
PROJECT_ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from difbayes import simulate, visualize, frequentist, diagnostics, models

# 시각화 기본 설정
plt.rcParams.update({
    "figure.dpi": 110,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.size": 10,
})

print("difbayes loaded. Project root =", PROJECT_ROOT)
"""))

# ============================================================
# 7. Rasch 1PL 모형
# ============================================================
cells.append(md("""## 7. Rasch 1PL 모형 — 수식과 가정

### 7.1. 모형 정의

Rasch 1PL(one-parameter logistic) 모형은 응답자 $i$ 가 문항 $j$ 에 정답할 확률을
**능력(ability) $\\theta_i$** 와 **난이도(difficulty) $b_j$** 의 차로 표현합니다.

$$
P(Y_{ij} = 1 \\mid \\theta_i, b_j) = \\text{sigmoid}(\\theta_i - b_j) = \\frac{1}{1 + \\exp(-(\\theta_i - b_j))}
$$

DIF가 존재하는 경우, 집단 $g_i \\in \\{0=\\text{ref}, 1=\\text{focal}\\}$ 에 따라 난이도가 달라집니다.

$$
P(Y_{ij} = 1 \\mid \\theta_i, b_j, g_i) = \\text{sigmoid}\\big(\\theta_i - (b_j + g_i \\cdot \\Delta b_j)\\big)
$$

- $\\Delta b_j > 0$ : focal 집단에 더 어려움(불리).
- $\\Delta b_j < 0$ : focal 집단에 더 쉬움(유리).
- $\\Delta b_j = 0$ : DIF 없음.

### 7.2. 가정 (Assumptions)

1. **단일차원성(unidimensionality)**: 한 가지 잠재특성만 측정.
2. **국소독립성(local independence)**: $\\theta$ 가 주어지면 응답들이 독립.
3. **등변별성(equal discrimination)**: 모든 문항의 변별도 $a_j = 1$ 로 고정 (Rasch의 특수성).

### 7.3. 식별성 (Identifiability)

$\\theta_i \\to \\theta_i + c$ 와 $b_j \\to b_j + c$ 의 동시 이동이 likelihood를 바꾸지 않으므로
하나의 제약이 필요합니다. 본 자료에서는 **$\\theta_i \\sim N(0, 1)$ 사전**으로 $\\theta$ 의 평균을 0 근처에
약하게 묶어두는 방식을 사용합니다 (soft identifiability).

### 7.4. ⚠️ 1PL의 한계 — Non-uniform DIF 불가

> **중요**: Rasch 1PL은 등변별성을 가정하므로 **uniform DIF($\\Delta b$)만** 모형화할 수 있습니다.
> 능력 수준에 따라 집단 격차의 방향이 달라지는 **non-uniform DIF**는 변별도 차이($\\Delta a$)로
> 발생하며 **2PL 이상**에서만 다룰 수 있습니다.
> 본 노트북의 시뮬레이션은 1PL DGP를 사용하므로 이 한계가 문제되지 않습니다.
> 2PL 확장은 **부록 A**에서 다룹니다.
"""))

# ============================================================
# 8. 시뮬레이션 자료 생성
# ============================================================
cells.append(md("""## 8. 시뮬레이션 자료 생성

진짜 DIF가 있는 문항을 미리 알고 있는 상태에서, 모형이 얼마나 잘 찾아내는지 확인할 것입니다.

- $n_{\\text{ref}} = 300$, $n_{\\text{focal}} = 300$
- $J = 10$ 문항, $b_j$ = -2 ~ +2 등간
- 문항 5(인덱스 4): $\\Delta b = +0.8$ — 강한 uniform DIF (focal에 불리)
- 문항 8(인덱스 7): $\\Delta b = -0.4$ — 약한 uniform DIF (focal에 유리)
- 두 집단의 능력 평균 동일 ($\\mu_{\\text{ref}} = \\mu_{\\text{focal}} = 0$) — 즉 *impact 없음*.
"""))

cells.append(code("""data = simulate.scenario_intro_10items(seed=2026)

print(f"표본 크기 : N = {data.N},  n_ref = {data.n_ref},  n_focal = {data.n_focal}")
print(f"문항 수   : J = {data.J}")
print()
print("문항별 진짜 모수 (true parameters):")
truth_df = pd.DataFrame({
    "Item": np.arange(1, data.J + 1),
    "b_true (ref)": data.b_true.round(2),
    "Δb_true (DIF)": data.delta_b_true.round(2),
    "b_focal": (data.b_true + data.delta_b_true).round(2),
    "DIF": ["YES" if abs(d) > 1e-6 else "—" for d in data.delta_b_true],
})
print(truth_df.to_string(index=False))
"""))

# ============================================================
# 9. ICC 시각화
# ============================================================
cells.append(md("""## 9. 문항특성곡선(ICC) 시각화

각 문항에 대해 두 집단의 ICC를 한 그림에 겹쳐 그립니다.
**두 곡선이 겹쳐 있으면 DIF 없음**, 분리되어 있으면 DIF가 존재합니다.
DIF가 있는 문항(5, 8)의 패널 제목이 빨간색으로 강조됩니다.
"""))

cells.append(code("""fig, axes = visualize.plot_icc_grid(
    b_true=data.b_true,
    delta_b_true=data.delta_b_true,
    ncols=5,
)
fig.savefig("../outputs/00_icc_grid.png", dpi=120, bbox_inches="tight")
plt.show()
"""))

cells.append(md("""**관찰 포인트**

- 문항 1~4, 6, 7, 9, 10: 두 집단 ICC가 거의 일치 → DIF 없음.
- 문항 5 ($\\Delta b = +0.8$): focal(빨간 점선)이 ref(파란 실선)보다 오른쪽으로 이동.
  같은 능력 $\\theta$ 에서 focal의 정답 확률이 더 낮음.
- 문항 8 ($\\Delta b = -0.4$): focal이 ref보다 왼쪽으로 이동. focal의 정답 확률이 더 높음.

이것이 **uniform DIF**의 시각적 형태입니다. 능력 수준과 무관하게 한 집단에 일관되게
유리/불리한 차이가 나타납니다.
"""))

# 단일 문항 확대 그림
cells.append(code("""# 문항 5 (강한 DIF) 확대
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
j_strong = 4   # 문항 5 (0-based index)
visualize.plot_icc_two_groups(
    b_ref=data.b_true[j_strong],
    b_focal=data.b_true[j_strong] + data.delta_b_true[j_strong],
    ax=axes[0],
    title=f"Item {j_strong+1}: strong DIF, Δb = {data.delta_b_true[j_strong]:+.2f}",
)

j_weak = 7   # 문항 8
visualize.plot_icc_two_groups(
    b_ref=data.b_true[j_weak],
    b_focal=data.b_true[j_weak] + data.delta_b_true[j_weak],
    ax=axes[1],
    title=f"Item {j_weak+1}: weak DIF, Δb = {data.delta_b_true[j_weak]:+.2f}",
)
fig.savefig("../outputs/00_icc_dif_items_zoom.png", dpi=120, bbox_inches="tight")
plt.show()
"""))

# 능력 분포 비교
cells.append(md("""### 능력 분포 비교 (Ability Distributions)

본 시나리오에서는 두 집단의 능력 평균이 동일하게 설정되어 있습니다.
즉 집단 간 **impact**(실질적 능력 차이)는 없고, DIF만 존재하는 깨끗한 상황입니다.
"""))

cells.append(code("""fig, ax = visualize.plot_ability_distributions(data.theta, data.group)
fig.savefig("../outputs/00_ability_distributions.png", dpi=120, bbox_inches="tight")
plt.show()
"""))

# ============================================================
# 10. 개념 정리
# ============================================================
cells.append(md("""## 10. 핵심 개념 정리 (Key Concepts)

### 10.1. 편향(Bias) vs 영향(Impact) — 가장 중요한 개념 구분

DIF 분석을 처음 접하는 분들이 가장 헷갈리는 지점이 **편향(item bias)과 영향(item impact)의 차이**입니다.
간단한 비유로 시작합시다.

#### 비유: 수학 시험과 두 집단

공대생(group A)과 문과생(group B)에게 수학 시험을 보입니다. 두 가지 종류의 차이가 있을 수 있습니다.

- **(가능성 1)** 공대생이 평균적으로 수학 능력 자체가 더 높을 수 있다.
- **(가능성 2)** 어떤 문항이 "기계 부품의 응력"이라는 공학 용어를 사용해서,
  같은 수학 실력의 공대생/문과생이라도 공대생만 잘 풀 수 있다.

이 두 요소는 **서로 다른 종류의 차이**입니다. 각각 영향(impact)과 편향(bias)에 해당합니다.

#### 두 차원, 네 가지 조합

| 차원 | 무엇이 다른가? | DIF 분석의 대상? |
|---|---|---|
| **영향 (Impact)** | 두 집단의 **능력 분포(θ) 자체**가 다름 (예: 평균 공대생 > 평균 문과생) | **아님** (불공정과 무관) |
| **편향 (Bias)** | **같은 능력**의 응답자라도 **이 문항**에 대한 응답확률이 집단 간 다름 | **그것이 바로 DIF** |

영향과 편향은 **서로 독립적으로 공존**할 수 있어 네 가지 조합이 가능합니다.

| 시나리오 | 두 집단의 능력 분포 | 같은 θ에서 문항 응답확률 | 무엇이 일어나는가 |
|---|---|---|---|
| **A** | 같음 | 같음 | 완전 동등. 평균 점수도 같고, ICC도 두 집단이 일치. |
| **B** | **같음** | **다름** | **DIF만 존재** — 능력은 같은데 문항 자체가 한 집단에 불리. |
| **C** | **다름** | 같음 | **Impact만 존재** — 한 집단이 평균적으로 능력이 높을 뿐, 문항은 공정. |
| **D** | 다름 | 다름 | 영향과 편향이 둘 다 존재. |

> 🔍 **본 노트북 시뮬레이션은 시나리오 B**입니다.
> §8에서 능력 평균을 동일하게($\\mu_{ref} = \\mu_{focal} = 0$) 설정했지만,
> 문항 5와 8에는 $\\Delta b \\neq 0$ 을 의도적으로 넣었습니다.
> §9의 ICC 격자 그림에서 문항 5·8 만 두 곡선이 분리되어 나타난 것이 그 시각적 증거입니다.

#### DIF 분석은 무엇을 어떻게 가려내는가

위 네 시나리오에서 DIF 분석의 임무는 **B와 D를 식별**하는 것입니다.
즉 *같은 능력에서 문항 응답이 다른 경우*만을 골라냅니다.
어떻게 가능할까요? **능력 $\\theta$ 를 통제(control)** 함으로써 가능합니다.

- **빈도주의 MH**: 응답자를 능력 점수의 분위(stratum)로 나누어 매칭. 같은 분위 안에서 집단 차이를 봄.
- **베이지안 1PL**: 모형 안에서 $\\theta_i$ 를 명시적 모수로 두어 추정. 같은 $\\theta_i$ 에서 집단 효과 $\\Delta b_j$ 만 분리.

이 통제(control) 절차가 있기에 DIF 분석은 시나리오 C(impact만)와 시나리오 B(DIF만)를 구별할 수 있습니다.
**능력이 다르다는 사실 자체**는 검출 대상이 아닙니다 — 그것은 시험 결과의 정당한 일부일 수 있습니다.

#### 그러면 "편향(bias) = DIF" 인가? — 아닙니다

**DIF가 검출되었다고 곧 편향(bias)인 것은 아닙니다.** DIF는 통계적 사실에 불과하고,
그것이 부당한 편향인지 정당한 영향인지의 해석은 별개의 작업입니다.

- 정말 부당하게 한 집단에 불리하면 → **item bias**
- 측정하려는 개념과 무관한 부차적 차원 때문이면 → **construct-irrelevant variance**
- 두 집단의 정당한 실력 차이의 일부이면 → **item impact**

이 해석 단계가 §1에서 본 **DIF 분석의 목적 2 (원인 탐색·맥락 해석)** 입니다.
본 노트북은 검출(목적 1)만 다루지만, 결과 해석 시 항상 이 구분을 떠올려야 합니다.

### 10.2. Uniform vs Non-uniform DIF

| 구분 | 메커니즘 | 1PL? | 2PL? |
|---|---|---|---|
| **Uniform DIF** | $\\Delta b_j$ (난이도 차이) | ✓ 검출 가능 | ✓ 검출 가능 |
| **Non-uniform DIF** | $\\Delta a_j$ (변별도 차이) | ✗ 모형 불가 | ✓ 검출 가능 |

본 노트북은 1PL 기반이므로 **uniform DIF만** 다룹니다. Non-uniform DIF는 **부록 A**에서 2PL로 확장합니다.
"""))

# ============================================================
# 11. 베이지안 적합
# ============================================================
cells.append(md("""## 11. 첫 베이지안 1PL DIF 적합

이제 모형이 진짜 DIF 문항을 찾아낼 수 있는지 확인합니다.
백엔드는 위에서 설정한 `BACKEND`(`"stan"` 또는 `"numpyro"`) 를 사용합니다.

**모형:**
$$
\\text{logit } P(Y_{ij} = 1) = \\theta_i + g_i \\mu_{\\text{focal}} - (b_j + g_i \\Delta b_j)
$$

**Priors:**
- $\\theta_i \\sim N(0, 1)$
- $b_j \\sim N(0, 2)$
- $\\Delta b_j \\sim N(0, 1)$ (weakly-informative; non-hierarchical)
- $\\mu_{\\text{focal}} \\sim N(0, 1)$ (집단 impact)

> ⏱ Stan은 첫 실행 시 모형 컴파일에 ~30초 정도 걸릴 수 있습니다 (이후 캐시됨).
"""))

cells.append(code("""# 모형 적합 (백엔드는 위에서 설정된 BACKEND 사용)
fit = models.fit_rasch_dif(
    Y=data.Y,
    group=data.group,
    backend=BACKEND,
    n_chains=4,
    n_warmup=500,
    n_samples=1000,
    prior_sigma_delta=1.0,
    seed=2026,
)

print(f"Backend used: {fit['backend']}")
print(f"Sample shape (delta): {fit['samples']['delta'].shape}  # (n_chains, n_draws, J)")
"""))

# 진단
cells.append(code("""# 진단 — R-hat, ESS 점검
summary = fit["summary"]
# delta 모수만 필터
delta_summary = summary[summary["parameter"].str.startswith("delta[")]
delta_summary = delta_summary.reset_index(drop=True)
delta_summary.insert(0, "Item", np.arange(1, data.J + 1))
delta_summary["truth"] = data.delta_b_true.round(2)
print("Posterior summary of Δb_j:")
print(delta_summary.round(3).to_string(index=False))
"""))

# Forest plot
cells.append(md("""### 11.1. 사후 Forest Plot

각 문항의 $\\Delta b_j$ 사후 분포를 한 그림에 표시합니다.
- 검은 점·막대: 사후 평균 ± 95% 신용구간(credible interval).
- 빨간 × : 진짜 값 (시뮬레이션이기에 알고 있음).
- 초록 음영: ROPE [-0.2, 0.2] — 실질적으로 0과 다르지 않다고 볼 영역.
"""))

cells.append(code("""# delta 사후 표본을 (S, J) 형태로 reshape
delta_samples = fit["samples"]["delta"]                # (n_chains, n_draws, J)
delta_samples = delta_samples.reshape(-1, data.J)      # (S, J)

fig, ax = visualize.plot_dif_forest(
    delta_samples,
    item_labels=[f"Item {j+1}" for j in range(data.J)],
    truth=data.delta_b_true,
    rope=(-0.2, 0.2),
    title="Posterior of Δb_j  (non-hierarchical Bayesian 1PL)"
)
fig.savefig("../outputs/00_posterior_forest.png", dpi=120, bbox_inches="tight")
plt.show()
"""))

# 단일 문항 사후 밀도
cells.append(md("""### 11.2. 강한 DIF 문항(문항 5) 사후 밀도

문항 5의 $\\Delta b_5$ 사후 분포를 자세히 살펴봅니다.
- 사후 평균이 진짜 값 +0.8 근처인지?
- 분포가 ROPE 밖에 명확히 위치하는지?
"""))

cells.append(code("""samples_item5 = delta_samples[:, 4]
fig, ax = visualize.plot_posterior_density(
    samples_item5, truth=data.delta_b_true[4],
    rope=(-0.2, 0.2),
    title="Posterior density: Delta b (Item 5)"
)
fig.savefig("../outputs/00_posterior_item5.png", dpi=120, bbox_inches="tight")
plt.show()

# Posterior-probability statements
p_above = diagnostics.posterior_prob_above_threshold(samples_item5, 0.0, "greater")
p_rope  = diagnostics.rope_probability(samples_item5, rope=(-0.2, 0.2))
print(f"P(Delta_b > 0 | data)               = {p_above:.3f}")
print(f"P(Delta_b in ROPE [-0.2, 0.2] | data) = {p_rope['in_rope']:.3f}")
print(f"P(Delta_b > 0.2 | data)              = {p_rope['above']:.3f}")
print(f"P(Delta_b < -0.2 | data)             = {p_rope['below']:.3f}")
"""))


# ============================================================
# 11.3. 표본 크기 효과 — 30+30 보조 시뮬레이션
# ============================================================
cells.append(md("""### 11.3. 표본 크기 효과 — 30+30 보조 시뮬레이션

지금까지는 $n_{ref} = n_{focal} = 300$ 으로 비교적 충분한 표본을 사용했습니다.
표본이 **10배 작아지면(30+30)** 베이지안 사후분포는 어떻게 변할까요?

**예상되는 변화 (이론)**:
- 사후 표준편차(posterior SD)가 약 $\\sqrt{10} \\approx 3.16$ 배 넓어짐.
- weakly-informative prior의 상대적 영향력이 커져, 추정치가 0 쪽으로 더 축소(shrinkage)됨.
- 95% 신용구간이 ROPE와 자주 겹쳐 **의사결정이 "보류(undecided)"**로 빠지는 경우가 늘어남.
- 같은 자료에 MH를 돌리면 분위(stratum)당 6명 정도라 매우 불안정해짐.

이제 실제로 확인해봅시다. 같은 자료생성과정으로 표본만 줄여 다시 적합합니다.
"""))

cells.append(code("""# 작은 표본 자료 생성 — 진짜 모수는 동일
data_small = simulate.scenario_intro_10items(seed=2026)
import numpy as np
rng = np.random.default_rng(99)
ref_idx_all   = np.where(data_small.group == 0)[0]
focal_idx_all = np.where(data_small.group == 1)[0]
sel_ref   = rng.choice(ref_idx_all,   size=30, replace=False)
sel_focal = rng.choice(focal_idx_all, size=30, replace=False)
sel = np.concatenate([sel_ref, sel_focal])

Y_small      = data_small.Y[sel]
group_small  = data_small.group[sel]
print(f"Small sample: N = {len(sel)}, n_ref = 30, n_focal = 30")
"""))

cells.append(code("""# 베이지안 적합 (30+30)
fit_small = models.fit_rasch_dif(
    Y=Y_small, group=group_small, backend=BACKEND,
    n_chains=4, n_warmup=500, n_samples=1000,
    prior_sigma_delta=1.0, seed=2026,
)
delta_small = fit_small["samples"]["delta"].reshape(-1, data.J)
print(f"Posterior shape (small): {delta_small.shape}")
"""))

cells.append(code("""# 두 사후 분포 나란히 비교
fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharex=True)
visualize.plot_dif_forest(
    delta_samples, truth=data.delta_b_true, rope=(-0.2, 0.2),
    item_labels=[f"Item {j+1}" for j in range(data.J)],
    ax=axes[0],
    title="n_ref = n_focal = 300  (baseline)"
)
visualize.plot_dif_forest(
    delta_small, truth=data.delta_b_true, rope=(-0.2, 0.2),
    item_labels=[f"Item {j+1}" for j in range(data.J)],
    ax=axes[1],
    title="n_ref = n_focal = 30  (10x smaller)"
)
fig.savefig("../outputs/00_smallsample_compare.png", dpi=120, bbox_inches="tight")
plt.show()
"""))

cells.append(code("""# 사후폭(SD) 비교 + 의사결정 변화
def decide(samples, rope=(-0.2, 0.2)):
    lo, hi = np.quantile(samples, [0.025, 0.975])
    if hi < rope[0] or lo > rope[1]:
        return "Reject null (DIF)"
    if lo >= rope[0] and hi <= rope[1]:
        return "Accept null (no DIF)"
    return "Undecided"

rows = []
for j in range(data.J):
    rows.append(dict(
        Item=j+1,
        truth=data.delta_b_true[j],
        sd_300=delta_samples[:, j].std().round(3),
        sd_30=delta_small[:, j].std().round(3),
        ratio=(delta_small[:, j].std() / delta_samples[:, j].std()).round(2),
        decision_300=decide(delta_samples[:, j]),
        decision_30=decide(delta_small[:, j]),
    ))
print(pd.DataFrame(rows).to_string(index=False))
"""))

cells.append(md("""**관찰 가이드**

- `sd_30 / sd_300` 비가 이론치 $\\sqrt{10} \\approx 3.16$ 에 근접해야 합니다.
- 진짜 강한 DIF인 **문항 5**는 30+30에서도 양성 검출 가능성이 있지만,
  95% 신용구간이 훨씬 넓어집니다.
- 진짜 약한 DIF인 **문항 8**은 30+30에서는 사실상 식별 불가 (Undecided).
- DIF가 없는 문항(1, 2, 3, ...)도 30+30에서는 0 근처로 좁혀지지 않고 폭이 ±0.7 안팎.

**핵심 메시지**

> 표본이 작아질수록 베이지안 사후는 **더 정직하게 "모른다"를 표현**합니다.
> 같은 자료에서 MH는 분위가 비어 불안정하지만, 베이지안은 사후분포가 단지 *넓어질 뿐* 작동을 멈추지 않습니다.
> 이것이 **장점 #1(소표본 안정성)**과 **장점 #2(불확실성 정량화)**가 함께 발휘되는 모습입니다.
> 이 주제는 Notebook 01에서 반복 시뮬레이션으로 정밀하게 다룹니다.
"""))

# ============================================================
# 12. MH 비교
# ============================================================
cells.append(md("""## 12. 빈도주의 MH 결과와 비교

같은 자료에 대해 **Mantel-Haenszel** 통계량과 비교합니다.
ETS A/B/C 분류는 다음과 같이 해석합니다:
- **A**: 무시 가능 ($|\\Delta_{MH}| < 1.0$ 또는 비유의)
- **B**: 중간 ($1.0 \\leq |\\Delta_{MH}| < 1.5$ 이고 유의)
- **C**: 큼 ($|\\Delta_{MH}| \\geq 1.5$ 이고 유의)
"""))


cells.append(code("""mh_results = frequentist.mantel_haenszel_all(data.Y, data.group, n_strata=5)

post_mean = delta_samples.mean(axis=0)
post_p_zero = np.array([
    diagnostics.posterior_prob_above_threshold(delta_samples[:, j], 0.2, "two-sided")
    for j in range(data.J)
])

compare_df = pd.DataFrame({
    "Item": np.arange(1, data.J + 1),
    "delta_b_true": data.delta_b_true.round(2),
    "MH delta_MH": [r.delta_mh for r in mh_results],
    "MH p-value": [r.pvalue for r in mh_results],
    "ETS class": [r.ets_class for r in mh_results],
    "Bayes mean": post_mean.round(3),
    "P(|delta|>0.2)": post_p_zero.round(3),
})
print("MH vs Bayesian comparison:")
print(compare_df.round(3).to_string(index=False))
"""))

cells.append(md("""**해석 가이드**

- MH delta_MH 와 Bayes 평균의 부호·크기를 비교. 두 방법이 문항 5에서 가장 큰 신호를 잡아야.
- MH p < 0.05 vs P(|delta|>0.2) > 0.9 같은 베이지안 기준이 어떤 결론으로 이어지는지 살펴봅니다.
- 약한 DIF 문항 8 에서 두 방법이 일치/엇갈리는지가 흥미로운 관찰 지점입니다.

> 💡 **포인트**: 두 방법이 일반적으로 비슷한 결론이지만, 베이지안은 "확률" 형태로 직접 진술 가능.
"""))

cells.append(md("""## 13. 요약 (Summary)

본 노트북에서 다룬 내용:

1. **DIF 분석의 두 목적** — 검출 + 원인·맥락 해석.
2. **DIF 연구의 5단계 발전사** — Zumbo의 3세대 프레임워크로 매핑.
3. **베이지안 검출의 6대 장점** — 소표본 안정성, 불확실성 정량화, 다중검정 완화, 사전정보 통합, 모형 확장 용이, anchor-free.
4. **편향(bias) vs 영향(impact)** — 4가지 시나리오와 통제(control) 개념.
5. **ICC 시각화** — DIF의 시각적 본질.
6. **첫 베이지안 1PL DIF 적합** — Stan/NumPyro, 사후·신용구간·ROPE.
7. **30+30 보조 시뮬레이션** — 표본 크기와 사후폭의 관계.
8. **MH와의 비교** — 표현력의 차이.

### 다음 노트북

- **Notebook 01** — 소표본·희소집단 반복 시뮬레이션.
- **Notebook 02** — 사후확률 기반 의사결정의 풍부함.
- **Notebook 03** — 위계 사전과 자동 shrinkage.
- **Notebook 04** — Spike-and-slab / Horseshoe prior, anchor-free.
- **부록 A** — 2PL로 non-uniform DIF 확장.
"""))


# ============================================================
# 노트북 메타데이터 & 저장
# ============================================================
notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3 (ipykernel)", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11",
                          "mimetype": "text/x-python",
                          "codemirror_mode": {"name": "ipython", "version": 3},
                          "pygments_lexer": "ipython3",
                          "nbconvert_exporter": "python",
                          "file_extension": ".py"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

NB_PATH.parent.mkdir(parents=True, exist_ok=True)
NB_PATH.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"Notebook saved: {NB_PATH}")
print(f"Total cells: {len(cells)}")
