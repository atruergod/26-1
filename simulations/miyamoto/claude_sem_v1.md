# PCM-SEM 통합 모형: 학문 목적 한국어 학습자의 쓰기 태도 연구

**기반 논문**: 주월랑(2022), 「학문 목적 한국어 학습자의 한국어 쓰기 태도 연구」  
**모형 버전**: sem_pcm_v2 (베이지안 PCM-SEM, 완전 식별)

---

## 1. 원논문 핵심 요약

### 1.1 연구 개요

- **대상**: 한국 대학 재학 외국인 유학생 86명 (13개국)
- **측정 도구**: Likert 5점 척도 21문항 — 쓰기인식(4문항), 쓰기반응(11문항), 수행태도(6문항)
- **신뢰도**: Cronbach α = .854 / **타당도**: KMO = .756, Bartlett p < .001

### 1.2 원논문의 주요 발견

| 쓰기 태도 하위 범주 | 전체 평균(M) | 표준편차(SD) |
|---|---|---|
| 쓰기인식 (필요성·가치 인식) | 4.015 | 0.642 |
| 쓰기반응 (흥미·자신감) | 3.348 | 0.539 |
| 수행태도 (전략적 행동) | 3.171 | 0.585 |
| **쓰기 태도 전체** | **3.511** | **0.469** |

- 학년, 성별, TOPIK 급수, 어학원 경험 등 **어떤 학습자 요인도 쓰기 태도와 유의미한 차이를 만들지 못함**
- 한국어 숙달도(TOPIK)와 쓰기 태도 간 상관관계 없음 → 인지적 능력과 정의적 태도는 독립적으로 작동

### 1.3 원논문의 방법론적 한계 (본 연구가 극복하는 지점)

1. Likert 척도를 연속형 변수로 취급 (PCM으로 해결)
2. 하위 범주 간 인과 메커니즘 미검증 (SEM으로 해결)
3. 측정 오차 미통제 (잠재 변수 모형으로 해결)
4. 소표본·비균형 집단에서의 낮은 검정력 (베이지안 추론으로 해결)

---

## 2. 연구 모형 설계

### 2.1 부분 매개 모형 (Partial Mediation Model)

$$X \;\xrightarrow{\beta_1}\; M \;\xrightarrow{\beta_2}\; Y$$
$$X \;\xrightarrow{\gamma_1}\; Y \quad \text{(직접 효과)}$$

| 기호 | 의미 |
|---|---|
| $X$ | 쓰기인식 (Writing Awareness) — 외생 잠재 변수 |
| $M$ | 쓰기반응 (Writing Reaction) — 매개 잠재 변수 |
| $Y$ | 수행태도 (Performance Attitude) — 종속 잠재 변수 |
| $G$ | 성별 (Gender) — 관측 통제 변인 |

### 2.2 가설 경로

- **H1** (X → M): 한국어 쓰기의 필요성과 가치를 높게 인식할수록 쓰기에 대한 흥미와 자신감이 높아진다.
- **H2** (M → Y): 쓰기에 대한 자신감과 긍정적 흥미가 높을수록 실제 글쓰기의 전략적 태도가 적극적으로 나타난다.
- **H3** (X → Y): 쓰기인식은 수행태도에 직접적인 영향을 미친다.

### 2.3 잠재 변수와 측정 문항 매핑

| 잠재 변수 | 측정 문항 | 문항 번호 | 문항 수 |
|---|---|---|---|
| **X (쓰기인식)** | 쓰기의 필요성 및 자아 표현 수단으로서의 가치 | y₁ ~ y₄ | 4 |
| **M (쓰기반응)** | 쓰기에 대한 재미, 난이도 지각, 자신감, 학습 의지 | y₅ ~ y₁₅ | 11 |
| **Y (수행태도)** | 주제 선정, 독자 고려, 조직화, 고쳐쓰기 전략 | y₁₆ ~ y₂₁ | 6 |

---

## 3. 수학적 모형 명세

### 3.1 측정 방정식 — PCM (Partial Credit Model)

잠재 변수 $\theta$에 대한 문항 $i$의 범주 $k$ 응답 확률:

$$\log\!\left(\frac{P(y_{ni}=k)}{P(y_{ni}=k-1)}\right) = \theta_n - \delta_{ik}$$

$$P(y_{ni}=k \mid \theta_n) = \frac{\exp\!\left(k\,\theta_n - \sum_{j=1}^{k}\delta_{ij}\right)}{\sum_{m=0}^{K-1}\exp\!\left(m\,\theta_n - \sum_{j=1}^{m}\delta_{ij}\right)}$$

- $\theta_n$: $n$번째 학생의 해당 잠재 변수 값
- $\delta_{ik}$: 문항 $i$의 $k-1$점과 $k$점 사이의 단계 난이도 (Step Difficulty)
- $K=5$ (Likert 5점) → 문항당 $K-1=4$개의 $\delta$ 파라미터

> **PCM 선택 이유**: Rasch 계열 모형으로 문항 변별도(discrimination)가 1로 고정되어 잠재 변수 척도가 log-odds 단위로 자동 식별됨. 순서형 응답의 범주 간 간격이 동일하지 않아도 됨.

### 3.2 구조 방정식 — 절편 포함, 잔차 분산 고정

$$M_n = \alpha_M + \beta_1 X_n + \gamma_M G_n + \zeta_{M,n}, \qquad \zeta_{M,n} \sim \mathcal{N}(0,\,1)$$

$$Y_n = \alpha_Y + \gamma_1 X_n + \beta_2 M_n + \gamma_Y G_n + \zeta_{Y,n}, \qquad \zeta_{Y,n} \sim \mathcal{N}(0,\,1)$$

| 파라미터 | 설명 |
|---|---|
| $\alpha_M$ | M의 절편 — X=0, G=0일 때 M의 기대 수준 |
| $\alpha_Y$ | Y의 절편 — X=0, M=0, G=0일 때 Y의 기대 수준 |
| $\beta_1$ | X → M 경로 계수 |
| $\beta_2$ | M → Y 경로 계수 |
| $\gamma_1$ | X → Y 직접 효과 |
| $\gamma_M$ | 성별 → M 효과 |
| $\gamma_Y$ | 성별 → Y 효과 |

### 3.3 매개 효과 분해

$$\text{간접 효과} = \beta_1 \times \beta_2$$
$$\text{총 효과} = \gamma_1 + \beta_1\beta_2$$
$$\text{매개 비율} = \frac{\beta_1\beta_2}{\gamma_1 + \beta_1\beta_2}$$

### 3.4 식별 조건 (Identification Constraints)

완전 식별을 위해 세 가지 제약을 동시 적용합니다.

| 제약 종류 | 적용 대상 | 수학적 표현 | 해결하는 문제 |
|---|---|---|---|
| **① N(0,1) 사전 분포** | X (외생 잠재 변수) | $\theta_n^{(X)} \sim \mathcal{N}(0,1)$ | X의 척도·위치 |
| **② 잔차 분산 고정** | M, Y (내생 잠재 변수) | $\sigma_M = \sigma_Y = 1$ | M·Y의 척도 |
| **③ 합-영 제약** | M·Y 하위 척도 임계값 | $\sum_{i \in S}\sum_k \delta_{ik} = 0$ | M·Y의 위치 |

> **합-영 제약의 필요성**: 임의의 상수 $c$에 대해 $\delta_{ik} \to \delta_{ik}+c$ 변환이 PCM 우도를 불변으로 만들므로, $\alpha_M$과 $\delta^{(M)}$의 평균이 공통 이동 $c$에 대해 비식별됩니다. 합-영 제약 $\sum \delta = 0$ 하에서는 $c=0$만 허용되어 완전 식별됩니다.

---

## 4. 파라미터 전체 목록

| 구분 | 파라미터 | 개수 | 설명 |
|---|---|---|---|
| PCM 임계값 | $\delta_{ik}$ | $21 \times 4 = \mathbf{84}$ | 21문항 × 4단계 |
| 학생 잠재 변수 | $\theta_n = (X_n, M_n, Y_n)$ | $86 \times 3 = \mathbf{258}$ | 86명 × 3개 잠재 변수 |
| 경로 계수 | $\beta_1, \beta_2, \gamma_1$ | **3** | 구조 경로 |
| 성별 효과 | $\gamma_M, \gamma_Y$ | **2** | 통제 변인 |
| 절편 | $\alpha_M, \alpha_Y$ | **2** | 위치 식별 |
| **합계** | | **349** | |

> N=86의 소표본에서 파라미터 수 349는 베이지안 추론의 사전 분포(prior)를 통해 과적합 없이 추정 가능합니다. 빈도주의 ML 추정은 이 경우 수렴 실패 위험이 있습니다.

---

## 5. 데이터 요구사항

| 목표 수준 | 표본 크기 | 가능한 분석 |
|---|---|---|
| 최소 (기존 데이터) | N ≥ 86 | 단일 집단 PCM-SEM, 불확실성 넓음 |
| 안정적 | N ≥ 200 | 다집단 분석 (어권별 등) |
| 고도화 | N ≥ 300 | 혼합 모형, 복잡한 계층 구조 |

**데이터 형식**: N × 22 행렬 (문항 y1~y21 + gender)

```
y1  y2  y3  ... y21  gender
 4   3   5  ...   4       1
 3   4   4  ...   3       0
...
```

---

## 6. Stan 구현 코드 (sem_pcm_v2.stan)

```stan
// sem_pcm_v2.stan
// PCM-SEM 통합 모형 — 완전 식별 버전
// [척도 식별] σ_M = σ_Y = 1 고정
// [위치 식별] α_M, α_Y 명시적 절편
//            + M·Y 하위 척도 합-영(sum-to-zero) 제약

data {
  int<lower=1> N;                          // 학생 수 (예: 86)
  int<lower=1> I;                          // 전체 문항 수 (21)
  int<lower=1> K;                          // 응답 범주 수 (5)
  int<lower=1, upper=I-1>   I_X;          // 인식 하위 척도 문항 수 (4)
  int<lower=1, upper=I-I_X> I_M;          // 반응 하위 척도 문항 수 (11)
  // I_Y = I - I_X - I_M = 6: transformed data에서 자동 계산

  array[N, I] int<lower=1, upper=K> y;    // 응답 행렬 (N × I)
  vector[N] gender;                        // 성별: 남=0, 여=1
}

transformed data {
  int I_Y = I - I_X - I_M;               // 수행 하위 척도 문항 수 (6)
}

parameters {
  // 1. PCM 문항 임계값 (raw; 합-영 제약 전)
  array[I] vector[K-1] delta_raw;

  // 2. 학생별 잠재 변수 벡터 (X=인식, M=반응, Y=수행)
  array[N] vector[3] theta;

  // 3. 구조 경로 계수
  real b1;       // β₁ : X → M
  real b2;       // β₂ : M → Y
  real g1;       // γ₁ : X → Y (직접 효과)

  // 4. 성별 효과
  real gamma_M;  // γ_M : G → M
  real gamma_Y;  // γ_Y : G → Y

  // 5. 구조 방정식 절편 (위치 식별)
  real alpha_M;  // M 절편: X=0, G=0일 때 M의 기대 수준
  real alpha_Y;  // Y 절편: X=0, M=0, G=0일 때 Y의 기대 수준
}

transformed parameters {
  // M, Y 하위 척도: 합-영 제약 적용
  // 제약: Σ_{i∈subscale} Σ_k δ_{ik} = 0
  array[I] vector[K-1] delta;

  // X 하위 척도 (문항 1 ~ I_X): X ~ N(0,1)로 위치 고정, 제약 불필요
  for (i in 1:I_X)
    delta[i] = delta_raw[i];

  // M 하위 척도 (문항 I_X+1 ~ I_X+I_M): 합-영 제약
  {
    real M_offset = 0.0;
    for (i in (I_X + 1):(I_X + I_M))
      M_offset += sum(delta_raw[i]);
    M_offset /= (I_M * (K - 1));

    for (i in (I_X + 1):(I_X + I_M))
      delta[i] = delta_raw[i] - M_offset;
  }

  // Y 하위 척도 (문항 I_X+I_M+1 ~ I): 합-영 제약
  {
    real Y_offset = 0.0;
    for (i in (I_X + I_M + 1):I)
      Y_offset += sum(delta_raw[i]);
    Y_offset /= (I_Y * (K - 1));

    for (i in (I_X + I_M + 1):I)
      delta[i] = delta_raw[i] - Y_offset;
  }
}

model {
  // ── 사전 분포 (Priors) ──────────────────────────────────────
  for (i in 1:I)
    delta_raw[i] ~ normal(0, 3);   // 약정보적: 임계값 범위 넓게 허용

  b1      ~ normal(0, 1);          // σ=1 단위의 표준화 경로 계수
  b2      ~ normal(0, 1);
  g1      ~ normal(0, 1);
  gamma_M ~ normal(0, 1);
  gamma_Y ~ normal(0, 1);

  alpha_M ~ normal(0, 1);          // 합-영 제약 후 M의 실질 수준
  alpha_Y ~ normal(0, 1);

  // ── 학생 루프 ───────────────────────────────────────────────
  for (n in 1:N) {

    // ── 구조 방정식 ─────────────────────────────────────────
    // X (외생): 척도·위치 완전 고정
    theta[n][1] ~ normal(0, 1);

    // M (매개): 절편 포함, σ_M = 1 고정
    theta[n][2] ~ normal(
        alpha_M + b1 * theta[n][1] + gamma_M * gender[n],
        1.0);

    // Y (종속): 절편 포함, σ_Y = 1 고정
    theta[n][3] ~ normal(
        alpha_Y + g1 * theta[n][1] + b2 * theta[n][2]
        + gamma_Y * gender[n],
        1.0);

    // ── PCM 측정 방정식 ─────────────────────────────────────
    for (i in 1:I) {
      int      latent_idx;    // 잠재 변수 인덱스 (1=X, 2=M, 3=Y)
      vector[K] log_probs;   // 누적 비정규화 로그 확률

      // 문항 번호 → 잠재 변수 매핑
      if      (i <= I_X)           latent_idx = 1;
      else if (i <= I_X + I_M)     latent_idx = 2;
      else                         latent_idx = 3;

      // PCM 로그 확률: log P*(y=k) = Σ_{j=1}^{k} (θ - δ_j)
      log_probs[1] = 0;
      for (k in 2:K)
        log_probs[k] = log_probs[k-1]
                     + (theta[n][latent_idx] - delta[i][k-1]);

      y[n, i] ~ categorical_logit(log_probs);
    }
  }
}

generated quantities {
  // 간접 효과: β₁ × β₂
  real indirect_effect = b1 * b2;

  // 총 효과: γ₁ + β₁β₂
  real total_effect = g1 + indirect_effect;

  // 매개 비율 (proportion mediated)
  real prop_mediated = (fabs(total_effect) > 1e-10)
                       ? indirect_effect / total_effect
                       : not_a_number();
}
```

---

## 7. Python 실행 스크립트 (run_sem.py)

```python
import numpy as np
import pandas as pd
import arviz as az
from cmdstanpy import CmdStanModel

# ── 1. 데이터 준비 ──────────────────────────────────────────────
# df: N행 × 22열 (문항 y1~y21 + gender)
y_matrix      = df[[f'y{i}' for i in range(1, 22)]].values.astype(int)
gender_vector = df['gender'].values.astype(float)   # 남=0, 여=1

stan_data = {
    'N'      : len(df),
    'I'      : 21,
    'K'      : 5,
    'I_X'    : 4,    # 쓰기인식 문항 수
    'I_M'    : 11,   # 쓰기반응 문항 수 (I_Y=6은 자동 계산)
    'y'      : y_matrix,
    'gender' : gender_vector,
}

# ── 2. 모델 컴파일 및 샘플링 ────────────────────────────────────
model = CmdStanModel(stan_file='sem_pcm_v2.stan')

fit = model.sample(
    data            = stan_data,
    iter_warmup     = 1000,
    iter_sampling   = 2000,
    chains          = 4,
    parallel_chains = 4,       # RTX 3090 병렬 활용
    seed            = 42,
    adapt_delta     = 0.95,    # 복잡한 기하구조 대응
    max_treedepth   = 12,
)

# ── 3. 수렴 진단 ─────────────────────────────────────────────────
idata = az.from_cmdstanpy(fit)

key_params = [
    'alpha_M', 'alpha_Y',
    'b1', 'b2', 'g1',
    'gamma_M', 'gamma_Y',
    'indirect_effect', 'total_effect', 'prop_mediated',
]
summary = az.summary(idata, var_names=key_params, round_to=3)
print(summary)
# 확인 기준: R-hat < 1.01, Bulk-ESS / Tail-ESS > 400

az.plot_energy(idata)        # BFMI > 0.2 권장
az.plot_trace(idata, var_names=['b1', 'b2', 'g1', 'alpha_M', 'alpha_Y'])

# ── 4. 합-영 제약 검증 ───────────────────────────────────────────
delta_samples = fit.stan_variable('delta')   # shape: (draws, 21, 4)

for (start, end), name in [((4, 15), 'M (반응)'), ((15, 21), 'Y (수행)')]:
    sums = delta_samples[:, start:end, :].sum(axis=(1, 2))
    print(f"{name} 하위 척도 임계값 합: "
          f"mean={sums.mean():.8f}, std={sums.std():.8f}")
    # → 수치 오차(~1e-10) 범위에서 0에 수렴해야 함

# ── 5. 핵심 결과 시각화 ──────────────────────────────────────────
# 경로 계수 및 절편 사후 분포
az.plot_posterior(
    idata,
    var_names=['alpha_M', 'alpha_Y', 'b1', 'b2', 'g1'],
    ref_val=0, hdi_prob=0.95
)

# 매개 효과 사후 분포
az.plot_posterior(
    idata,
    var_names=['indirect_effect', 'total_effect', 'prop_mediated'],
    ref_val=0, hdi_prob=0.95
)
```

---

## 8. 결과 해석 가이드

### 8.1 핵심 결과 해석 체계

| 결과 항목 | 보고 형식 | 해석 |
|---|---|---|
| 경로 계수 $\beta_1$ | 사후 평균 ± 95% HDI | X가 1단위 증가 시 M의 평균 변화 |
| 경로 계수 $\beta_2$ | 사후 평균 ± 95% HDI | M이 1단위 증가 시 Y의 평균 변화 |
| 직접 효과 $\gamma_1$ | 사후 평균 ± 95% HDI | X가 Y에 직접 미치는 영향 |
| 간접 효과 $\beta_1\beta_2$ | 사후 분포 전체 | 완전 매개 vs. 부분 매개 판단 |
| 매개 비율 | 사후 평균 + HDI | 총 효과 중 매개 경로가 차지하는 비율 |
| 절편 $\alpha_M, \alpha_Y$ | 사후 평균 ± 95% HDI | 통제 변인이 0일 때 M·Y의 기준 수준 |

### 8.2 기존 연구와의 비교 해석

- **기존 논문 결론**: "숙달도가 높아져도 쓰기 태도는 변화하지 않는다"
- **본 SEM 기반 해석 가능성**: "쓰기인식(X)이 수행태도(Y)에 직접 영향을 주지만, 쓰기반응(M, 자신감·흥미)을 거치는 간접 경로가 단절되어 있어 숙달도 향상이 최종 수행 태도 변화로 이어지지 않는다" → **병목 구간 식별 가능**

### 8.3 수렴 실패 시 대응 방안

| 증상 | 원인 | 조치 |
|---|---|---|
| R-hat > 1.01 | 체인 간 불일치 | chains 수 증가, iter_warmup 증가 |
| ESS < 400 | 강한 자기상관 | adapt_delta → 0.99, max_treedepth → 15 |
| Divergence 발생 | 후험 기하구조 불량 | 사전 분포 더 정보적으로 조정 |
| BFMI < 0.2 | Neal's Funnel 잔존 | 재중심화(non-centered parameterization) 검토 |

---

## 9. 연구 확장 방향

### 9.1 국적(어권) 변인 활용: 계층적 베이즈 모형

국적별 $n$이 작아도 **부분 풀링(partial pooling)**으로 안정적 추정 가능.

```stan
// 국적 랜덤 효과 추가 예시
parameters {
  array[J] vector[3] alpha_country;  // 국적별 잠재변수 오프셋
  vector<lower=0>[3] sigma_country;  // 국적 간 분산
}
model {
  sigma_country ~ exponential(2);
  for (j in 1:J)
    alpha_country[j] ~ normal(0, sigma_country);
  // theta[n][2] 에 alpha_country[country[n]][2] 추가
}
```

### 9.2 종단 연구: 상태공간 모형 (State-Space Model)

학기별 반복 측정 + 중도 탈락 학생을 `obs_mask`로 처리.

```stan
// 잠재 상태 전이 방정식
for (n in 1:N) {
  theta_longitudinal[n, 1] ~ normal(0, 1);
  for (t in 2:T)
    theta_longitudinal[n, t] ~ normal(
        phi * theta_longitudinal[n, t-1] + mu_trend, sigma_process);
  // obs_mask로 결측 처리: 관측된 시점만 우도 계산
}
```

### 9.3 생성형 AI 활용과 쓰기 태도: 베이즈 매개 모형

AI 활용 빈도를 추가 매개 변인으로 포함하여 쓰기 태도 → AI 활용 → 쓰기 성취도 경로 추정.

### 9.4 잠재 프로파일 분석: 학습자 유형 분류

베이즈 혼합 모형(Bayesian Mixture Model)으로 "의욕형", "회피형", "자신감결여형" 등 학습자 유형을 데이터 기반으로 탐색. LOO-CV로 최적 군집 수 결정.

---

## 10. 식별 조건 요약표

| 조건 | 수식 | Stan 구현 |
|---|---|---|
| X 위치 | $\theta_n^{(X)} \sim \mathcal{N}(0,1)$ | `theta[n][1] ~ normal(0, 1)` |
| X 척도 | (N(0,1) 사전 분포로 포함) | — |
| M 척도 | $\sigma_M = 1$ | `normal(..., 1.0)` |
| M 위치 | $\sum_{i \in M}\sum_k \delta_{ik} = 0$ | `transformed parameters` 블록의 `M_offset` |
| Y 척도 | $\sigma_Y = 1$ | `normal(..., 1.0)` |
| Y 위치 | $\sum_{i \in Y}\sum_k \delta_{ik} = 0$ | `transformed parameters` 블록의 `Y_offset` |

---

*작성일: 2026-04-22*  
*Stan 버전: 2.32+ (최신 배열 문법 적용)*  
*CmdStanPy 버전: 1.2.0+*
