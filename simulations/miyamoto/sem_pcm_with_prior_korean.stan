//
// sem_pcm_with_prior.stan
// ══════════════════════════════════════════════════════════════════
// PCM-SEM 통합 모형 — 원논문 사전 정보 반영 버전
//
// 원논문: 주월랑(2022), 「외국인 유학생의 한국어 쓰기 태도 연구」
//   N=86, 21문항 Likert 5점 척도
//   X = 쓰기인식(writing recognition), M = 쓰기반응(writing response),
//   Y = 수행태도(performance attitude)
//
// 식별 조건 (sem_pcm_v2.stan 과 동일):
//   [척도] σ_M = σ_Y = 1 고정
//   [위치] α_M, α_Y 절편 + M·Y 하위 척도 합-영 제약
//
// 사전 정보 (data 블록으로 전달 → 민감도 분석 용이):
//   - 문항 임계값 중심: 원논문 구인별 평균에서 역산
//   - 경로 계수: 관련 문헌의 전형적 효과 크기 / 보고된 상관
//   - 성별 효과: 원논문 r(성별, X) = .290** 참고
// ══════════════════════════════════════════════════════════════════

data {
  // ── 관측 데이터 ─────────────────────────────────────────────
  int<lower=1> N;                            // 표본 크기
  int<lower=1> I;                            // 전체 문항 수
  int<lower=1> K;                            // 응답 범주 수
  int<lower=1, upper=I-1>   I_X;            // X 구인 문항 수
  int<lower=1, upper=I-I_X> I_M;            // M 구인 문항 수
  array[N, I] int<lower=1, upper=K> y;      // 응답 행렬 (N×I)
  vector[N] gender;                          // 성별 공변량 (0/1)

  // ── 경로 계수 사전 분포 (구조 방정식) ───────────────────────
  real prior_b1_mu;           // β₁: X → M  사전 평균
  real<lower=0> prior_b1_sd;  // β₁ 사전 표준편차
  real prior_b2_mu;           // β₂: M → Y  사전 평균
  real<lower=0> prior_b2_sd;
  real prior_g1_mu;           // γ₁: X → Y (직접)  사전 평균
  real<lower=0> prior_g1_sd;

  // ── 절편 사전 분포 ──────────────────────────────────────────
  real prior_aM_mu;           // α_M 사전 평균
  real<lower=0> prior_aM_sd;
  real prior_aY_mu;           // α_Y 사전 평균
  real<lower=0> prior_aY_sd;

  // ── 성별 효과 사전 분포 ─────────────────────────────────────
  real prior_gM_mu;           // γ_M: 성별 → M
  real<lower=0> prior_gM_sd;
  real prior_gY_mu;           // γ_Y: 성별 → Y
  real<lower=0> prior_gY_sd;

  // ── 문항 임계값 사전 분포 ───────────────────────────────────
  // 구인별 베이스라인 임계값 벡터 (K-1 개): 원논문 평균에서 역산
  vector[K-1] prior_delta_X;  // X 구인 문항 임계값 사전 중심
  vector[K-1] prior_delta_M;  // M 구인 문항 임계값 사전 중심
  vector[K-1] prior_delta_Y;  // Y 구인 문항 임계값 사전 중심
  real<lower=0> prior_delta_sd;  // 공통 사전 표준편차 (문항 간 변이)

  // ── 사전 분포만 샘플링 플래그 ───────────────────────────────
  // prior_only = 1 이면 우도 무시 → 사전 예측 점검(PPC)에 활용
  int<lower=0, upper=1> prior_only;
}

transformed data {
  int I_Y = I - I_X - I_M;
}

parameters {
  array[I] vector[K-1] delta_raw;   // 비제약 문항 임계값
  array[N] vector[3]   theta;       // 개인 잠재 변수 (θ_X, θ_M, θ_Y)

  real b1;       // β₁ : X → M
  real b2;       // β₂ : M → Y
  real g1;       // γ₁ : X → Y 직접 효과
  real gamma_M;  // 성별 → M
  real gamma_Y;  // 성별 → Y
  real alpha_M;  // M 절편
  real alpha_Y;  // Y 절편
}

transformed parameters {
  // 합-영 제약: M·Y 구인 임계값의 위치 식별
  array[I] vector[K-1] delta;

  // X 구인: 제약 없음 (X ~ N(0,1) 사전 분포로 척도·위치 모두 식별)
  for (i in 1:I_X)
    delta[i] = delta_raw[i];

  // M 구인: 합-영 제약 적용
  {
    real off = 0.0;
    for (i in (I_X+1):(I_X+I_M))
      off += sum(delta_raw[i]);
    off /= (I_M * (K-1));
    for (i in (I_X+1):(I_X+I_M))
      delta[i] = delta_raw[i] - off;
  }

  // Y 구인: 합-영 제약 적용
  {
    real off = 0.0;
    for (i in (I_X+I_M+1):I)
      off += sum(delta_raw[i]);
    off /= (I_Y * (K-1));
    for (i in (I_X+I_M+1):I)
      delta[i] = delta_raw[i] - off;
  }
}

model {
  // ── 사전 분포: 경로 계수 ───────────────────────────────────
  b1      ~ normal(prior_b1_mu, prior_b1_sd);
  b2      ~ normal(prior_b2_mu, prior_b2_sd);
  g1      ~ normal(prior_g1_mu, prior_g1_sd);
  gamma_M ~ normal(prior_gM_mu, prior_gM_sd);
  gamma_Y ~ normal(prior_gY_mu, prior_gY_sd);
  alpha_M ~ normal(prior_aM_mu, prior_aM_sd);
  alpha_Y ~ normal(prior_aY_mu, prior_aY_sd);

  // ── 사전 분포: 문항 임계값 ─────────────────────────────────
  // 구인별 중심 주변 ±prior_delta_sd 범위에서 문항 변이 허용
  for (i in 1:I_X)
    delta_raw[i] ~ normal(prior_delta_X, prior_delta_sd);
  for (i in (I_X+1):(I_X+I_M))
    delta_raw[i] ~ normal(prior_delta_M, prior_delta_sd);
  for (i in (I_X+I_M+1):I)
    delta_raw[i] ~ normal(prior_delta_Y, prior_delta_sd);

  // ── 잠재 변수 구조 모형 ────────────────────────────────────
  for (n in 1:N) {
    // X 구인: 표준 정규 사전 (척도·위치 고정)
    theta[n][1] ~ normal(0, 1);
    // M 구인: 구조 방정식에서 예측 (σ_M = 1 고정)
    theta[n][2] ~ normal(alpha_M + b1*theta[n][1] + gamma_M*gender[n], 1.0);
    // Y 구인: 구조 방정식에서 예측 (σ_Y = 1 고정)
    theta[n][3] ~ normal(alpha_Y + g1*theta[n][1] + b2*theta[n][2] + gamma_Y*gender[n], 1.0);
  }

  // ── PCM 측정 모형 (우도) ────────────────────────────────────
  if (!prior_only) {
    for (n in 1:N) {
      for (i in 1:I) {
        int latent_idx;
        vector[K] log_probs;

        if      (i <= I_X)        latent_idx = 1;
        else if (i <= I_X + I_M)  latent_idx = 2;
        else                      latent_idx = 3;

        log_probs[1] = 0;
        for (k in 2:K)
          log_probs[k] = log_probs[k-1] + (theta[n][latent_idx] - delta[i][k-1]);

        y[n,i] ~ categorical_logit(log_probs);
      }
    }
  }
}

generated quantities {
  real indirect_effect = b1 * b2;
  real total_effect    = g1 + indirect_effect;
  real prop_mediated   = (fabs(total_effect) > 1e-10)
                         ? indirect_effect / total_effect
                         : not_a_number();

  // 인과 방향성: 간접 효과가 양수일 사후 확률은 MCMC 외부에서 계산
  // P(β₁>0 ∧ β₂>0 | data) = mean(b1>0 and b2>0 across draws)
}
