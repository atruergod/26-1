// --
// sem_pcm_with_prior.stan
// ------------------------------------------------------------------
// PCM-SEM integrated model — version with prior information from original paper
//
// Original paper: Joo Wollang (2022), "A Study on Korean Writing Attitudes of International Students"
//   N=86, 21 items, Likert 5-point scale
//   X = writing recognition, M = writing response,
//   Y = performance attitude
//
// Identification conditions (same as sem_pcm_v2.stan):
//   [Scale] σ_M = σ_Y = 1 fixed
//   [Location] α_M, α_Y intercept + M·Y subscale sum-to-zero constraint
//
// Prior information (passed via data block → easy for sensitivity analysis):
//   - Item threshold centers: back-calculated from original paper's construct means
//   - Path coefficients: typical effect sizes from literature / reported correlations
//   - Gender effect: see original paper r(gender, X) = .290**
// ------------------------------------------------------------------

data {
  // ── Observed data ──────────────────────────────────────────
  int<lower=1> N;                            // sample size
  int<lower=1> I;                            // total number of items
  int<lower=1> K;                            // number of response categories
  int<lower=1, upper=I-1>   I_X;            // number of X construct items
  int<lower=1, upper=I-I_X> I_M;            // number of M construct items
  array[N, I] int<lower=1, upper=K> y;      // response matrix (N×I)
  vector[N] gender;                          // gender covariate (0/1)

  // ── Prior distributions for path coefficients (SEM) ────────
  real prior_b1_mu;           // beta1: X -> M  prior mean
  real<lower=0> prior_b1_sd;  // beta1 prior stddev
  real prior_b2_mu;           // beta2: M -> Y  prior mean
  real<lower=0> prior_b2_sd;
  real prior_g1_mu;           // gamma1: X -> Y (direct) prior mean
  real<lower=0> prior_g1_sd;

  // ── Prior distributions for intercepts ─────────────────────
  real prior_aM_mu;           // alpha_M prior mean
  real<lower=0> prior_aM_sd;
  real prior_aY_mu;           // alpha_Y prior mean
  real<lower=0> prior_aY_sd;

  // ── Prior distributions for gender effects ────────────────
  real prior_gM_mu;           // gamma_M: gender -> M
  real<lower=0> prior_gM_sd;
  real prior_gY_mu;           // gamma_Y: gender -> Y
  real<lower=0> prior_gY_sd;

  // ── Prior distributions for item thresholds ────────────────
  // Baseline threshold vector for each construct (K-1): back-calculated from original means
  vector[K-1] prior_delta_X;  // X construct item threshold prior center
  vector[K-1] prior_delta_M;  // M construct item threshold prior center
  vector[K-1] prior_delta_Y;  // Y construct item threshold prior center
  real<lower=0> prior_delta_sd;  // common prior stddev (item variability)

  // ── Prior-only sampling flag ──────────────────────────────
  // prior_only = 1 ignores likelihood → for prior predictive checks (PPC)
  int<lower=0, upper=1> prior_only;
}

transformed data {
  int I_Y = I - I_X - I_M;
}

parameters {
  array[I] vector[K-1] delta_raw;   // 비제약 문항 임계값
  array[N] vector[3]   theta;       // 개인 잠재 변수 (θ_X, θ_M, θ_Y)

  real b1;       // beta1 : X -> M
  real b2;       // beta2 : M -> Y
  real g1;       // gamma1 : X -> Y direct effect
  real gamma_M;  // gender -> M
  real gamma_Y;  // gender -> Y
  real alpha_M;  // M intercept
  real alpha_Y;  // Y intercept
}

transformed parameters {
  // Sum-to-zero constraint: location identification for M·Y construct thresholds
  array[I] vector[K-1] delta;

  // X construct: no constraint (X ~ N(0,1) prior identifies both scale and location)
  for (i in 1:I_X)
    delta[i] = delta_raw[i];

  // M construct: apply sum-to-zero constraint
  {
    real off = 0.0;
    for (i in (I_X+1):(I_X+I_M))
      off += sum(delta_raw[i]);
    off /= (I_M * (K-1));
    for (i in (I_X+1):(I_X+I_M))
      delta[i] = delta_raw[i] - off;
  }

  // Y construct: apply sum-to-zero constraint
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
  // ── Prior: path coefficients ──────────────────────────────
  b1      ~ normal(prior_b1_mu, prior_b1_sd);
  b2      ~ normal(prior_b2_mu, prior_b2_sd);
  g1      ~ normal(prior_g1_mu, prior_g1_sd);
  gamma_M ~ normal(prior_gM_mu, prior_gM_sd);
  gamma_Y ~ normal(prior_gY_mu, prior_gY_sd);
  alpha_M ~ normal(prior_aM_mu, prior_aM_sd);
  alpha_Y ~ normal(prior_aY_mu, prior_aY_sd);

  // ── Prior: item thresholds ────────────────────────────────
  // Allow item variability around construct-specific center ±prior_delta_sd
  for (i in 1:I_X)
    delta_raw[i] ~ normal(prior_delta_X, prior_delta_sd);
  for (i in (I_X+1):(I_X+I_M))
    delta_raw[i] ~ normal(prior_delta_M, prior_delta_sd);
  for (i in (I_X+I_M+1):I)
    delta_raw[i] ~ normal(prior_delta_Y, prior_delta_sd);

  // ── Latent variable structural model ──────────────────────
  for (n in 1:N) {
    // X construct: standard normal prior (scale/location fixed)
    theta[n][1] ~ normal(0, 1);
    // M construct: predicted by SEM (σ_M = 1 fixed)
    theta[n][2] ~ normal(alpha_M + b1*theta[n][1] + gamma_M*gender[n], 1.0);
    // Y construct: predicted by SEM (σ_Y = 1 fixed)
    theta[n][3] ~ normal(alpha_Y + g1*theta[n][1] + b2*theta[n][2] + gamma_Y*gender[n], 1.0);
  }

  // ── PCM measurement model (likelihood) ───────────────────
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
  real prop_mediated   = (abs(total_effect) > 1e-10)
                         ? indirect_effect / total_effect
                         : not_a_number();

  // Causal direction: posterior probability of positive indirect effect is calculated outside MCMC
  // P(beta1>0 and beta2>0 | data) = mean(b1>0 and b2>0 across draws)
}
