// sem_pcm_with_prior.stan
// ------------------------------------------------------------------
// PCM-SEM integrated model -- version with prior information
//
// Original paper: Joo Wollang (2022),
//   "A Study on Korean Writing Attitudes of International Students"
//   N=86, 21 items, Likert 5-point scale
//   X = writing recognition (sseugi-insik)
//   M = writing response   (sseugi-baneung)
//   Y = performance attitude (suhaeng-taedo)
//
// Identification conditions (same as sem_pcm_v2.stan):
//   [Scale]    sigma_M = sigma_Y = 1 (fixed)
//   [Location] alpha_M, alpha_Y intercepts +
//              M/Y subscale sum-to-zero threshold constraint
//
// Prior information passed via data block (easy sensitivity analysis):
//   - Item threshold centers: back-calculated from paper composite means
//   - Path coefficients: literature typical values / reported correlations
//   - Gender effect: paper reports r(gender, X) = .290**
// ------------------------------------------------------------------

data {
  // --- Observed data ---
  int<lower=1> N;                            // sample size
  int<lower=1> I;                            // total number of items
  int<lower=1> K;                            // number of response categories
  int<lower=1, upper=I-1>   I_X;            // number of X construct items
  int<lower=1, upper=I-I_X> I_M;            // number of M construct items
  array[N, I] int<lower=1, upper=K> y;      // response matrix (N x I)
  vector[N] gender;                          // gender covariate (0/1)

  // --- Prior distributions: path coefficients (SEM) ---
  real prior_b1_mu;           // beta1: X -> M  prior mean
  real<lower=0> prior_b1_sd;  // beta1 prior stddev
  real prior_b2_mu;           // beta2: M -> Y  prior mean
  real<lower=0> prior_b2_sd;
  real prior_g1_mu;           // gamma1: X -> Y (direct) prior mean
  real<lower=0> prior_g1_sd;

  // --- Prior distributions: intercepts ---
  real prior_aM_mu;           // alpha_M prior mean
  real<lower=0> prior_aM_sd;
  real prior_aY_mu;           // alpha_Y prior mean
  real<lower=0> prior_aY_sd;

  // --- Prior distributions: gender effects ---
  real prior_gM_mu;           // gamma_M: gender -> M  prior mean
  real<lower=0> prior_gM_sd;
  real prior_gY_mu;           // gamma_Y: gender -> Y  prior mean
  real<lower=0> prior_gY_sd;

  // --- Prior distributions: item thresholds ---
  // Baseline threshold vector per construct (K-1 values each),
  // back-calculated from original paper composite score means.
  vector[K-1] prior_delta_X;  // X construct threshold prior center
  vector[K-1] prior_delta_M;  // M construct threshold prior center
  vector[K-1] prior_delta_Y;  // Y construct threshold prior center
  real<lower=0> prior_delta_sd;  // common prior SD (item-level variability)

  // --- Prior-only flag ---
  // Set prior_only = 1 to ignore likelihood (prior predictive check)
  int<lower=0, upper=1> prior_only;
}

transformed data {
  int I_Y = I - I_X - I_M;
}

parameters {
  array[I] vector[K-1] delta_raw;   // unconstrained item thresholds
  array[N] vector[3]   theta;       // person latent variables (X, M, Y)

  real b1;       // beta1  : X -> M
  real b2;       // beta2  : M -> Y
  real g1;       // gamma1 : X -> Y direct effect
  real gamma_M;  // gender -> M
  real gamma_Y;  // gender -> Y
  real alpha_M;  // M equation intercept
  real alpha_Y;  // Y equation intercept
}

transformed parameters {
  // Apply sum-to-zero constraint on M and Y thresholds (location identification)
  array[I] vector[K-1] delta;

  // X construct: no constraint (X ~ N(0,1) prior fixes scale and location)
  for (i in 1:I_X)
    delta[i] = delta_raw[i];

  // M construct: sum-to-zero constraint
  {
    real off = 0.0;
    for (i in (I_X+1):(I_X+I_M))
      off += sum(delta_raw[i]);
    off /= (I_M * (K-1));
    for (i in (I_X+1):(I_X+I_M))
      delta[i] = delta_raw[i] - off;
  }

  // Y construct: sum-to-zero constraint
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
  // --- Informative priors: path coefficients ---
  b1      ~ normal(prior_b1_mu, prior_b1_sd);
  b2      ~ normal(prior_b2_mu, prior_b2_sd);
  g1      ~ normal(prior_g1_mu, prior_g1_sd);
  gamma_M ~ normal(prior_gM_mu, prior_gM_sd);
  gamma_Y ~ normal(prior_gY_mu, prior_gY_sd);
  alpha_M ~ normal(prior_aM_mu, prior_aM_sd);
  alpha_Y ~ normal(prior_aY_mu, prior_aY_sd);

  // --- Informative priors: item thresholds ---
  // Each item's thresholds drawn near its construct-level center
  for (i in 1:I_X)
    delta_raw[i] ~ normal(prior_delta_X, prior_delta_sd);
  for (i in (I_X+1):(I_X+I_M))
    delta_raw[i] ~ normal(prior_delta_M, prior_delta_sd);
  for (i in (I_X+I_M+1):I)
    delta_raw[i] ~ normal(prior_delta_Y, prior_delta_sd);

  // --- Structural model: latent variable equations ---
  for (n in 1:N) {
    // X: standard normal prior (scale and location fixed)
    theta[n][1] ~ normal(0, 1);
    // M: structural equation, residual SD = 1 fixed
    theta[n][2] ~ normal(alpha_M + b1*theta[n][1] + gamma_M*gender[n], 1.0);
    // Y: structural equation, residual SD = 1 fixed
    theta[n][3] ~ normal(alpha_Y + g1*theta[n][1] + b2*theta[n][2]
                         + gamma_Y*gender[n], 1.0);
  }

  // --- PCM measurement model (likelihood) ---
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
          log_probs[k] = log_probs[k-1]
                         + (theta[n][latent_idx] - delta[i][k-1]);

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
  // Note: P(b1>0 AND b2>0 | data) is computed externally from posterior draws
}
