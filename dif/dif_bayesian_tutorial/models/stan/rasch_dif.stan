// rasch_dif.stan
// 1PL Rasch + uniform DIF (non-hierarchical: all delta_j independent).
// Model:  logit(P(Y_ij = 1)) = theta_i + g_i*mu_focal - (b_j + g_i*delta_j)
//   g_i in {0=reference, 1=focal}

data {
  int<lower=1> N;
  int<lower=1> J;
  int<lower=1> K;
  array[K] int<lower=1, upper=N> ii;
  array[K] int<lower=1, upper=J> jj;
  array[K] int<lower=0, upper=1> gg;
  array[K] int<lower=0, upper=1> y;
  real<lower=0> prior_sigma_delta;   // prior SD for delta_j
}

parameters {
  vector[N] theta;
  vector[J] b;
  vector[J] delta;                   // uniform DIF effect
  real mu_focal;                     // group impact (focal mean shift)
}

model {
  theta ~ normal(0, 1);              // reference group baseline
  b ~ normal(0, 2);
  delta ~ normal(0, prior_sigma_delta);
  mu_focal ~ normal(0, 1);

  vector[K] eta;
  for (k in 1:K) {
    real th = theta[ii[k]] + gg[k] * mu_focal;
    eta[k] = th - (b[jj[k]] + gg[k] * delta[jj[k]]);
  }
  y ~ bernoulli_logit(eta);
}
