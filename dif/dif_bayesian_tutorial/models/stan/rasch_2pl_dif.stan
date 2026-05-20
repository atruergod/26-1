// rasch_2pl_dif.stan
// 2PL + simultaneous estimation of uniform (delta_b) and non-uniform
// (delta_a, log-multiplicative) DIF.
//   a_eff = a_j * exp(g_i * log_delta_a_j)
//   b_eff = b_j + g_i * delta_b_j
//   logit(P) = a_eff * (theta_i - b_eff)

data {
  int<lower=1> N;
  int<lower=1> J;
  int<lower=1> K;
  array[K] int<lower=1, upper=N> ii;
  array[K] int<lower=1, upper=J> jj;
  array[K] int<lower=0, upper=1> gg;
  array[K] int<lower=0, upper=1> y;
}

parameters {
  vector[N] theta;
  vector[J] b;
  vector<lower=0>[J] a;         // discrimination, positive
  vector[J] delta_b;            // uniform DIF
  vector[J] log_delta_a;        // log of multiplicative non-uniform DIF
  real mu_focal;
}

model {
  theta ~ normal(0, 1);
  b ~ normal(0, 2);
  a ~ lognormal(0, 0.4);           // a prior centered near 1
  delta_b ~ normal(0, 0.5);        // weakly-informative
  log_delta_a ~ normal(0, 0.3);    // weakly-informative
  mu_focal ~ normal(0, 1);

  vector[K] eta;
  for (k in 1:K) {
    real th = theta[ii[k]] + gg[k] * mu_focal;
    real a_eff = a[jj[k]] * exp(gg[k] * log_delta_a[jj[k]]);
    real b_eff = b[jj[k]] + gg[k] * delta_b[jj[k]];
    eta[k] = a_eff * (th - b_eff);
  }
  y ~ bernoulli_logit(eta);
}

generated quantities {
  // Convenience: ratio form of delta_a (focal / reference)
  vector[J] delta_a_ratio;
  for (j in 1:J) delta_a_ratio[j] = exp(log_delta_a[j]);
}
