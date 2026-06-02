// rasch_horseshoe.stan
// 1PL Rasch + horseshoe prior (Carvalho, Polson, Scott 2010).
//   delta_j = tau * lambda_j * delta_raw_j
//   lambda_j ~ half-Cauchy(0, 1)         (local shrinkage)
//   tau     ~ half-Cauchy(0, 0.3)        (global shrinkage)
// Horseshoe is a continuous alternative to spike-and-slab:
// large effects pass through almost unchanged, small effects are aggressively
// shrunk toward zero.

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
  vector[J] delta_raw;
  vector<lower=0>[J] lambda;
  real<lower=0> tau;
  real mu_focal;
}

transformed parameters {
  vector[J] delta = tau * lambda .* delta_raw;
}

model {
  theta ~ normal(0, 1);
  b ~ normal(0, 2);
  mu_focal ~ normal(0, 1);

  delta_raw ~ normal(0, 1);
  lambda ~ cauchy(0, 1);      // half-Cauchy (positivity by sampler)
  tau ~ cauchy(0, 0.3);       // global scale, strong prior mass near 0

  vector[K] eta;
  for (k in 1:K) {
    real th = theta[ii[k]] + gg[k] * mu_focal;
    eta[k] = th - (b[jj[k]] + gg[k] * delta[jj[k]]);
  }
  y ~ bernoulli_logit(eta);
}

generated quantities {
  // Shrinkage factor kappa_j = 1 / (1 + lambda_j^2 * tau^2)
  // kappa ~= 1 -> strong shrinkage (no DIF), kappa ~= 0 -> no shrinkage (DIF)
  vector[J] shrinkage_factor;
  for (j in 1:J) {
    shrinkage_factor[j] = 1.0 / (1.0 + (lambda[j] * tau)^2);
  }
}
