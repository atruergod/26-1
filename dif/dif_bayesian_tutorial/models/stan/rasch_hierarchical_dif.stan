// rasch_hierarchical_dif.stan
// 1PL Rasch + hierarchical prior on DIF:
//   delta_j ~ Normal(0, tau^2),  tau ~ half-Normal(0, 0.5)
// As tau shrinks toward 0, all delta_j shrink toward 0 (auto-shrinkage).

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
  vector[J] delta_raw;       // non-centered parameterization
  real<lower=0> tau;
  real mu_focal;
}

transformed parameters {
  vector[J] delta = tau * delta_raw;
}

model {
  theta ~ normal(0, 1);
  b ~ normal(0, 2);
  delta_raw ~ normal(0, 1);
  tau ~ normal(0, 0.5);          // half-Normal (lower=0 enforced by sampler)
  mu_focal ~ normal(0, 1);

  vector[K] eta;
  for (k in 1:K) {
    real th = theta[ii[k]] + gg[k] * mu_focal;
    eta[k] = th - (b[jj[k]] + gg[k] * delta[jj[k]]);
  }
  y ~ bernoulli_logit(eta);
}
