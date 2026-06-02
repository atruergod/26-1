// rasch_spike_slab.stan
// 1PL Rasch + continuous relaxation of spike-and-slab prior.
//
// Instead of an explicit discrete indicator z_j in {0,1},
//   delta_j ~ pi * N(0, slab_sd^2) + (1-pi) * N(0, spike_sd^2)
// is implemented as a mixture of two normals (spike_sd << slab_sd).
// HMC handles this automatically via log_mix.

data {
  int<lower=1> N;
  int<lower=1> J;
  int<lower=1> K;
  array[K] int<lower=1, upper=N> ii;
  array[K] int<lower=1, upper=J> jj;
  array[K] int<lower=0, upper=1> gg;
  array[K] int<lower=0, upper=1> y;

  real<lower=0> slab_sd;            // slab SD (the "large" component)
  real<lower=0, upper=1> prior_inclusion;  // prior mean of pi_j (Beta)
}

transformed data {
  real spike_sd = slab_sd / 50.0;   // very narrow spike
}

parameters {
  vector[N] theta;
  vector[J] b;
  vector[J] delta;
  vector<lower=0, upper=1>[J] pi_j;   // item-wise inclusion probability
  real mu_focal;
}

model {
  theta ~ normal(0, 1);
  b ~ normal(0, 2);
  mu_focal ~ normal(0, 1);

  // Weak Beta prior on pi_j; concentration 5 -> Beta(inclusion*5, (1-inclusion)*5)
  for (j in 1:J) {
    pi_j[j] ~ beta(prior_inclusion * 5.0,
                   (1.0 - prior_inclusion) * 5.0);
    // Mixture log-density for delta_j
    target += log_mix(pi_j[j],
                      normal_lpdf(delta[j] | 0, slab_sd),
                      normal_lpdf(delta[j] | 0, spike_sd));
  }

  vector[K] eta;
  for (k in 1:K) {
    real th = theta[ii[k]] + gg[k] * mu_focal;
    eta[k] = th - (b[jj[k]] + gg[k] * delta[jj[k]]);
  }
  y ~ bernoulli_logit(eta);
}

generated quantities {
  // Posterior approximate inclusion probability
  vector[J] post_inclusion;
  for (j in 1:J) {
    real lp_slab  = log(pi_j[j])      + normal_lpdf(delta[j] | 0, slab_sd);
    real lp_spike = log1m(pi_j[j])    + normal_lpdf(delta[j] | 0, spike_sd);
    post_inclusion[j] = exp(lp_slab - log_sum_exp(lp_slab, lp_spike));
  }
}
