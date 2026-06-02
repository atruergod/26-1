// rasch_basic.stan
// 1PL Rasch model, no-DIF baseline.
// Identifiability: theta is softly anchored at mean 0 via the prior.

data {
  int<lower=1> N;                 // number of respondents
  int<lower=1> J;                 // number of items
  int<lower=1> K;                 // total responses (= N*J, long format)
  array[K] int<lower=1, upper=N> ii;
  array[K] int<lower=1, upper=J> jj;
  array[K] int<lower=0, upper=1> y;
}

parameters {
  vector[N] theta;                // ability
  vector[J] b;                    // item difficulty
}

model {
  // priors
  theta ~ normal(0, 1);
  b ~ normal(0, 2);

  // likelihood
  vector[K] eta;
  for (k in 1:K) {
    eta[k] = theta[ii[k]] - b[jj[k]];
  }
  y ~ bernoulli_logit(eta);
}
