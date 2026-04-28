data {
  int<lower=1> N;
  int<lower=1> J;
  int<lower=1> Nobs;
  array[Nobs] int<lower=1,upper=N> nn;
  array[Nobs] int<lower=1,upper=J> jj;
  array[Nobs] int<lower=0,upper=1> y;
}
parameters {
  vector[N] theta;
  real<lower=0> sigma_theta;
  vector[J-1] beta_free;
}
transformed parameters {
  vector[J] beta;
  beta = append_row(beta_free, -sum(beta_free));
}
model {
  sigma_theta ~ exponential(1);
  theta ~ normal(0, sigma_theta);
  beta_free ~ normal(0, 2);
  for (obs in 1:Nobs)
    y[obs] ~ bernoulli_logit(theta[nn[obs]] - beta[jj[obs]]);
}
generated quantities {
  array[Nobs] int y_rep;
  vector[Nobs] log_lik;
  for (obs in 1:Nobs) {
    real lp = theta[nn[obs]] - beta[jj[obs]];
    y_rep[obs]   = bernoulli_logit_rng(lp);
    log_lik[obs] = bernoulli_logit_lpmf(y[obs] | lp);
  }
}
