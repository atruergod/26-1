"""NumPyro 1PL Rasch + uniform DIF (non-hierarchical)."""
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist


def model(N, J, K, ii, jj, gg, y, prior_sigma_delta=1.0):
    ii0 = ii - 1
    jj0 = jj - 1
    theta = numpyro.sample("theta", dist.Normal(0.0, 1.0).expand([N]))
    b = numpyro.sample("b", dist.Normal(0.0, 2.0).expand([J]))
    delta = numpyro.sample("delta",
                           dist.Normal(0.0, prior_sigma_delta).expand([J]))
    mu_focal = numpyro.sample("mu_focal", dist.Normal(0.0, 1.0))

    th = theta[ii0] + gg * mu_focal
    b_eff = b[jj0] + gg * delta[jj0]
    logits = th - b_eff
    numpyro.sample("y", dist.Bernoulli(logits=logits), obs=y)
