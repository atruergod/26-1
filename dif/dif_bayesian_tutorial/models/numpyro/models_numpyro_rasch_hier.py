"""NumPyro 1PL Rasch + hierarchical DIF prior."""
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist


def model(N, J, K, ii, jj, gg, y):
    ii0 = ii - 1
    jj0 = jj - 1
    theta = numpyro.sample("theta", dist.Normal(0.0, 1.0).expand([N]))
    b = numpyro.sample("b", dist.Normal(0.0, 2.0).expand([J]))
    mu_focal = numpyro.sample("mu_focal", dist.Normal(0.0, 1.0))

    tau = numpyro.sample("tau", dist.HalfNormal(0.5))
    delta_raw = numpyro.sample("delta_raw", dist.Normal(0.0, 1.0).expand([J]))
    delta = numpyro.deterministic("delta", tau * delta_raw)

    th = theta[ii0] + gg * mu_focal
    b_eff = b[jj0] + gg * delta[jj0]
    logits = th - b_eff
    numpyro.sample("y", dist.Bernoulli(logits=logits), obs=y)
