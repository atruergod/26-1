"""NumPyro 2PL + uniform / non-uniform DIF."""
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist


def model(N, J, K, ii, jj, gg, y):
    ii0 = ii - 1
    jj0 = jj - 1
    theta = numpyro.sample("theta", dist.Normal(0.0, 1.0).expand([N]))
    b = numpyro.sample("b", dist.Normal(0.0, 2.0).expand([J]))
    a = numpyro.sample("a", dist.LogNormal(0.0, 0.4).expand([J]))
    delta_b = numpyro.sample("delta_b", dist.Normal(0.0, 0.5).expand([J]))
    log_delta_a = numpyro.sample("log_delta_a",
                                 dist.Normal(0.0, 0.3).expand([J]))
    mu_focal = numpyro.sample("mu_focal", dist.Normal(0.0, 1.0))

    th = theta[ii0] + gg * mu_focal
    a_eff = a[jj0] * jnp.exp(gg * log_delta_a[jj0])
    b_eff = b[jj0] + gg * delta_b[jj0]
    logits = a_eff * (th - b_eff)
    numpyro.sample("y", dist.Bernoulli(logits=logits), obs=y)

    numpyro.deterministic("delta_a_ratio", jnp.exp(log_delta_a))
