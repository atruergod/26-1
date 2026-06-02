"""NumPyro 1PL Rasch + horseshoe prior."""
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist


def model(N, J, K, ii, jj, gg, y):
    ii0 = ii - 1
    jj0 = jj - 1
    theta = numpyro.sample("theta", dist.Normal(0.0, 1.0).expand([N]))
    b = numpyro.sample("b", dist.Normal(0.0, 2.0).expand([J]))
    mu_focal = numpyro.sample("mu_focal", dist.Normal(0.0, 1.0))

    tau = numpyro.sample("tau", dist.HalfCauchy(0.3))
    lam = numpyro.sample("lambda_", dist.HalfCauchy(1.0).expand([J]))
    delta_raw = numpyro.sample("delta_raw", dist.Normal(0.0, 1.0).expand([J]))
    delta = numpyro.deterministic("delta", tau * lam * delta_raw)

    # shrinkage factor
    shrinkage = numpyro.deterministic(
        "shrinkage_factor", 1.0 / (1.0 + (lam * tau) ** 2)
    )

    th = theta[ii0] + gg * mu_focal
    b_eff = b[jj0] + gg * delta[jj0]
    logits = th - b_eff
    numpyro.sample("y", dist.Bernoulli(logits=logits), obs=y)
