"""NumPyro 1PL Rasch (DIF 없음 가정)."""
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist


def model(N, J, K, ii, jj, y):
    # 0-based index 변환
    ii0 = ii - 1
    jj0 = jj - 1
    theta = numpyro.sample("theta", dist.Normal(0.0, 1.0).expand([N]))
    b = numpyro.sample("b", dist.Normal(0.0, 2.0).expand([J]))
    logits = theta[ii0] - b[jj0]
    numpyro.sample("y", dist.Bernoulli(logits=logits), obs=y)
