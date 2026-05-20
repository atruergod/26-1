"""NumPyro 1PL Rasch + continuous spike-and-slab prior."""
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist


def model(N, J, K, ii, jj, gg, y, slab_sd=1.0, prior_inclusion=0.2):
    ii0 = ii - 1
    jj0 = jj - 1
    spike_sd = slab_sd / 50.0

    theta = numpyro.sample("theta", dist.Normal(0.0, 1.0).expand([N]))
    b = numpyro.sample("b", dist.Normal(0.0, 2.0).expand([J]))
    mu_focal = numpyro.sample("mu_focal", dist.Normal(0.0, 1.0))

    pi_j = numpyro.sample(
        "pi_j",
        dist.Beta(prior_inclusion * 5.0,
                  (1.0 - prior_inclusion) * 5.0).expand([J]),
    )
    # δ 의 혼합 분포 (continuous mixture)
    mix = dist.MixtureSameFamily(
        dist.Categorical(probs=jnp.stack([pi_j, 1.0 - pi_j], axis=-1)),
        dist.Normal(jnp.zeros((J, 2)),
                    jnp.stack([jnp.full(J, slab_sd),
                               jnp.full(J, spike_sd)], axis=-1)),
    )
    delta = numpyro.sample("delta", mix)

    th = theta[ii0] + gg * mu_focal
    b_eff = b[jj0] + gg * delta[jj0]
    logits = th - b_eff
    numpyro.sample("y", dist.Bernoulli(logits=logits), obs=y)

    # 사후 inclusion 근사
    log_slab = jnp.log(pi_j) + dist.Normal(0.0, slab_sd).log_prob(delta)
    log_spike = jnp.log1p(-pi_j) + dist.Normal(0.0, spike_sd).log_prob(delta)
    post_inclusion = jnp.exp(log_slab - jnp.logaddexp(log_slab, log_spike))
    numpyro.deterministic("post_inclusion", post_inclusion)
