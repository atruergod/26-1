"""
ss_fig_posterior.py
--------------------
Generates posterior distribution comparison figures for the two Bayesian
PCM-SEM models (weak vs. strong prior) estimated on the N=86 simulated dataset.

Outputs:
  ss_fig_posterior_paths.png  -- structural path coefficients (b1, b2, g1)
  ss_fig_posterior_mediation.png -- indirect effect + gender effects
  ss_fig_posterior_combined.png  -- 2x3 combined panel for paper

Usage:
  python ss_fig_posterior.py
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

## Removed DroidSansFallbackFull font loading (Linux-only, not needed on Windows)
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False
import matplotlib.patches as mpatches
import os

def gaussian_kde_np(samples, bw_method=0.25):
    """Minimal Gaussian KDE using numpy (no scipy needed)."""
    n = len(samples)
    h = bw_method * np.std(samples, ddof=1)
    def kde(x):
        x = np.asarray(x)
        result = np.zeros_like(x, dtype=float)
        for xi in samples:
            result += np.exp(-0.5 * ((x - xi) / h) ** 2)
        return result / (n * h * np.sqrt(2 * np.pi))
    return kde

def gaussian_kde(samples, bw_method=0.25):
    return gaussian_kde_np(samples, bw_method)

# ── paths ──────────────────────────────────────────────────────────────────
HERE = os.path.dirname(os.path.abspath(__file__))
WEAK_CSV   = os.path.join(HERE, "ss_mcmc_weakprior_N86.csv")
STRONG_CSV = os.path.join(HERE, "ss_mcmc_strongprior_N86.csv")

# ── colors ─────────────────────────────────────────────────────────────────
C_WEAK   = "#4878CF"   # blue
C_STRONG = "#E87722"   # orange
ALPHA    = 0.30
LW       = 2.0

# ── helpers ────────────────────────────────────────────────────────────────
def plot_posterior(ax, samples_weak, samples_strong,
                   param_label, xlo=None, xhi=None,
                   vline=0.0):
    """
    Overlay KDE densities for weak (blue) and strong (orange) posteriors.
    Shade 95% CrI.  Mark posterior mean with vertical dashed lines.
    """
    for samples, color, label in [
            (samples_weak,   C_WEAK,   "약한 사전"),
            (samples_strong, C_STRONG, "강한 사전")]:
        lo, hi = np.percentile(samples, [2.5, 97.5])
        mn = samples.mean()

        # KDE
        kde = gaussian_kde(samples, bw_method=0.25)
        xs = np.linspace(samples.min() - 0.1, samples.max() + 0.1, 200)
        ys = kde(xs)

        ax.plot(xs, ys, color=color, lw=LW, label=label)

        # shade 95% CrI
        mask = (xs >= lo) & (xs <= hi)
        ax.fill_between(xs[mask], ys[mask], alpha=ALPHA, color=color)

        # posterior mean tick
        ax.axvline(mn, color=color, lw=1.2, ls="--", alpha=0.8)

    # zero line
    if vline is not None:
        ax.axvline(vline, color="black", lw=0.8, ls=":", alpha=0.6)

    ax.set_title(param_label, fontsize=11, pad=4)
    ax.set_yticks([])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    if xlo is not None:
        ax.set_xlim(xlo, xhi)

# ── load data ──────────────────────────────────────────────────────────────
weak   = pd.read_csv(WEAK_CSV)
strong = pd.read_csv(STRONG_CSV)

# ── Figure 1: structural path coefficients (b1, b2, g1) + indirect ─────────
fig1, axes1 = plt.subplots(1, 4, figsize=(14, 3.6))
fig1.suptitle(
    "Bayesian PCM-SEM: Posterior Distributions (N=86)\n"
    "Weak Prior (blue) vs. Strong Prior (orange), shaded 95% CrI",
    fontsize=11, y=1.02
)

specs = [
    ("b1",              r"$\beta_1$: X$\rightarrow$M",           -0.3, 1.0),
    ("b2",              r"$\beta_2$: M$\rightarrow$Y",           -0.4, 0.8),
    ("g1",              r"$\gamma_1$: X$\rightarrow$Y (direct)", -0.6, 0.6),
    ("indirect_effect", r"Indirect ($\beta_1\beta_2$)",          -0.15, 0.35),
]
for ax, (col, lbl, xlo, xhi) in zip(axes1, specs):
    plot_posterior(ax, weak[col].values, strong[col].values, lbl, xlo, xhi)

# shared legend
handles = [
    mpatches.Patch(color=C_WEAK,   alpha=0.7, label="약한 사전 (Weak)"),
    mpatches.Patch(color=C_STRONG, alpha=0.7, label="강한 사전 (Strong)"),
]
fig1.legend(handles=handles, loc="lower center", ncol=2,
            bbox_to_anchor=(0.5, -0.08), fontsize=10, frameon=False)

fig1.tight_layout()
out1 = os.path.join(HERE, "ss_fig_posterior_paths.png")
fig1.savefig(out1, dpi=150, bbox_inches="tight")
print(f"Saved: {out1}")
plt.close(fig1)

# ── Figure 2: gender effects ──────────────────────────────────────────────
fig2, axes2 = plt.subplots(1, 2, figsize=(8, 3.6))
fig2.suptitle(
    "Gender Effects: Posterior Distributions (N=86)",
    fontsize=11, y=1.02
)
gender_specs = [
    ("gamma_M", r"$\gamma_M$: Gender$\rightarrow$M", -1.2, 1.0),
    ("gamma_Y", r"$\gamma_Y$: Gender$\rightarrow$Y", -1.2, 1.0),
]
for ax, (col, lbl, xlo, xhi) in zip(axes2, gender_specs):
    plot_posterior(ax, weak[col].values, strong[col].values, lbl, xlo, xhi)

fig2.legend(handles=handles, loc="lower center", ncol=2,
            bbox_to_anchor=(0.5, -0.10), fontsize=10, frameon=False)
fig2.tight_layout()
out2 = os.path.join(HERE, "ss_fig_posterior_gender.png")
fig2.savefig(out2, dpi=150, bbox_inches="tight")
print(f"Saved: {out2}")
plt.close(fig2)

# ── Figure 3: combined 2x3 panel for paper ────────────────────────────────
fig3, axes3 = plt.subplots(2, 3, figsize=(13, 7))
fig3.suptitle(
    "그림 1. 베이지안 PCM-SEM 사후 분포 비교 (N=86)\n"
    "파란색: 약한 사전 분포 / 주황색: 강한 사전 분포 (음영: 95% 신용 구간, 점선: 사후 평균)",
    fontsize=11, y=1.01
)

panel_specs = [
    ("b1",              r"(a) $\beta_1$: 쓰기인식$\rightarrow$쓰기반응",   -0.3, 1.0),
    ("b2",              r"(b) $\beta_2$: 쓰기반응$\rightarrow$수행태도",   -0.4, 0.8),
    ("g1",              r"(c) $\gamma_1$: 쓰기인식$\rightarrow$수행태도 (직접)", -0.6, 0.6),
    ("indirect_effect", r"(d) 간접 효과 ($\beta_1 \cdot \beta_2$)",         -0.15, 0.35),
    ("gamma_M",         r"(e) $\gamma_M$: 성별$\rightarrow$쓰기반응",       -1.2, 1.0),
    ("gamma_Y",         r"(f) $\gamma_Y$: 성별$\rightarrow$수행태도",       -1.2, 1.0),
]

for ax, (col, lbl, xlo, xhi) in zip(axes3.flat, panel_specs):
    plot_posterior(ax, weak[col].values, strong[col].values, lbl, xlo, xhi)
    ax.set_xlabel("Parameter value", fontsize=8)

fig3.legend(handles=handles, loc="lower center", ncol=2,
            bbox_to_anchor=(0.5, -0.03), fontsize=11, frameon=False)
fig3.tight_layout(rect=[0, 0.04, 1, 1])
out3 = os.path.join(HERE, "ss_fig_posterior_combined.png")
fig3.savefig(out3, dpi=150, bbox_inches="tight")
print(f"Saved: {out3}")
plt.close(fig3)

# ── Summary stats table ───────────────────────────────────────────────────
print("\n=== Posterior Summary ===")
params = ["b1","b2","g1","gamma_M","gamma_Y","indirect_effect","total_effect"]
rows = []
for p in params:
    for df, label in [(weak, "Weak"), (strong, "Strong")]:
        s = df[p]
        lo, hi = float(s.quantile(0.025)), float(s.quantile(0.975))
        p_pos = float((s > 0).mean())
        rows.append(dict(model=label, param=p,
                         mean=round(float(s.mean()),3), sd=round(float(s.std()),3),
                         ci_lo=round(lo,3), ci_hi=round(hi,3),
                         prob_pos=round(p_pos,4)))
sumdf = pd.DataFrame(rows)
print(sumdf.to_string(index=False))
