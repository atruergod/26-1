"""시각화 유틸리티 — ICC, 사후분포, shrinkage plot.

모든 함수는 matplotlib Figure/Axes 를 반환하여 노트북에서 추가 편집이 가능하다.
"""

from __future__ import annotations
from typing import Optional, Sequence
import numpy as np
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# 1) 문항특성곡선 (Item Characteristic Curve, ICC)
# ---------------------------------------------------------------------------
def plot_icc_two_groups(
    b_ref: float,
    b_focal: float,
    a_ref: float = 1.0,
    a_focal: float = 1.0,
    theta_range=(-4.0, 4.0),
    ax: Optional[plt.Axes] = None,
    title: Optional[str] = None,
    show_difference: bool = True,
):
    """단일 문항에 대해 두 집단의 ICC를 한 그림에 표시한다.

    Parameters
    ----------
    b_ref, b_focal : float
        두 집단의 (유효) 난이도.
    a_ref, a_focal : float
        두 집단의 변별도. 1PL이면 둘 다 1.0 (또는 동일 값).
    show_difference : bool
        두 곡선 사이 영역을 음영으로 표시할지.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(6.0, 4.0))
    else:
        fig = ax.figure

    theta = np.linspace(*theta_range, 400)
    p_ref = 1.0 / (1.0 + np.exp(-a_ref * (theta - b_ref)))
    p_focal = 1.0 / (1.0 + np.exp(-a_focal * (theta - b_focal)))

    ax.plot(theta, p_ref, lw=2.2, label="Reference group", color="#1f77b4")
    ax.plot(theta, p_focal, lw=2.2, label="Focal group", color="#d62728", ls="--")

    if show_difference:
        ax.fill_between(theta, p_ref, p_focal,
                        where=(p_ref >= p_focal), alpha=0.15, color="#1f77b4")
        ax.fill_between(theta, p_ref, p_focal,
                        where=(p_ref < p_focal), alpha=0.15, color="#d62728")

    ax.axhline(0.5, color="gray", lw=0.6, ls=":")
    ax.set_xlabel(r"Ability $\theta$")
    ax.set_ylabel(r"$P(Y=1\mid\theta)$")
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlim(*theta_range)
    ax.legend(loc="lower right", framealpha=0.9)
    if title:
        ax.set_title(title)
    ax.grid(True, alpha=0.25)
    return fig, ax


def plot_icc_grid(
    b_true: np.ndarray,
    delta_b_true: np.ndarray,
    a_true: Optional[np.ndarray] = None,
    delta_a_true: Optional[np.ndarray] = None,
    ncols: int = 5,
    item_labels: Optional[Sequence[str]] = None,
    figsize_per_panel=(2.6, 2.2),
):
    """모든 문항의 ICC를 격자(grid) 형태로 한꺼번에 표시.

    DIF가 있는 문항을 한눈에 식별할 수 있도록 패널 제목에 표시한다.
    """
    J = len(b_true)
    nrows = int(np.ceil(J / ncols))
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(figsize_per_panel[0] * ncols, figsize_per_panel[1] * nrows),
        sharex=True, sharey=True,
    )
    axes = np.array(axes).reshape(nrows, ncols)

    a_true = np.ones(J) if a_true is None else np.asarray(a_true)
    delta_a_true = np.ones(J) if delta_a_true is None else np.asarray(delta_a_true)

    for j in range(J):
        ax = axes[j // ncols, j % ncols]
        b_ref = b_true[j]
        b_focal = b_true[j] + delta_b_true[j]
        a_ref = a_true[j]
        a_focal = a_true[j] * delta_a_true[j]
        plot_icc_two_groups(
            b_ref=b_ref, b_focal=b_focal,
            a_ref=a_ref, a_focal=a_focal,
            ax=ax, show_difference=True,
        )
        has_dif_b = abs(delta_b_true[j]) > 1e-6
        has_dif_a = abs(delta_a_true[j] - 1.0) > 1e-6
        if has_dif_b or has_dif_a:
            tag = []
            if has_dif_b:
                tag.append(f"Δb={delta_b_true[j]:+.2f}")
            if has_dif_a:
                tag.append(f"Δa×={delta_a_true[j]:.2f}")
            title = f"Item {j+1}  [DIF: {', '.join(tag)}]"
            ax.set_title(title, color="#b00020", fontsize=9)
        else:
            ax.set_title(f"Item {j+1}", fontsize=9)
        ax.legend().set_visible(False)
        ax.set_xlabel("")
        ax.set_ylabel("")

    # 공통 라벨
    for r in range(nrows):
        axes[r, 0].set_ylabel(r"$P(Y=1\mid\theta)$")
    for c in range(ncols):
        axes[-1, c].set_xlabel(r"$\theta$")

    # 빈 패널 숨김
    for j in range(J, nrows * ncols):
        axes[j // ncols, j % ncols].set_visible(False)

    fig.suptitle("Item Characteristic Curves: Reference vs Focal", y=1.0, fontsize=12)
    fig.tight_layout()
    return fig, axes


# ---------------------------------------------------------------------------
# 2) 능력 분포 비교
# ---------------------------------------------------------------------------
def plot_ability_distributions(theta, group, ax=None, bins=30):
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 3.5))
    else:
        fig = ax.figure
    theta = np.asarray(theta)
    group = np.asarray(group)
    ax.hist(theta[group == 0], bins=bins, alpha=0.55,
            label=f"Reference (n={(group==0).sum()})", color="#1f77b4", density=True)
    ax.hist(theta[group == 1], bins=bins, alpha=0.55,
            label=f"Focal (n={(group==1).sum()})", color="#d62728", density=True)
    ax.set_xlabel(r"True ability $\theta$")
    ax.set_ylabel("Density")
    ax.set_title("Ability distributions of two groups")
    ax.legend()
    ax.grid(alpha=0.25)
    return fig, ax


# ---------------------------------------------------------------------------
# 3) DIF 모수 사후분포 forest plot
# ---------------------------------------------------------------------------
def plot_dif_forest(
    delta_samples: np.ndarray,
    item_labels: Optional[Sequence[str]] = None,
    truth: Optional[np.ndarray] = None,
    rope: Optional[tuple] = (-0.2, 0.2),
    hdi_prob: float = 0.95,
    ax=None,
    title: str = "Posterior of DIF effect $\\Delta b_j$",
):
    """문항별 Δb 사후 분포 시각화 (forest plot).

    Parameters
    ----------
    delta_samples : (S, J) ndarray
        S개의 사후 표본 × J 문항.
    truth : (J,) array, optional
        진짜 값 (시뮬레이션용).
    rope : (lo, hi) tuple, optional
        Region of Practical Equivalence — 음영으로 표시.
    hdi_prob : float
        신용구간 확률.
    """
    S, J = delta_samples.shape
    if item_labels is None:
        item_labels = [f"Item {j+1}" for j in range(J)]

    if ax is None:
        fig, ax = plt.subplots(figsize=(6.5, 0.45 * J + 1.5))
    else:
        fig = ax.figure

    means = delta_samples.mean(axis=0)
    alpha = (1.0 - hdi_prob) / 2.0
    lo = np.quantile(delta_samples, alpha, axis=0)
    hi = np.quantile(delta_samples, 1 - alpha, axis=0)

    ys = np.arange(J)[::-1]
    ax.errorbar(means, ys,
                xerr=[means - lo, hi - means],
                fmt="o", color="#222", capsize=3, lw=1.5,
                label=f"Posterior mean ± {int(hdi_prob*100)}% CI")

    if truth is not None:
        ax.scatter(truth, ys, marker="x", color="#d62728", s=70, zorder=5, label="True value")

    if rope is not None:
        ax.axvspan(rope[0], rope[1], alpha=0.12, color="green", label=f"ROPE [{rope[0]}, {rope[1]}]")
    ax.axvline(0, color="gray", lw=0.7, ls=":")

    ax.set_yticks(ys)
    ax.set_yticklabels(item_labels)
    ax.set_xlabel(r"$\Delta b_j$ (focal − reference)")
    ax.set_title(title)
    ax.legend(loc="best", framealpha=0.9, fontsize=9)
    ax.grid(axis="x", alpha=0.25)
    return fig, ax


# ---------------------------------------------------------------------------
# 4) 단일 모수 사후 밀도 (with ROPE)
# ---------------------------------------------------------------------------
def plot_posterior_density(samples, truth=None, rope=(-0.2, 0.2),
                           ax=None, title="Posterior density"):
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 3.5))
    else:
        fig = ax.figure
    samples = np.asarray(samples).ravel()
    ax.hist(samples, bins=40, density=True, alpha=0.6, color="#444")
    if rope is not None:
        ax.axvspan(rope[0], rope[1], alpha=0.18, color="green",
                   label=f"ROPE [{rope[0]}, {rope[1]}]")
    if truth is not None:
        ax.axvline(truth, color="#d62728", lw=2, ls="--", label=f"True = {truth:.2f}")
    ax.axvline(np.mean(samples), color="#1f77b4", lw=2, label=f"Mean = {np.mean(samples):.2f}")
    ax.set_xlabel(r"$\Delta b$")
    ax.set_ylabel("Density")
    ax.set_title(title)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.25)
    return fig, ax


# ---------------------------------------------------------------------------
# 5) Shrinkage plot — raw 추정 vs 위계모형 추정
# ---------------------------------------------------------------------------
def plot_shrinkage(raw_estimates, shrunk_estimates, truth=None,
                   item_labels=None, ax=None,
                   title="Shrinkage: non-hierarchical → hierarchical"):
    J = len(raw_estimates)
    if item_labels is None:
        item_labels = [f"{j+1}" for j in range(J)]
    if ax is None:
        fig, ax = plt.subplots(figsize=(6.5, 4.5))
    else:
        fig = ax.figure
    xs_raw = np.full(J, 0.0)
    xs_shr = np.full(J, 1.0)
    for j in range(J):
        ax.plot([0, 1], [raw_estimates[j], shrunk_estimates[j]],
                color="gray", alpha=0.5, lw=1.0)
        ax.scatter(0, raw_estimates[j], color="#1f77b4", s=40, zorder=3)
        ax.scatter(1, shrunk_estimates[j], color="#2ca02c", s=40, zorder=3)
        ax.text(1.04, shrunk_estimates[j], item_labels[j], fontsize=8, va="center")
    if truth is not None:
        for j in range(J):
            ax.scatter(-0.06, truth[j], color="#d62728", marker="x", s=50)
    ax.axhline(0, color="gray", lw=0.5, ls=":")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Non-hierarchical\n(independent)", "Hierarchical\n(shrunk)"])
    ax.set_ylabel(r"$\Delta b_j$ estimate")
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.25)
    return fig, ax
