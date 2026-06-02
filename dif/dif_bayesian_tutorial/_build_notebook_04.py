"""Notebook 04 빌더 — Spike-and-Slab / Horseshoe prior 와 anchor-free 검출."""
import json
from pathlib import Path

NB_PATH = Path(__file__).parent / "notebooks" / "04_spike_slab_horseshoe.ipynb"


def md(text): return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}
def code(text): return {"cell_type": "code", "metadata": {}, "source": text.splitlines(keepends=True),
                        "outputs": [], "execution_count": None}


cells = []

cells.append(md("""# Notebook 04 — Spike-and-Slab / Horseshoe Prior 와 Anchor-Free DIF 검출

> **장점 #4 + #6 동시 시연**: 사전정보(sparsity 가정)의 통합과 anchor item 선택 문제의 완화.
> 두 장점이 **같은 prior 설계**로 해결되기 때문에 한 노트북에서 함께 다룹니다.

학습 목표 (Learning Objectives)
- "대부분의 문항은 DIF가 없다"는 가정을 prior로 표현하는 두 방법을 안다:
  - **Spike-and-slab prior** — 이산 indicator를 연속 혼합으로 근사
  - **Horseshoe prior** — 무거운 꼬리(heavy tail)의 연속 정규화
- 두 방법의 사후 **inclusion probability** / **shrinkage factor** 를 해석한다.
- 전통적 anchor item 사전 지정 없이 DIF를 자동 식별한다.
"""))

cells.append(md("""## 1. Anchor item 문제 (The Anchor Item Problem)

전통적 IRT DIF 검출은 "DIF가 없는 anchor item"을 미리 지정해야 식별성이 확보됩니다.
모든 문항을 자유롭게 두면 두 집단의 척도가 같은지 다른지 알 수 없기 때문입니다.

**문제**: anchor 선택이 결과를 좌우합니다.
- DIF가 있는 문항을 잘못 anchor로 지정 → 다른 문항에 가짜 DIF가 표시됨.
- Anchor를 너무 적게 지정 → 식별성 약화.
- 보수적으로 모든 문항을 anchor 후보로 시도 → 계산 폭증, 안정성 저하.

**해결 아이디어**: 모든 문항의 $\\Delta b_j$ 에 **sparsity 유도 prior**를 부여한다.
- 진짜 DIF가 없는 문항은 prior에 의해 자동으로 0에 가깝게 추정 → de facto anchor.
- 진짜 DIF가 있는 문항만 0에서 멀어짐.
→ **사전에 anchor를 지정할 필요 없음 (anchor-free)**.
"""))

cells.append(code("""# Backend selection
BACKEND = "stan"
import platform, importlib.util, warnings
def _resolve(req):
    req = req.lower()
    if req == "numpyro":
        ok = (importlib.util.find_spec("jax") is not None
              and importlib.util.find_spec("numpyro") is not None)
        if not ok:
            warnings.warn("numpyro unavailable → Stan fallback")
            return "stan"
    return "stan" if req != "numpyro" else "numpyro"
BACKEND = _resolve(BACKEND)
print(f"Active backend: {BACKEND}")
"""))

cells.append(code("""import sys
from pathlib import Path
PROJECT_ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from difbayes import simulate, visualize, frequentist, diagnostics, models
plt.rcParams.update({"figure.dpi": 110, "axes.spines.top": False, "axes.spines.right": False, "font.size": 10})
"""))

cells.append(md("""## 2. 두 사전 비교

### Spike-and-Slab Prior (연속 근사)

$$
\\Delta b_j \\sim \\pi_j \\cdot N(0, \\sigma_{\\text{slab}}^2) + (1-\\pi_j) \\cdot N(0, \\sigma_{\\text{spike}}^2)
$$

- $\\sigma_{\\text{spike}} \\ll \\sigma_{\\text{slab}}$: spike는 좁고, slab은 넓음.
- $\\pi_j$ 는 "DIF가 있을 사전 확률" — Beta prior로 약하게 학습.
- 사후 **inclusion probability** 가 자연스럽게 산출됨.

### Horseshoe Prior (Carvalho, Polson & Scott, 2010)

$$
\\Delta b_j = \\tau \\cdot \\lambda_j \\cdot z_j,\\quad
\\lambda_j \\sim \\mathrm{Cauchy}^+(0,1),\\;
\\tau \\sim \\mathrm{Cauchy}^+(0, 0.3)
$$

- **두꺼운 꼬리 + 좁은 봉우리**: 큰 효과는 거의 그대로 통과, 작은 효과는 강하게 0으로 축소.
- **Shrinkage factor** $\\kappa_j = 1/(1 + \\lambda_j^2 \\tau^2) \\in [0,1]$:
  - $\\kappa_j \\approx 1$ → 강하게 축소 (DIF 없음 추정).
  - $\\kappa_j \\approx 0$ → 축소 없음 (DIF 존재 추정).
"""))

cells.append(md("""## 3. 자료 — 10문항 중 진짜 DIF 2개
"""))

cells.append(code("""data = simulate.scenario_sparse_dif_10items(n_ref=400, n_focal=400, seed=2026)
print("True Δb:", data.delta_b_true.round(2))
print("진짜 DIF 문항:", (np.where(np.abs(data.delta_b_true) > 1e-6)[0] + 1).tolist())
"""))

cells.append(md("""## 4. 세 가지 prior 비교 적합
"""))

cells.append(code("""# (a) Weakly-informative Normal (Notebook 00과 동일)
fit_normal = models.fit_rasch_dif(
    Y=data.Y, group=data.group, backend=BACKEND,
    n_chains=4, n_warmup=500, n_samples=1000,
    prior_sigma_delta=1.0, seed=2026,
)
delta_normal = fit_normal["samples"]["delta"].reshape(-1, data.J)
"""))

cells.append(code("""# (b) Spike-and-slab
fit_ss = models.fit_rasch_spike_slab(
    Y=data.Y, group=data.group, backend=BACKEND,
    n_chains=4, n_warmup=800, n_samples=1500,
    slab_sd=1.0, prior_inclusion=0.2, seed=2026,
)
delta_ss = fit_ss["samples"]["delta"].reshape(-1, data.J)
post_incl = fit_ss["samples"]["post_inclusion"].reshape(-1, data.J)
print("Posterior inclusion probabilities (mean per item):")
print(post_incl.mean(axis=0).round(3))
"""))

cells.append(code("""# (c) Horseshoe
fit_hs = models.fit_rasch_horseshoe(
    Y=data.Y, group=data.group, backend=BACKEND,
    n_chains=4, n_warmup=800, n_samples=1500, seed=2026,
)
delta_hs = fit_hs["samples"]["delta"].reshape(-1, data.J)
shrink = fit_hs["samples"]["shrinkage_factor"].reshape(-1, data.J)
print("Shrinkage factor κ (mean per item):  (1 = strong shrink, 0 = none)")
print(shrink.mean(axis=0).round(3))
"""))

cells.append(md("""## 5. Forest plot 3-way 비교
"""))

cells.append(code("""fig, axes = plt.subplots(1, 3, figsize=(15, 5))
visualize.plot_dif_forest(delta_normal, truth=data.delta_b_true, rope=(-0.2, 0.2),
                          ax=axes[0], title="Normal weakly-informative")
visualize.plot_dif_forest(delta_ss, truth=data.delta_b_true, rope=(-0.2, 0.2),
                          ax=axes[1], title="Spike-and-slab")
visualize.plot_dif_forest(delta_hs, truth=data.delta_b_true, rope=(-0.2, 0.2),
                          ax=axes[2], title="Horseshoe")
fig.tight_layout()
fig.savefig("../outputs/04_forest_priors.png", dpi=120, bbox_inches="tight")
plt.show()
"""))

cells.append(md("""## 6. Inclusion probability / Shrinkage factor 시각화

Spike-and-slab의 **inclusion probability** 와 Horseshoe의 **shrinkage factor (1−κ)** 를
나란히 표시합니다. 둘 다 "DIF가 있다는 신호의 강도"로 해석됩니다.
"""))

cells.append(code("""fig, axes = plt.subplots(1, 2, figsize=(13, 4))
items = np.arange(1, data.J + 1)
truth_mask = np.abs(data.delta_b_true) > 1e-6
colors = ["#d62728" if t else "#888" for t in truth_mask]

# (a) Spike-and-slab inclusion probability
incl_mean = post_incl.mean(axis=0)
axes[0].bar(items, incl_mean, color=colors)
axes[0].axhline(0.5, color="gray", lw=0.5, ls=":")
axes[0].set_xticks(items)
axes[0].set_xlabel("Item")
axes[0].set_ylabel("Posterior inclusion probability")
axes[0].set_ylim(0, 1)
axes[0].set_title("Spike-and-slab: P(slab membership)")
axes[0].grid(axis="y", alpha=0.25)

# (b) Horseshoe: 1 - κ (signal strength)
signal = 1.0 - shrink.mean(axis=0)
axes[1].bar(items, signal, color=colors)
axes[1].axhline(0.5, color="gray", lw=0.5, ls=":")
axes[1].set_xticks(items)
axes[1].set_xlabel("Item")
axes[1].set_ylabel("1 − κ (signal strength)")
axes[1].set_ylim(0, 1)
axes[1].set_title("Horseshoe: signal strength")
axes[1].grid(axis="y", alpha=0.25)

fig.suptitle("Sparsity-induced anchor-free DIF detection  (red = true DIF)", y=1.03)
fig.tight_layout()
fig.savefig("../outputs/04_inclusion_shrinkage.png", dpi=120, bbox_inches="tight")
plt.show()
"""))

cells.append(md("""## 7. 비교 요약 표
"""))

cells.append(code("""rows = []
for j in range(data.J):
    rows.append(dict(
        Item=j+1,
        truth=data.delta_b_true[j],
        normal_mean=delta_normal[:, j].mean(),
        ss_mean=delta_ss[:, j].mean(),
        ss_incl=post_incl[:, j].mean(),
        hs_mean=delta_hs[:, j].mean(),
        hs_signal=1 - shrink[:, j].mean(),
    ))
out = pd.DataFrame(rows).round(3)
print(out.to_string(index=False))
"""))

cells.append(md("""## 8. 요약

**핵심 관찰**

1. Normal prior 도 자료가 충분하면 진짜 DIF 문항을 잡지만, **0 근처 추정의 분산**이 큼.
2. Spike-and-slab 의 **inclusion probability** 와 Horseshoe 의 **1−κ** 는 둘 다 "DIF 신호 강도"의
   직관적 척도로 활용 가능 (예: > 0.5 이면 양성).
3. 두 sparsity prior 모두 **anchor item을 사전에 지정하지 않고도** DIF를 식별.

**Sparsity prior 의 실무 권장**

| 조건 | 권장 |
|---|---|
| 문항 수 적음 (J ≤ 20), DIF 비율 추측 가능 | Spike-and-slab (`prior_inclusion` 조정) |
| 문항 수 큼 (J > 30), 연속적 비교가 좋다 | Horseshoe |
| 단일 강한 효과만 탐색 | Horseshoe |
| 사후 inclusion 확률을 직접 보고하고 싶다 | Spike-and-slab |

**다음 노트북**

- **부록 A** — 2PL로 확장하여 **non-uniform DIF**(변별도 차이)까지 검출.
"""))

notebook = {"cells": cells,
            "metadata": {"kernelspec": {"display_name": "Python 3 (ipykernel)", "language": "python", "name": "python3"},
                         "language_info": {"name": "python", "version": "3.11"}},
            "nbformat": 4, "nbformat_minor": 5}
NB_PATH.parent.mkdir(parents=True, exist_ok=True)
NB_PATH.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"Notebook saved: {NB_PATH}")
print(f"Cells: {len(cells)}")
