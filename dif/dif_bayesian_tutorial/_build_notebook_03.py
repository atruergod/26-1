"""Notebook 03 빌더 — 위계모형 shrinkage 와 다중검정 완화."""
import json
from pathlib import Path

NB_PATH = Path(__file__).parent / "notebooks" / "03_hierarchical_shrinkage.ipynb"


def md(text): return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}
def code(text): return {"cell_type": "code", "metadata": {}, "source": text.splitlines(keepends=True),
                        "outputs": [], "execution_count": None}


cells = []

cells.append(md("""# Notebook 03 — 위계모형과 자동 Shrinkage

> **장점 #3**: 베이지안 위계모형(hierarchical model)은 문항별 DIF 모수를 공통 분포에서
> 추출되는 것으로 모델링하여 **자동적인 축소(shrinkage)**를 일으키고, 결과적으로 다중검정 문제를 완화합니다.

학습 목표 (Learning Objectives)
- **위계 사전(hierarchical prior)**의 구조 $\\Delta b_j \\sim N(0, \\tau^2)$를 이해한다.
- **non-hierarchical** vs **hierarchical** 추정을 같은 자료에 적용해 차이를 본다.
- **Shrinkage plot** 으로 축소 효과를 시각적으로 본다.
- 다중검정 상황에서 false positive rate / power 비교를 본다.
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

cells.append(md("""## 1. 모형 비교

| 모형 | 사전 | 특징 |
|---|---|---|
| Non-hierarchical | $\\Delta b_j \\sim N(0, \\sigma^2)$ with **fixed** $\\sigma=1$ | 문항 간 독립, 데이터가 사전을 거의 갈아엎음 |
| Hierarchical | $\\Delta b_j \\sim N(0, \\tau^2)$, $\\tau \\sim \\mathrm{HalfNormal}(0.5)$ | **τ가 데이터에서 추정** → 자동 shrinkage |

위계모형에서는 모든 문항이 "DIF는 작을 것"이라는 정보를 **데이터로부터 함께 학습**합니다.
DIF가 드문 상황(대부분 0)에서는 $\\tau$ 가 작아져 모든 문항이 0 쪽으로 축소됩니다.
"""))

cells.append(md("""## 2. 메인 시나리오 — 10문항, 진짜 DIF는 2개

Notebook 00과 유사하지만, DIF가 있는 문항을 강조하기 위해 약간 다른 시나리오 사용.
"""))

cells.append(code("""data = simulate.scenario_sparse_dif_10items(n_ref=300, n_focal=300, seed=2026)
truth = data.delta_b_true
print("True Δb:", truth.round(2))
print(f"DIF 문항: {np.where(np.abs(truth) > 1e-6)[0] + 1}")
"""))

cells.append(code("""# (a) Non-hierarchical
fit_indep = models.fit_rasch_dif(
    Y=data.Y, group=data.group, backend=BACKEND,
    n_chains=4, n_warmup=500, n_samples=1000,
    prior_sigma_delta=1.0, seed=2026,
)
delta_indep = fit_indep["samples"]["delta"].reshape(-1, data.J)

# (b) Hierarchical
fit_hier = models.fit_rasch_hierarchical_dif(
    Y=data.Y, group=data.group, backend=BACKEND,
    n_chains=4, n_warmup=500, n_samples=1000, seed=2026,
)
delta_hier = fit_hier["samples"]["delta"].reshape(-1, data.J)
tau_post = fit_hier["samples"]["tau"].flatten()
print(f"τ posterior mean: {tau_post.mean():.3f}  (smaller = stronger shrinkage)")
"""))

cells.append(md("""## 3. Shrinkage plot

각 문항의 추정치가 non-hierarchical → hierarchical 로 가면서 얼마나 0 쪽으로 끌려가는지를
연결선으로 시각화합니다.
"""))

cells.append(code("""raw_means = delta_indep.mean(axis=0)
shr_means = delta_hier.mean(axis=0)
fig, ax = visualize.plot_shrinkage(
    raw_estimates=raw_means,
    shrunk_estimates=shr_means,
    truth=truth,
    item_labels=[f"{j+1}" for j in range(data.J)],
)
fig.savefig("../outputs/03_shrinkage.png", dpi=120, bbox_inches="tight")
plt.show()
"""))

cells.append(md("""**해석**

- 진짜로 0인 문항(파란 점들)이 hierarchical에서 더 0 쪽으로 모임 → **거짓양성 감소**.
- 진짜 DIF 문항(예: 문항 3, 8)도 약간 축소되지만 0과 명확히 구분됨 → **민감도 유지**.
- 빨간 ×는 진짜 값. 축소된 추정이 진짜 값에 더 가까운지 비교.
"""))

cells.append(md("""## 4. Forest plot 비교
"""))

cells.append(code("""fig, axes = plt.subplots(1, 2, figsize=(13, 5))
visualize.plot_dif_forest(delta_indep, truth=truth, rope=(-0.2, 0.2),
                          ax=axes[0], title="Non-hierarchical")
visualize.plot_dif_forest(delta_hier, truth=truth, rope=(-0.2, 0.2),
                          ax=axes[1], title="Hierarchical (with τ)")
fig.tight_layout()
fig.savefig("../outputs/03_forest_compare.png", dpi=120, bbox_inches="tight")
plt.show()
"""))

cells.append(md("""## 5. 확장 시뮬레이션 — J = 30 으로 다중검정 효과 확인

10문항으로는 다중검정의 영향이 잘 드러나지 않을 수 있어, J=30인 보조 시나리오로
**false positive rate**를 비교합니다.
- 30문항 중 5문항만 진짜 DIF (Δb ∈ ±[0.5, 0.8])
- 나머지 25문항은 Δb = 0
"""))

cells.append(code("""rng = np.random.default_rng(2026)
J_big = 30
b_big = np.linspace(-2.5, 2.5, J_big)
delta_big = np.zeros(J_big)
dif_idx = rng.choice(J_big, size=5, replace=False)
for j in dif_idx:
    sign = rng.choice([-1, 1])
    delta_big[j] = sign * rng.uniform(0.5, 0.8)
print(f"진짜 DIF 문항 (index): {sorted(dif_idx)}")
print(f"진짜 Δb: {delta_big[dif_idx].round(2)}")

big_data = simulate.simulate_rasch_dif(
    n_ref=400, n_focal=400, b_true=b_big, delta_b_true=delta_big, seed=2027,
)
"""))

cells.append(code("""# Non-hierarchical
fit_b_indep = models.fit_rasch_dif(
    Y=big_data.Y, group=big_data.group, backend=BACKEND,
    n_chains=4, n_warmup=500, n_samples=1000,
    prior_sigma_delta=1.0, seed=2027,
)
d_indep = fit_b_indep["samples"]["delta"].reshape(-1, J_big)

# Hierarchical
fit_b_hier = models.fit_rasch_hierarchical_dif(
    Y=big_data.Y, group=big_data.group, backend=BACKEND,
    n_chains=4, n_warmup=500, n_samples=1000, seed=2027,
)
d_hier = fit_b_hier["samples"]["delta"].reshape(-1, J_big)
print(f"τ posterior mean (J=30): {fit_b_hier['samples']['tau'].mean():.3f}")
"""))

cells.append(code("""# 의사결정 규칙: 95% CI 가 ROPE [-0.2, +0.2] 와 겹치지 않으면 'DIF 양성'
def detect(samples, rope=(-0.2, 0.2)):
    lo = np.quantile(samples, 0.025)
    hi = np.quantile(samples, 0.975)
    return (hi < rope[0]) or (lo > rope[1])

true_pos = np.abs(delta_big) > 1e-6   # 진짜 DIF mask
detect_indep = np.array([detect(d_indep[:, j]) for j in range(J_big)])
detect_hier  = np.array([detect(d_hier[:, j])  for j in range(J_big)])

def tpr_fpr(detected, true_pos):
    tpr = np.mean(detected[true_pos])   # sensitivity
    fpr = np.mean(detected[~true_pos])  # false positive rate
    return tpr, fpr

tpr_i, fpr_i = tpr_fpr(detect_indep, true_pos)
tpr_h, fpr_h = tpr_fpr(detect_hier, true_pos)

cmp = pd.DataFrame({
    "Method": ["Non-hierarchical", "Hierarchical"],
    "True positive rate (power)": [tpr_i, tpr_h],
    "False positive rate":        [fpr_i, fpr_h],
}).round(3)
print(cmp.to_string(index=False))
"""))

cells.append(code("""# 시각화
fig, axes = plt.subplots(1, 2, figsize=(14, 7))
visualize.plot_dif_forest(d_indep, truth=delta_big, rope=(-0.2, 0.2),
                          item_labels=[f"I{j+1}" for j in range(J_big)],
                          ax=axes[0], title=f"Non-hierarchical (J={J_big})")
visualize.plot_dif_forest(d_hier, truth=delta_big, rope=(-0.2, 0.2),
                          item_labels=[f"I{j+1}" for j in range(J_big)],
                          ax=axes[1], title=f"Hierarchical (J={J_big})")
fig.tight_layout()
fig.savefig("../outputs/03_forest_big.png", dpi=120, bbox_inches="tight")
plt.show()
"""))

cells.append(md("""## 6. 요약

**관찰**

1. 위계모형의 **추정치 분산이 작다** → 0 근처 문항이 명확히 0으로 압축.
2. **False positive rate가 감소**하면서도 true positive(검출력)은 거의 유지.
3. $\\tau$ 가 데이터에서 학습되므로, DIF가 많은 자료에서는 자동으로 축소가 약해짐(adaptive).

**왜 다중검정 보정이 자동으로 되는가?**

전통적으로는 J개 문항에 대한 독립 검정 후 Bonferroni 등의 보정이 필요합니다.
베이지안 위계모형에서는 모든 문항이 공통 분포를 공유하므로,
**"대부분의 문항은 DIF가 작다"**는 정보가 데이터로부터 자동 학습되어 각 추정에 반영됩니다.
이는 다중검정의 false discovery rate 측면에서 동일한 효과를 냅니다 (Gelman et al., 2012).

다음 노트북(**04**)에서는 **spike-and-slab / horseshoe prior** 로 anchor item 문제까지 함께 해결합니다.
"""))

notebook = {"cells": cells,
            "metadata": {"kernelspec": {"display_name": "Python 3 (ipykernel)", "language": "python", "name": "python3"},
                         "language_info": {"name": "python", "version": "3.11"}},
            "nbformat": 4, "nbformat_minor": 5}
NB_PATH.parent.mkdir(parents=True, exist_ok=True)
NB_PATH.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"Notebook saved: {NB_PATH}")
print(f"Cells: {len(cells)}")
