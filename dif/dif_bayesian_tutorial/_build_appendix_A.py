"""Appendix A 빌더 — 2PL 확장: non-uniform DIF 검출."""
import json
from pathlib import Path

NB_PATH = Path(__file__).parent / "notebooks" / "appendix_A_2pl_extension.ipynb"


def md(text): return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}
def code(text): return {"cell_type": "code", "metadata": {}, "source": text.splitlines(keepends=True),
                        "outputs": [], "execution_count": None}


cells = []

cells.append(md("""# 부록 A — 2PL 확장: Non-Uniform DIF 검출

> 본 부록은 1PL이 다루지 못하는 **non-uniform DIF**(변별도 차이로 인한 DIF)를
> 2PL 모형으로 확장하여 검출하는 방법을 보여줍니다.

학습 목표 (Learning Objectives)
- 2PL 모형의 변별도(discrimination, $a_j$)와 DIF의 관계를 이해한다.
- Uniform DIF($\\Delta b$)와 non-uniform DIF($\\Delta a$)를 **동시에** 추정한다.
- 메인 노트북에서 배운 베이지안 도구들이 2PL로 그대로 일반화됨을 체험한다.
"""))

cells.append(md("""## 1. 2PL 모형과 DIF

### 1.1. 응답확률

$$
P(Y_{ij} = 1 \\mid \\theta_i, a_j, b_j, g_i) = \\text{sigmoid}\\big(a_j^{\\text{eff}}(g_i) \\cdot (\\theta_i - b_j^{\\text{eff}}(g_i))\\big)
$$

여기서 집단별 효과는:

$$
a_j^{\\text{eff}}(g_i) = a_j \\cdot \\exp(g_i \\cdot \\log\\Delta a_j),\\quad
b_j^{\\text{eff}}(g_i) = b_j + g_i \\cdot \\Delta b_j
$$

### 1.2. DIF 유형 정리

| 모수 | 변화 | DIF 유형 | 시각적 형태 |
|---|---|---|---|
| $\\Delta b_j \\neq 0$ | 난이도 이동 | **Uniform DIF** | ICC가 좌우 평행 이동 |
| $\\Delta a_j \\neq 1$ | 변별도 변화 | **Non-uniform DIF** | ICC가 한 점에서 교차 |
| 둘 다 변화 | 복합 | 둘 다 존재 | 교차 + 이동 |

### 1.3. Priors (weakly-informative)

- $a_j \\sim \\mathrm{LogNormal}(0, 0.4)$ — 변별도는 양수, 1 근처 집중.
- $b_j \\sim N(0, 2)$
- $\\Delta b_j \\sim N(0, 0.5)$ — uniform DIF (weakly-informative shrinkage)
- $\\log \\Delta a_j \\sim N(0, 0.3)$ — non-uniform DIF (multiplicative ratio)
"""))

cells.append(code("""# Backend
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

cells.append(md("""## 2. 시뮬레이션 자료 — 3가지 DIF 유형 혼합

10문항, 두 집단 각 400명. DIF 시나리오:
- **문항 3**: uniform DIF만 ($\\Delta b = +0.6$, $\\Delta a = 1.0$)
- **문항 6**: non-uniform DIF만 ($\\Delta b = 0$, $\\Delta a = 1.6$)
- **문항 9**: 두 가지 동시 ($\\Delta b = -0.4$, $\\Delta a = 0.7$)
- 나머지 7문항: DIF 없음
"""))

cells.append(code("""data = simulate.scenario_2pl_nonuniform(seed=2026)
print(f"N = {data.N}, J = {data.J}, n_ref = {data.n_ref}, n_focal = {data.n_focal}")
truth = pd.DataFrame({
    "Item": np.arange(1, data.J + 1),
    "b_true": data.b_true.round(2),
    "a_true": data.a_true.round(2),
    "Δb_true": data.delta_b_true.round(2),
    "Δa_true": data.delta_a_true.round(2),
    "DIF type": [
        ("none" if (abs(db) < 1e-6 and abs(da - 1.0) < 1e-6) else
         "uniform" if abs(da - 1.0) < 1e-6 else
         "non-uniform" if abs(db) < 1e-6 else
         "both")
        for db, da in zip(data.delta_b_true, data.delta_a_true)
    ],
})
print(truth.to_string(index=False))
"""))

cells.append(md("""## 3. ICC 격자 시각화 — Non-uniform DIF 의 모습

특히 문항 6의 ICC를 잘 보세요. 두 집단 곡선이 **한 점에서 교차**합니다.
"""))

cells.append(code("""fig, axes = visualize.plot_icc_grid(
    b_true=data.b_true,
    delta_b_true=data.delta_b_true,
    a_true=data.a_true,
    delta_a_true=data.delta_a_true,
    ncols=5,
)
fig.savefig("../outputs/A_icc_grid_2pl.png", dpi=120, bbox_inches="tight")
plt.show()
"""))

cells.append(md("""**관찰**

- 문항 3 (uniform DIF): ICC가 평행 이동.
- 문항 6 (non-uniform DIF): ICC가 한 점에서 교차 — 능력이 낮을 때 focal에 유리, 능력이 높을 때 reference에 유리(또는 그 반대).
- 문항 9: 교차 + 이동 = 복합.

> 💡 **포인트**: 1PL은 변별도를 동일하게 가정하므로 문항 6 같은 교차형 DIF를 모형화할 수 없습니다.
> 2PL에서 $\\Delta a_j$ 를 도입해야 진짜 모습을 잡을 수 있습니다.
"""))

cells.append(md("""## 4. 베이지안 2PL DIF 적합

> ⏱ 2PL은 1PL보다 모수가 2배 많아 적합 시간이 길어집니다 (~2~3분).
"""))

cells.append(code("""fit = models.fit_2pl_dif(
    Y=data.Y, group=data.group, backend=BACKEND,
    n_chains=4, n_warmup=800, n_samples=1500, seed=2026,
)
delta_b = fit["samples"]["delta_b"].reshape(-1, data.J)
delta_a_ratio = fit["samples"]["delta_a_ratio"].reshape(-1, data.J)
print(f"delta_b shape: {delta_b.shape}")
print(f"delta_a_ratio shape: {delta_a_ratio.shape}")
"""))

cells.append(md("""## 5. 결과 표

진짜 값과 사후 평균을 나란히 비교합니다.
"""))

cells.append(code("""rows = []
for j in range(data.J):
    rows.append(dict(
        Item=j+1,
        b_true_Δ=data.delta_b_true[j].round(2),
        b_post_Δ=delta_b[:, j].mean().round(3),
        b_ci=f"[{np.quantile(delta_b[:, j], 0.025):+.2f}, {np.quantile(delta_b[:, j], 0.975):+.2f}]",
        a_true_ratio=data.delta_a_true[j].round(2),
        a_post_ratio=delta_a_ratio[:, j].mean().round(3),
        a_ci=f"[{np.quantile(delta_a_ratio[:, j], 0.025):.2f}, {np.quantile(delta_a_ratio[:, j], 0.975):.2f}]",
    ))
print(pd.DataFrame(rows).to_string(index=False))
"""))

cells.append(md("""## 6. 두 종류 DIF 모수의 Forest plot
"""))

cells.append(code("""fig, axes = plt.subplots(1, 2, figsize=(13, 5))
# Uniform DIF (Δb)
visualize.plot_dif_forest(
    delta_b, truth=data.delta_b_true, rope=(-0.2, 0.2),
    ax=axes[0], title="Uniform DIF: Δb"
)
# Non-uniform DIF (Δa ratio): ROPE는 1 근처
# 비율의 ROPE: [0.8, 1.25] 정도가 무의미 영역
log_ratio = np.log(delta_a_ratio)
visualize.plot_dif_forest(
    log_ratio, truth=np.log(data.delta_a_true), rope=(np.log(0.8), np.log(1.25)),
    ax=axes[1], title="Non-uniform DIF: log Δa"
)
axes[1].set_xlabel(r"$\\log(\\Delta a)$  (0 means no non-uniform DIF)")
fig.tight_layout()
fig.savefig("../outputs/A_forest_2pl.png", dpi=120, bbox_inches="tight")
plt.show()
"""))

cells.append(md("""## 7. 사후 ICC — 추정된 모형이 진짜를 잘 따라가는가?

문항 6의 추정된 ICC를 진짜 ICC와 비교합니다.
"""))

cells.append(code("""b_post = fit["samples"]["b"].reshape(-1, data.J).mean(axis=0)
a_post = fit["samples"]["a"].reshape(-1, data.J).mean(axis=0)

j = 5   # 문항 6 (non-uniform DIF)
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
visualize.plot_icc_two_groups(
    b_ref=data.b_true[j],
    b_focal=data.b_true[j] + data.delta_b_true[j],
    a_ref=data.a_true[j],
    a_focal=data.a_true[j] * data.delta_a_true[j],
    ax=axes[0],
    title=f"TRUE — Item {j+1} (non-uniform DIF)"
)
visualize.plot_icc_two_groups(
    b_ref=b_post[j],
    b_focal=b_post[j] + delta_b[:, j].mean(),
    a_ref=a_post[j],
    a_focal=a_post[j] * delta_a_ratio[:, j].mean(),
    ax=axes[1],
    title=f"POSTERIOR — Item {j+1}"
)
fig.tight_layout()
fig.savefig("../outputs/A_icc_recovery.png", dpi=120, bbox_inches="tight")
plt.show()
"""))

cells.append(md("""## 8. 1PL vs 2PL 비교 — Non-uniform DIF가 있을 때

같은 자료에 1PL DIF 모형을 적합해보면, 1PL은 non-uniform DIF를 어떻게 (잘못) 해석하는지 알 수 있습니다.
"""))

cells.append(code("""fit_1pl = models.fit_rasch_dif(
    Y=data.Y, group=data.group, backend=BACKEND,
    n_chains=4, n_warmup=500, n_samples=1000,
    prior_sigma_delta=0.5, seed=2026,
)
delta_b_1pl = fit_1pl["samples"]["delta"].reshape(-1, data.J)

cmp = pd.DataFrame({
    "Item": np.arange(1, data.J + 1),
    "true Δb": data.delta_b_true.round(2),
    "true Δa": data.delta_a_true.round(2),
    "1PL Δb (mean)": delta_b_1pl.mean(axis=0).round(3),
    "2PL Δb (mean)": delta_b.mean(axis=0).round(3),
    "2PL Δa (mean)": delta_a_ratio.mean(axis=0).round(3),
})
print(cmp.to_string(index=False))
"""))

cells.append(md("""**관찰 포인트**

- 문항 6은 **진짜 $\\Delta b = 0$, 진짜 $\\Delta a = 1.6$**.
  - 1PL은 변별도 차이를 모형화할 수 없어, $\\Delta a$ 의 효과를 (잘못된) **평균적 $\\Delta b$** 로 흡수하려고 함.
  - 2PL은 두 효과를 깨끗이 분리.
- 진짜로 uniform DIF만 있는 문항(예: 문항 3)에서는 두 모형이 비슷한 결과.

> 💡 **교훈**: 자료에 non-uniform DIF가 있을 가능성이 있다면 2PL 기반 분석이 안전합니다.
> 1PL은 단순성·해석성의 장점이 있지만, 등변별성 가정이 깨지면 결과가 왜곡될 수 있습니다.
"""))

cells.append(md("""## 9. 요약

**핵심**

1. **2PL 확장**으로 uniform과 non-uniform DIF를 **동시에** 검출 가능.
2. 본 자료의 모든 베이지안 도구(weak prior, hierarchical, spike-and-slab, horseshoe)는
   2PL에서도 그대로 작동 — 다만 모수가 2배라서 데이터 요구량이 늘어남.
3. Non-uniform DIF의 존재가 의심되면 2PL 적합 권장.

**확장 방향 (Further Extensions)**

- 다집단(multi-group) DIF: $g_i \\in \\{0, 1, 2, \\dots\\}$.
- 다차원 IRT(MIRT) + DIF.
- 종단(longitudinal) IRT — 시간 변화하는 DIF.
- Multilevel IRT — 학교/지역 효과 + DIF.

**참고 문헌**

- Lord, F. M. (1980). *Applications of Item Response Theory to Practical Testing Problems*.
- Raju, N. S. (1990). Determining the significance of estimated signed and unsigned areas between two ICCs. *Applied Psychological Measurement*.
- Fox, J.-P. (2010). *Bayesian Item Response Modeling*.
"""))

notebook = {"cells": cells,
            "metadata": {"kernelspec": {"display_name": "Python 3 (ipykernel)", "language": "python", "name": "python3"},
                         "language_info": {"name": "python", "version": "3.11"}},
            "nbformat": 4, "nbformat_minor": 5}
NB_PATH.parent.mkdir(parents=True, exist_ok=True)
NB_PATH.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"Notebook saved: {NB_PATH}")
print(f"Cells: {len(cells)}")
