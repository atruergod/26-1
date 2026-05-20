"""Notebook 02 빌더 — 불확실성의 자연스러운 정량화."""
import json
from pathlib import Path

NB_PATH = Path(__file__).parent / "notebooks" / "02_uncertainty_quantification.ipynb"


def md(text): return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}
def code(text): return {"cell_type": "code", "metadata": {}, "source": text.splitlines(keepends=True),
                        "outputs": [], "execution_count": None}


cells = []

cells.append(md("""# Notebook 02 — 불확실성의 자연스러운 정량화

> **장점 #2**: 베이지안 추정은 모수의 **사후분포(posterior distribution)** 전체를 제공하므로,
> "점추정 + p-value"의 빈도주의 표현을 넘어 **사후확률(posterior probability)**에 기반한
> 풍부하고 직관적인 진술이 가능합니다.

학습 목표 (Learning Objectives)
- 사후분포에서 직접 얻을 수 있는 다양한 진술의 종류를 이해한다.
- **신용구간(credible interval)**, **HDI(Highest Density Interval)**, **ROPE(Region of Practical Equivalence)**의 차이를 안다.
- p-value 와 사후확률의 해석상 차이를 비교한다.
- 의사결정(decision rule)을 사후확률 기반으로 설계할 수 있다.
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

cells.append(md("""## 1. 자료 준비

Notebook 00과 동일한 시나리오(10문항, 두 집단 300명씩)를 사용합니다.
- 문항 5: $\\Delta b = +0.8$ (강한 DIF)
- 문항 8: $\\Delta b = -0.4$ (약한 DIF)
- 그 외: DIF 없음
"""))

cells.append(code("""data = simulate.scenario_intro_10items(seed=2026)
print(f"N = {data.N},  J = {data.J}")

fit = models.fit_rasch_dif(
    Y=data.Y, group=data.group, backend=BACKEND,
    n_chains=4, n_warmup=500, n_samples=1500,
    prior_sigma_delta=1.0, seed=2026,
)
delta = fit["samples"]["delta"].reshape(-1, data.J)
print(f"Posterior shape: {delta.shape}")
"""))

cells.append(md("""## 2. 사후분포에서 직접 얻을 수 있는 진술들

베이지안에서는 다음과 같은 **확률적 진술**을 자연스럽게 만들 수 있습니다.

| 진술 형태 | 수식 | 해석 |
|---|---|---|
| 신용구간 | $P(\\Delta b \\in [L, U] \\mid \\text{data}) = 0.95$ | "Δb가 [L, U] 안에 있을 확률이 95%" |
| HDI | 동일 확률, 최단 길이 | 가장 가능성 높은 95% 영역 |
| 부호 확률 | $P(\\Delta b > 0 \\mid \\text{data})$ | DIF의 방향에 대한 확신 |
| 실질 효과 | $P(|\\Delta b| > \\delta \\mid \\text{data})$ | 의미 있는 DIF의 확률 |
| ROPE 외부 | $P(\\Delta b \\notin [-\\epsilon, +\\epsilon] \\mid \\text{data})$ | 실질적 0과의 구분 |
"""))

cells.append(code("""# 모든 문항에 대한 다양한 사후 통계
rows = []
for j in range(data.J):
    s = delta[:, j]
    rows.append(dict(
        Item=j+1,
        truth=data.delta_b_true[j],
        post_mean=s.mean(),
        post_sd=s.std(),
        ci_lo=np.quantile(s, 0.025),
        ci_hi=np.quantile(s, 0.975),
        p_positive=np.mean(s > 0),
        p_abs_gt_02=np.mean(np.abs(s) > 0.2),  # outside ROPE
        p_abs_gt_05=np.mean(np.abs(s) > 0.5),  # large effect
    ))
post_df = pd.DataFrame(rows).round(3)
print(post_df.to_string(index=False))
"""))

cells.append(md("""## 3. Forest plot + ROPE

ROPE(Region of Practical Equivalence)는 "실질적으로 0과 다르지 않은 영역"입니다.
- ROPE 안 → 실질적으로 DIF 없음.
- ROPE 밖 → 실질적으로 DIF 있음.
- 일부 안/밖 → 결론 보류, 추가 자료 필요.

본 자료에서는 $\\Delta b$ 의 ROPE를 [-0.2, 0.2]로 설정합니다(약한 효과는 무시 가능).
"""))

cells.append(code("""fig, ax = visualize.plot_dif_forest(
    delta_samples=delta,
    item_labels=[f"Item {j+1}" for j in range(data.J)],
    truth=data.delta_b_true,
    rope=(-0.2, 0.2),
    title="Posterior of Δb_j  with ROPE [-0.2, +0.2]"
)
fig.savefig("../outputs/02_forest_rope.png", dpi=120, bbox_inches="tight")
plt.show()
"""))

cells.append(md("""## 4. 단일 문항 사후 밀도 비교

DIF가 강한 문항(5), 약한 문항(8), 없는 문항(3)의 사후 분포를 비교합니다.
"""))

cells.append(code("""fig, axes = plt.subplots(1, 3, figsize=(13, 4))
items_to_show = [(5, "Strong DIF"), (8, "Weak DIF"), (3, "No DIF")]
for ax, (item, label) in zip(axes, items_to_show):
    j = item - 1
    visualize.plot_posterior_density(
        delta[:, j],
        truth=data.delta_b_true[j],
        rope=(-0.2, 0.2),
        ax=ax,
        title=f"Item {item}  ({label})"
    )
fig.tight_layout()
fig.savefig("../outputs/02_density_compare.png", dpi=120, bbox_inches="tight")
plt.show()
"""))

cells.append(md("""## 5. 의사결정 규칙 (Decision Rules)

베이지안 사후분포를 의사결정으로 연결하는 대표적 규칙들.

**규칙 1 — 부호 확률**

$$
\\text{DIF 양성} \\Leftrightarrow \\max\\big(P(\\Delta b > 0), \\,P(\\Delta b < 0)\\big) > 0.95
$$

**규칙 2 — ROPE 기반 (Kruschke 2018)**

| HDI 와 ROPE 관계 | 결론 |
|---|---|
| 95% HDI 가 ROPE 안에 완전히 들어감 | **DIF 없음 (accept null)** |
| 95% HDI 가 ROPE 밖에 완전히 위치 | **DIF 있음 (reject null)** |
| 일부 겹침 | **결론 보류 (undecided)** |

**규칙 3 — 의미 있는 효과**

$$
\\text{Practical DIF} \\Leftrightarrow P(|\\Delta b| > \\delta_{\\min} \\mid \\text{data}) > 0.95
$$

여기서 $\\delta_{\\min}$ 은 연구자가 정한 최소 실질 효과(예: 0.3).
"""))

cells.append(code("""def kruschke_decision(samples, rope=(-0.2, 0.2), hdi_prob=0.95):
    s = np.asarray(samples)
    alpha = (1 - hdi_prob) / 2
    lo, hi = np.quantile(s, [alpha, 1 - alpha])
    if lo >= rope[0] and hi <= rope[1]:
        return "Accept null (no DIF)"
    if hi < rope[0] or lo > rope[1]:
        return "Reject null (DIF detected)"
    return "Undecided"

print("Kruschke (ROPE+HDI) decisions:")
for j in range(data.J):
    d = kruschke_decision(delta[:, j], rope=(-0.2, 0.2))
    print(f"  Item {j+1:2d} (true Δb={data.delta_b_true[j]:+.2f}): {d}")
"""))

cells.append(md("""## 6. p-value 와 사후확률 비교

같은 자료에 대해 MH p-value와 사후확률 기반 진술을 나란히 둡니다.
**같은 결론이라도 표현력의 차이**가 어떻게 의사결정에 영향을 주는지 비교해보세요.
"""))

cells.append(code("""mh = frequentist.mantel_haenszel_all(data.Y, data.group, n_strata=5)
rows = []
for j in range(data.J):
    s = delta[:, j]
    rows.append(dict(
        Item=j+1,
        truth=data.delta_b_true[j],
        # 빈도주의
        MH_p=mh[j].pvalue,
        MH_class=mh[j].ets_class,
        # 베이지안
        Bayes_post_mean=s.mean().round(3),
        P_DIF_real=np.mean(np.abs(s) > 0.2).round(3),    # ROPE 밖
        P_focal_worse=np.mean(s > 0).round(3),           # focal에 불리할 확률
    ))
cmp_df = pd.DataFrame(rows)
print(cmp_df.to_string(index=False))
"""))

cells.append(md("""## 7. 요약

**사후분포가 제공하는 풍부함**

1. 점추정·구간추정 외에도 **확률 진술**을 직접 만들 수 있다.
2. **ROPE 기반 의사결정**은 효과크기를 의사결정에 명시적으로 통합한다.
3. **부호 확률**, **실질 효과 확률** 등 다양한 형태의 진술이 가능하다.

**해석상 주의**

- 사후확률은 prior와 모형에 의존 → 사전 민감도 분석 권장.
- ROPE 폭은 도메인 지식으로 정해야 함(자동 산출 X).
- "확률 = 1.0"이라는 진술도 모형이 잘못되었으면 의미 없음.

다음 노트북(**03**)에서는 위계 사전(hierarchical prior)을 통한 **자동 shrinkage**와
**다중검정 문제의 완화**를 다룹니다.
"""))

notebook = {"cells": cells,
            "metadata": {"kernelspec": {"display_name": "Python 3 (ipykernel)", "language": "python", "name": "python3"},
                         "language_info": {"name": "python", "version": "3.11"}},
            "nbformat": 4, "nbformat_minor": 5}
NB_PATH.parent.mkdir(parents=True, exist_ok=True)
NB_PATH.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"Notebook saved: {NB_PATH}")
print(f"Cells: {len(cells)}")
