
# ============================================================
# 11.3. 표본 크기 효과 — 30+30 보조 시뮬레이션
# ============================================================
cells.append(md("""### 11.3. 표본 크기 효과 — 30+30 보조 시뮬레이션

지금까지는 $n_{ref} = n_{focal} = 300$ 으로 비교적 충분한 표본을 사용했습니다.
표본이 **10배 작아지면(30+30)** 베이지안 사후분포는 어떻게 변할까요?

**예상되는 변화 (이론)**:
- 사후 표준편차(posterior SD)가 약 $\\sqrt{10} \\approx 3.16$ 배 넓어짐.
- weakly-informative prior의 상대적 영향력이 커져, 추정치가 0 쪽으로 더 축소(shrinkage)됨.
- 95% 신용구간이 ROPE와 자주 겹쳐 **의사결정이 "보류(undecided)"**로 빠지는 경우가 늘어남.
- 같은 자료에 MH를 돌리면 분위(stratum)당 6명 정도라 매우 불안정해짐.

이제 실제로 확인해봅시다. 같은 자료생성과정으로 표본만 줄여 다시 적합합니다.
"""))

cells.append(code("""# 작은 표본 자료 생성 — 진짜 모수는 동일
data_small = simulate.scenario_intro_10items(seed=2026)
import numpy as np
rng = np.random.default_rng(99)
ref_idx_all   = np.where(data_small.group == 0)[0]
focal_idx_all = np.where(data_small.group == 1)[0]
sel_ref   = rng.choice(ref_idx_all,   size=30, replace=False)
sel_focal = rng.choice(focal_idx_all, size=30, replace=False)
sel = np.concatenate([sel_ref, sel_focal])

Y_small      = data_small.Y[sel]
group_small  = data_small.group[sel]
print(f"Small sample: N = {len(sel)}, n_ref = 30, n_focal = 30")
"""))

cells.append(code("""# 베이지안 적합 (30+30)
fit_small = models.fit_rasch_dif(
    Y=Y_small, group=group_small, backend=BACKEND,
    n_chains=4, n_warmup=500, n_samples=1000,
    prior_sigma_delta=1.0, seed=2026,
)
delta_small = fit_small["samples"]["delta"].reshape(-1, data.J)
print(f"Posterior shape (small): {delta_small.shape}")
"""))

cells.append(code("""# 두 사후 분포 나란히 비교
fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharex=True)
visualize.plot_dif_forest(
    delta_samples, truth=data.delta_b_true, rope=(-0.2, 0.2),
    item_labels=[f"Item {j+1}" for j in range(data.J)],
    ax=axes[0],
    title="n_ref = n_focal = 300  (baseline)"
)
visualize.plot_dif_forest(
    delta_small, truth=data.delta_b_true, rope=(-0.2, 0.2),
    item_labels=[f"Item {j+1}" for j in range(data.J)],
    ax=axes[1],
    title="n_ref = n_focal = 30  (10x smaller)"
)
fig.savefig("../outputs/00_smallsample_compare.png", dpi=120, bbox_inches="tight")
plt.show()
"""))

cells.append(code("""# 사후폭(SD) 비교 + 의사결정 변화
def decide(samples, rope=(-0.2, 0.2)):
    lo, hi = np.quantile(samples, [0.025, 0.975])
    if hi < rope[0] or lo > rope[1]:
        return "Reject null (DIF)"
    if lo >= rope[0] and hi <= rope[1]:
        return "Accept null (no DIF)"
    return "Undecided"

rows = []
for j in range(data.J):
    rows.append(dict(
        Item=j+1,
        truth=data.delta_b_true[j],
        sd_300=delta_samples[:, j].std().round(3),
        sd_30=delta_small[:, j].std().round(3),
        ratio=(delta_small[:, j].std() / delta_samples[:, j].std()).round(2),
        decision_300=decide(delta_samples[:, j]),
        decision_30=decide(delta_small[:, j]),
    ))
print(pd.DataFrame(rows).to_string(index=False))
"""))

cells.append(md("""**관찰 가이드**

- `sd_30 / sd_300` 비가 이론치 $\\sqrt{10} \\approx 3.16$ 에 근접해야 합니다.
- 진짜 강한 DIF인 **문항 5**는 30+30에서도 양성 검출 가능성이 있지만,
  95% 신용구간이 훨씬 넓어집니다.
- 진짜 약한 DIF인 **문항 8**은 30+30에서는 사실상 식별 불가 (Undecided).
- DIF가 없는 문항(1, 2, 3, ...)도 30+30에서는 0 근처로 좁혀지지 않고 폭이 ±0.7 안팎.

**핵심 메시지**

> 표본이 작아질수록 베이지안 사후는 **더 정직하게 "모른다"를 표현**합니다.
> 같은 자료에서 MH는 분위가 비어 불안정하지만, 베이지안은 사후분포가 단지 *넓어질 뿐* 작동을 멈추지 않습니다.
> 이것이 **장점 #1(소표본 안정성)**과 **장점 #2(불확실성 정량화)**가 함께 발휘되는 모습입니다.
> 이 주제는 Notebook 01에서 반복 시뮬레이션으로 정밀하게 다룹니다.
"""))

# ============================================================
# 12. MH 비교
# ============================================================
cells.append(md("""## 12. 빈도주의 MH 결과와 비교

같은 자료에 대해 **Mantel-Haenszel** 통계량과 비교합니다.
ETS A/B/C 분류는 다음과 같이 해석합니다:
- **A**: 무시 가능 ($|\\Delta_{MH}| < 1.0$ 또는 비유의)
- **B**: 중간 ($1.0 \\leq |\\Delta_{MH}| < 1.5$ 이고 유의)
- **C**: 큼 ($|\\Delta_{MH}| \\geq 1.5$ 이고 유의)
"""))

cells.append(code("""mh_results = frequentist.mantel_haenszel_all(data.Y, data.group, n_strata=5)

post_mean = delta_samples.mean(axis=0)
post_p_zero = np.array([
    diagnostics.posterior_prob_above_threshold(delta_samples[:, j], 0.2, "two-sided")
    for j in range(data.J)
])

compare_df = pd.DataFrame({
    "Item": np.arange(1, data.J + 1),
    "delta_b_true": data.delta_b_true.round(2),
    "MH delta_MH": [r.delta_mh for r in mh_results],
    "MH p-value": [r.pvalue for r in mh_results],
    "ETS class": [r.ets_class for r in mh_results],
    "Bayes mean": post_mean.round(3),
    "P(|delta|>0.2)": post_p_zero.round(3),
})
print("MH vs Bayesian comparison:")
print(compare_df.round(3).to_string(index=False))
"""))

cells.append(md("""**해석 가이드**

- MH delta_MH 와 Bayes 평균의 부호·크기를 비교. 두 방법이 문항 5에서 가장 큰 신호를 잡아야.
- MH p < 0.05 vs P(|delta|>0.2) > 0.9 같은 베이지안 기준이 어떤 결론으로 이어지는지 살펴봅니다.
- 약한 DIF 문항 8 에서 두 방법이 일치/엇갈리는지가 흥미로운 관찰 지점입니다.

> 💡 **포인트**: 두 방법이 일반적으로 비슷한 결론이지만, 베이지안은 "확률" 형태로 직접 진술 가능.
"""))

# ============================================================
# 13. 요약
# ============================================================
cells.append(md("""## 13. 요약 (Summary)

본 노트북에서 다룬 내용:

1. **DIF 분석의 두 목적** — 검출 + 원인·맥락 해석.
2. **DIF 연구의 5단계 발전사** — Zumbo의 3세대 프레임워크로 매핑.
3. **베이지안 검출의 6대 장점** — 소표본 안정성, 불확실성 정량화, 다중검정 완화, 사전정보 통합, 모형 확장 용이, anchor-free.
4. **편향(bias) vs 영향(impact)** — 4가지 시나리오와 통제(control) 개념.
5. **ICC 시각화** — DIF의 시각적 본질.
6. **첫 베이지안 1PL DIF 적합** — Stan/NumPyro, 사후·신용구간·ROPE.
7. **30+30 보조 시뮬레이션** — 표본 크기와 사후폭의 관계.
8. **MH와의 비교** — 표현력의 차이.

### 다음 노트북

- **Notebook 01** — 소표본·희소집단 반복 시뮬레이션.
- **Notebook 02** — 사후확률 기반 의사결정의 풍부함.
- **Notebook 03** — 위계 사전과 자동 shrinkage.
- **Notebook 04** — Spike-and-slab / Horseshoe prior, anchor-free.
- **부록 A** — 2PL로 non-uniform DIF 확장.
"""))


# ============================================================
# 노트북 메타데이터 & 저장
# ============================================================
notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3 (ipykernel)",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "name": "python",
            "version": "3.11",
            "mimetype": "text/x-python",
            "codemirror_mode": {"name": "ipython", "version": 3},
            "pygments_lexer": "ipython3",
            "nbconvert_exporter": "python",
            "file_extension": ".py",
        },
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

NB_PATH.parent.mkdir(parents=True, exist_ok=True)
NB_PATH.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"Notebook saved: {NB_PATH}")
print(f"Total cells: {len(cells)}")
