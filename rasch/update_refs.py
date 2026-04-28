"""Update reference sections in all IRT notebooks with verified real papers."""
import json, os

BASE = os.path.dirname(os.path.abspath(__file__))

# ── Verified reference blocks per notebook ──────────────────────────────────

REF_COMMON_INTERNATIONAL = """### 국제 학술지 (확인된 논문)

1. **Bürkner, P.-C. (2021)**. Bayesian Item Response Modeling in R with brms and Stan. *Journal of Statistical Software, 100*(5), 1–54. https://doi.org/10.18637/jss.v100.i05

2. **Luo, Y., & Jiao, H. (2018)**. Using the Stan Program for Bayesian Item Response Theory. *Educational and Psychological Measurement, 78*(3), 384–408. https://doi.org/10.1177/0013164417693666

3. **Vehtari, A., Gelman, A., Simpson, D., Carpenter, B., & Bürkner, P.-C. (2021)**. Rank-Normalization, Folding, and Localization: An Improved R-hat for Assessing Convergence of MCMC. *Bayesian Analysis, 16*(2), 667–718. https://doi.org/10.1214/20-BA1221

4. **Gelman, A., Vehtari, A., Simpson, D., Margossian, C. C., Carpenter, B., Yao, Y., Kennedy, L., Gabry, J., Bürkner, P.-C., & Modrák, M. (2020)**. Bayesian Workflow. *arXiv:2011.01808*."""

REF_KCI_FOOTER = """
---

### KCI 등재 학술지

다음 논문들은 한국 KCI 포털(www.kci.go.kr)에서 검색하여 확인할 수 있습니다.

6. **한국어 학습자의 쓰기 수행 평가 신뢰도 분석 — 다국면 라쉬 모형을 사용하여** (KCI 논문번호: ART002387352)

7. **학문 목적 한국어 말하기 평가 과제 유형 개발 연구 — 다국면 라쉬 모형과 일반화가능도 이론 적용을 중심으로** (KCI 논문번호: ART001912809)

8. **문항반응이론을 활용한 학문 목적 한국어 말하기 평가 문항 및 채점자 특성 분석** (KCI 논문번호: ART002419170)

---

> **KCI 검색 방법**: www.kci.go.kr → 논문검색
> 추천 검색어: `다국면 라쉬`, `채점자 엄격성`, `MFRM`, `한국어 말하기 평가`"""

REFS = {

'IRT_P2_PCM_Combined.ipynb': """## 6. References / 참고 문헌

### 국제 학술지 (확인된 논문)

1. **Bürkner, P.-C. (2021)**. Bayesian Item Response Modeling in R with brms and Stan. *Journal of Statistical Software, 100*(5), 1–54. https://doi.org/10.18637/jss.v100.i05

2. **Masters, G. N. (1982)**. A Rasch model for partial credit scoring. *Psychometrika, 47*(2), 149–174. *(PCM 원전)*

3. **Luo, Y., & Jiao, H. (2018)**. Using the Stan Program for Bayesian Item Response Theory. *Educational and Psychological Measurement, 78*(3), 384–408. https://doi.org/10.1177/0013164417693666

4. **Kim, J., & Wilson, M. (2020)**. Polytomous Item Explanatory Item Response Theory Models. *Educational and Psychological Measurement, 80*(4), 726–755. https://doi.org/10.1177/0013164419892667

5. **Vehtari, A., Gelman, A., Simpson, D., Carpenter, B., & Bürkner, P.-C. (2021)**. Rank-Normalization, Folding, and Localization: An Improved R-hat for Assessing Convergence of MCMC. *Bayesian Analysis, 16*(2), 667–718. https://doi.org/10.1214/20-BA1221

---

### KCI 등재 학술지

6. **한국어 학습자의 쓰기 수행 평가 신뢰도 분석 — 다국면 라쉬 모형을 사용하여** (KCI 논문번호: ART002387352). KCI 포털(www.kci.go.kr)에서 전문 검색 가능.

7. **문항반응이론을 활용한 학문 목적 한국어 말하기 평가 문항 및 채점자 특성 분석** (KCI 논문번호: ART002419170). KCI 포털에서 검색 가능.

8. **Rasch 모형을 적용한 문항분석 및 차별기능문항 탐색** (KCI 논문번호: ART002889793). KCI 포털에서 검색 가능.

---

> **KCI 검색 방법**: www.kci.go.kr → 논문검색
> 추천 검색어: `부분점수 모형`, `PCM`, `다분 문항`, `문항 반응 이론`, `한국어 평가`""",

'IRT_P2_PCM_Decomposed.ipynb': """## 6. References / 참고 문헌

### 국제 학술지 (확인된 논문)

1. **Bürkner, P.-C. (2021)**. Bayesian Item Response Modeling in R with brms and Stan. *Journal of Statistical Software, 100*(5), 1–54. https://doi.org/10.18637/jss.v100.i05

2. **Masters, G. N. (1982)**. A Rasch model for partial credit scoring. *Psychometrika, 47*(2), 149–174. *(PCM 원전)*

3. **Luo, Y., & Jiao, H. (2018)**. Using the Stan Program for Bayesian Item Response Theory. *Educational and Psychological Measurement, 78*(3), 384–408. https://doi.org/10.1177/0013164417693666

4. **Kim, J., & Wilson, M. (2020)**. Polytomous Item Explanatory Item Response Theory Models. *Educational and Psychological Measurement, 80*(4), 726–755. https://doi.org/10.1177/0013164419892667

5. **De Boeck, P., & Wilson, M. (Eds.) (2004)**. Explanatory Item Response Models: A Generalized Linear and Nonlinear Approach. Springer. *(EIRT 이론 원전)*

---

### KCI 등재 학술지

6. **한국어 학습자의 쓰기 수행 평가 신뢰도 분석 — 다국면 라쉬 모형을 사용하여** (KCI 논문번호: ART002387352). KCI 포털(www.kci.go.kr)에서 전문 검색 가능.

7. **문항반응이론을 활용한 학문 목적 한국어 말하기 평가 문항 및 채점자 특성 분석** (KCI 논문번호: ART002419170). KCI 포털에서 검색 가능.

8. **Rasch 모형을 적용한 문항분석 및 차별기능문항 탐색** (KCI 논문번호: ART002889793). KCI 포털에서 검색 가능.

---

> **KCI 검색 방법**: www.kci.go.kr → 논문검색
> 추천 검색어: `부분점수 모형`, `분해형 PCM`, `문항 반응 이론`, `한국어 평가 타당도`""",

'IRT_P3_MFRM_Model.ipynb': """## 13. References / 참고 문헌

### 국제 학술지 (확인된 논문)

1. **Uto, M., & Ueno, M. (2020)**. A generalized many-facet Rasch model and its Bayesian estimation using Hamiltonian Monte Carlo. *Behaviormetrika, 47*, 469–496. https://doi.org/10.1007/s41237-020-00115-7

2. **Uto, M. (2022)**. A Bayesian many-facet Rasch model with Markov modeling for rater severity drift. *Behavior Research Methods*. https://doi.org/10.3758/s13428-022-01997-z

3. **Bürkner, P.-C. (2021)**. Bayesian Item Response Modeling in R with brms and Stan. *Journal of Statistical Software, 100*(5), 1–54. https://doi.org/10.18637/jss.v100.i05

4. **Luo, Y., & Jiao, H. (2018)**. Using the Stan Program for Bayesian Item Response Theory. *Educational and Psychological Measurement, 78*(3), 384–408. https://doi.org/10.1177/0013164417693666

5. **Vehtari, A., Gelman, A., Simpson, D., Carpenter, B., & Bürkner, P.-C. (2021)**. Rank-Normalization, Folding, and Localization: An Improved R-hat for Assessing Convergence of MCMC. *Bayesian Analysis, 16*(2), 667–718. https://doi.org/10.1214/20-BA1221

---

### KCI 등재 학술지

6. **한국어 학습자의 쓰기 수행 평가 신뢰도 분석 — 다국면 라쉬 모형을 사용하여** (KCI 논문번호: ART002387352). KCI 포털(www.kci.go.kr)에서 전문 검색 가능.

7. **학문 목적 한국어 말하기 평가 과제 유형 개발 연구 — 다국면 라쉬 모형과 일반화가능도 이론 적용을 중심으로** (KCI 논문번호: ART001912809). KCI 포털에서 검색 가능.

8. **문항반응이론을 활용한 학문 목적 한국어 말하기 평가 문항 및 채점자 특성 분석** (KCI 논문번호: ART002419170). KCI 포털에서 검색 가능.

---

> **KCI 검색 방법**: www.kci.go.kr → 논문검색
> 추천 검색어: `다국면 라쉬`, `채점자 엄격성`, `MFRM`, `한국어 말하기 평가`, `FACETS`""",

'IRT_P4_MFPCM_Model.ipynb': """## 13. References / 참고 문헌

### 국제 학술지 (확인된 논문)

1. **Uto, M., & Ueno, M. (2020)**. A generalized many-facet Rasch model and its Bayesian estimation using Hamiltonian Monte Carlo. *Behaviormetrika, 47*, 469–496. https://doi.org/10.1007/s41237-020-00115-7

2. **Uto, M. (2022)**. A multidimensional generalized many-facet Rasch model for rubric-based performance assessment. *Behaviormetrika*. https://doi.org/10.1007/s41237-021-00144-w

3. **Masters, G. N. (1982)**. A Rasch model for partial credit scoring. *Psychometrika, 47*(2), 149–174. *(PCM 원전)*

4. **Bürkner, P.-C. (2021)**. Bayesian Item Response Modeling in R with brms and Stan. *Journal of Statistical Software, 100*(5), 1–54. https://doi.org/10.18637/jss.v100.i05

5. **Vehtari, A., Gelman, A., Simpson, D., Carpenter, B., & Bürkner, P.-C. (2021)**. Rank-Normalization, Folding, and Localization: An Improved R-hat for Assessing Convergence of MCMC. *Bayesian Analysis, 16*(2), 667–718.

---

### KCI 등재 학술지

6. **한국어 학습자의 쓰기 수행 평가 신뢰도 분석 — 다국면 라쉬 모형을 사용하여** (KCI 논문번호: ART002387352). KCI 포털(www.kci.go.kr)에서 전문 검색 가능.

7. **학문 목적 한국어 말하기 평가 과제 유형 개발 연구** (KCI 논문번호: ART001912809). KCI 포털에서 검색 가능.

8. **문항반응이론을 활용한 학문 목적 한국어 말하기 평가 문항 및 채점자 특성 분석** (KCI 논문번호: ART002419170). KCI 포털에서 검색 가능.

---

> **KCI 검색 방법**: www.kci.go.kr → 논문검색
> 추천 검색어: `다국면 라쉬`, `MFPCM`, `문항 단계 난이도`, `한국어 평가`, `채점자 신뢰도`""",

'IRT_P5_PCM_EIRT.ipynb': """## 12. References / 참고 문헌

### 국제 학술지 (확인된 논문)

1. **Kim, J., & Wilson, M. (2020)**. Polytomous Item Explanatory Item Response Theory Models. *Educational and Psychological Measurement, 80*(4), 726–755. https://doi.org/10.1177/0013164419892667

2. **Bürkner, P.-C. (2021)**. Bayesian Item Response Modeling in R with brms and Stan. *Journal of Statistical Software, 100*(5), 1–54. https://doi.org/10.18637/jss.v100.i05

3. **De Boeck, P., & Wilson, M. (Eds.) (2004)**. Explanatory Item Response Models: A Generalized Linear and Nonlinear Approach. Springer. *(EIRT 이론 원전)*

4. **Gelman, A., Vehtari, A., Simpson, D., Margossian, C. C., Carpenter, B., Yao, Y., Kennedy, L., Gabry, J., Bürkner, P.-C., & Modrák, M. (2020)**. Bayesian Workflow. *arXiv:2011.01808*.

5. **Vehtari, A., Gelman, A., Simpson, D., Carpenter, B., & Bürkner, P.-C. (2021)**. Rank-Normalization, Folding, and Localization: An Improved R-hat for Assessing Convergence of MCMC. *Bayesian Analysis, 16*(2), 667–718.

---

### KCI 등재 학술지

6. **한국어 학습자의 쓰기 수행 평가 신뢰도 분석 — 다국면 라쉬 모형을 사용하여** (KCI 논문번호: ART002387352). KCI 포털(www.kci.go.kr)에서 전문 검색 가능.

7. **문항반응이론을 활용한 학문 목적 한국어 말하기 평가 문항 및 채점자 특성 분석** (KCI 논문번호: ART002419170). KCI 포털에서 검색 가능.

8. **Rasch 모형을 적용한 문항분석 및 차별기능문항 탐색** (KCI 논문번호: ART002889793). KCI 포털에서 검색 가능.

---

> **KCI 검색 방법**: www.kci.go.kr → 논문검색
> 추천 검색어: `설명적 IRT`, `LLTM`, `선형 로지스틱 테스트 모형`, `한국어 능력 평가`, `다층 측정 모형`""",

'IRT_P6_DIF_PCM.ipynb': """## 13. References / 참고 문헌

### 국제 학술지 (확인된 논문)

1. **Joo, S.-H., Lee, P., & Stark, S. (2022)**. Bayesian Approaches for Detecting Differential Item Functioning Using the Generalized Graded Unfolding Model. *Applied Psychological Measurement, 46*(2), 98–115. https://doi.org/10.1177/01466216211066606

2. **Bürkner, P.-C. (2021)**. Bayesian Item Response Modeling in R with brms and Stan. *Journal of Statistical Software, 100*(5), 1–54. https://doi.org/10.18637/jss.v100.i05

3. **Masters, G. N. (1982)**. A Rasch model for partial credit scoring. *Psychometrika, 47*(2), 149–174. *(PCM 원전)*

4. **Luo, Y., & Jiao, H. (2018)**. Using the Stan Program for Bayesian Item Response Theory. *Educational and Psychological Measurement, 78*(3), 384–408.

5. **Gelman, A., Vehtari, A., Simpson, D., Margossian, C. C., Carpenter, B., Yao, Y., Kennedy, L., Gabry, J., Bürkner, P.-C., & Modrák, M. (2020)**. Bayesian Workflow. *arXiv:2011.01808*.

---

### KCI 등재 학술지

6. **Rasch 모형을 적용한 문항분석 및 차별기능문항 탐색 — 2021학년도 물리인증제 1-2급을 바탕으로** (KCI 논문번호: ART002889793). KCI 포털(www.kci.go.kr)에서 전문 검색 가능.

7. **문항반응이론을 활용한 학문 목적 한국어 말하기 평가 문항 및 채점자 특성 분석** (KCI 논문번호: ART002419170). KCI 포털에서 검색 가능.

8. **한국어 학습자의 쓰기 수행 평가 신뢰도 분석 — 다국면 라쉬 모형을 사용하여** (KCI 논문번호: ART002387352). KCI 포털에서 검색 가능.

---

> **KCI 검색 방법**: www.kci.go.kr → 논문검색
> 추천 검색어: `차별 문항 기능`, `DIF`, `문항 편파성`, `TOPIK 공정성`, `측정 동등성`""",
}

def find_ref_cell(cells):
    """Find the index of the reference cell."""
    for i, cell in enumerate(cells):
        src = ''.join(cell.get('source', []))
        if 'References' in src or '참고 문헌' in src:
            return i
    return None

def set_ref_cell(cells, idx, new_src):
    """Replace or create reference cell."""
    lines = new_src.split('\n')
    cells[idx] = {"cell_type":"markdown","metadata":{},"source":lines}

updated = []
skipped = []

for nb_name, new_ref in REFS.items():
    path = os.path.join(BASE, nb_name)
    if not os.path.exists(path):
        skipped.append(f"{nb_name}: file not found")
        continue
    try:
        with open(path, encoding='utf-8') as f:
            nb = json.load(f)
        idx = find_ref_cell(nb['cells'])
        if idx is None:
            # Append a new reference cell
            lines = new_ref.split('\n')
            nb['cells'].append({"cell_type":"markdown","metadata":{},"source":lines})
            print(f"{nb_name}: added new reference cell (was missing)")
        else:
            set_ref_cell(nb['cells'], idx, new_ref)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(nb, f, ensure_ascii=False, indent=1)
        # Verify
        with open(path) as f:
            test = json.load(f)
        print(f"{nb_name}: updated OK ({len(test['cells'])} cells)")
        updated.append(nb_name)
    except Exception as e:
        print(f"{nb_name}: ERROR - {e}")
        skipped.append(nb_name)

print(f"\nUpdated: {len(updated)} notebooks")
if skipped:
    print(f"Skipped: {skipped}")
