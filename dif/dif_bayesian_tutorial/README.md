# Bayesian DIF Detection Tutorial (Rasch 1PL 기반)

베이지안(Bayesian) 차별기능문항(Differential Item Functioning, DIF) 검출 학습자료입니다.
Rasch 1PL 모형을 중심으로, 현대적 베이지안 방법론의 주요 장점들을
시뮬레이션과 시각화로 직접 체험할 수 있도록 구성되어 있습니다.

## 학습 목차

| # | 노트북 | 주제 |
|---|--------|------|
| 00 | `00_intro_rasch_dif_icc.ipynb` | DIF 개념, 발전사, ICC 시각화, 첫 Bayesian 적합 |
| 01 | `01_small_sample_stability.ipynb` | 소표본·희소집단에서의 안정성 |
| 02 | `02_uncertainty_quantification.ipynb` | 사후확률 기반 불확실성 정량화 |
| 03 | `03_hierarchical_shrinkage.ipynb` | 위계모형 shrinkage와 다중검정 완화 |
| 04 | `04_spike_slab_horseshoe.ipynb` | Sparsity prior와 anchor-free 검출 |
| A | `appendix_A_2pl_extension.ipynb` | 2PL 확장: non-uniform DIF 검출 |

## 설치

### 1) 기본 (Stan, 모든 OS)
```bash
pip install -r requirements-base.txt
python -c "import cmdstanpy; cmdstanpy.install_cmdstan()"
```

### 2) (선택) NumPyro 추가 — **Mac / Linux 권장, Windows 비권장**
```bash
pip install -r requirements-numpyro.txt
```

Windows에서는 jax 공식 빌드가 제공되지 않으므로 NumPyro 사용이 어렵습니다.
Windows 사용자는 Stan 백엔드만으로 모든 노트북을 실행할 수 있습니다.

## 백엔드 선택

각 노트북 상단의 설정 셀에서 백엔드를 변경할 수 있습니다.

```python
BACKEND = "stan"      # 기본값: 모든 OS
# BACKEND = "numpyro" # Mac / Linux 에서만
```

NumPyro를 지정했더라도 jax가 없으면 자동으로 Stan으로 fallback 됩니다.

## Windows 사용자: UTF-8 인코딩 안내 (중요)

Windows에서 cmdstanpy(Stan 백엔드)를 쓸 때 다음 오류가 자주 발생합니다.

```
UnicodeDecodeError: 'cp949' codec can't decode byte 0x... in position ...
```

원인은 Stan 컴파일러 출력은 UTF-8인데 Windows 기본 인코딩이 cp949(한국어) 또는 cp1252이기 때문입니다.

**해결 방법** — 다음 중 하나:

1. **(권장)** Jupyter를 UTF-8 모드로 시작하세요. 명령 프롬프트에서:
   ```
   python -X utf8 -m jupyter notebook
   ```
   또는 환경 변수를 먼저 지정:
   ```
   set PYTHONUTF8=1
   jupyter notebook
   ```
2. **(영구)** 시스템 환경 변수 `PYTHONUTF8=1`을 추가. 재로그인 후 모든 Python 세션에 적용됨.
3. **(미봉책)** Notebook 00 상단의 자동 설정 셀이 일부 상황을 처리하지만, 이미 cp949로 시작된
   Jupyter에서는 완전 해결이 어려우므로 1번 방법을 우선 사용하세요.

Mac/Linux 사용자는 기본이 UTF-8이므로 이 작업이 필요하지 않습니다.

## 폴더 구조

```
dif_bayesian_tutorial/
├── README.md
├── requirements-base.txt
├── requirements-numpyro.txt
├── difbayes/                # 공통 파이썬 모듈
│   ├── simulate.py          # Rasch 1PL/2PL + DIF 데이터 생성
│   ├── visualize.py         # ICC, posterior, shrinkage plot
│   ├── frequentist.py       # MH, logistic regression (비교 기준선)
│   ├── diagnostics.py       # R-hat, ESS, 진단 요약
│   └── models.py            # 백엔드 디스패치 (stan / numpyro)
├── models/
│   ├── stan/                # .stan 모형 파일
│   └── numpyro/             # numpyro 모형 정의
├── notebooks/               # 메인 학습 노트북
└── outputs/                 # 그림/시뮬결과 저장
```

## 참고 문헌 (Selected References)

- Holland, P. W., & Thayer, D. T. (1988). Differential item performance and the Mantel-Haenszel procedure. *Test Validity*.
- Camilli, G., & Shepard, L. A. (1994). *Methods for Identifying Biased Test Items*.
- Fox, J.-P. (2010). *Bayesian Item Response Modeling*.
- Soares, T. M., Gonçalves, F. B., & Gamerman, D. (2009). An integrated Bayesian model for DIF analysis. *Journal of Educational and Behavioral Statistics*.
- Frederickx, S., Tuerlinckx, F., De Boeck, P., & Magis, D. (2010). RIM: A random item mixture model to detect DIF. *Journal of Educational Measurement*.

## 라이선스

교육 목적의 자유로운 사용·수정·재배포를 허용합니다.
