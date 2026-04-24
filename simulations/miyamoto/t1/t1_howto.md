# t1 연구 실행 가이드

## 연구 개요

**제목**: 부분 신용 모형–구조 방정식 모형(PCM-SEM)의 방법론적 우위 검증:
한국어 쓰기 태도 자료 조건에서의 몬테카를로 연구

**목적**: 합산 점수 OLS(CS-OLS), MAP+Laplace 베이지안 PCM-SEM, 잠재 변수 오라클
OLS(LV-OLS)의 3-방향 성능 비교를 몬테카를로 시뮬레이션으로 정량화한다.
아울러 독립 생성 데이터셋 10개에 대한 MCMC 반복 적용으로 결과의 재현 가능성을 확인한다.

**CmdStan 필요 여부**:
- CS-OLS vs LV-OLS 몬테카를로 (`t1_monte_carlo.py`): **불필요** (~5분)
- MAP+Laplace 몬테카를로 (`t1_mc_laplace.py`): **필요** (~90–120분)
- MCMC 10회 반복 (`t1_mcmc_10reps.py`): **필요** (~150분)

---

## 필수 파일 (상위 폴더 miyamoto/ 에 있어야 함)

| 파일 | 내용 | 상태 |
|------|------|------|
| `sem_pcm_v2.stan` | 약한 사전 분포 Stan 모형 | 기존 파일 |
| `sem_pcm_with_prior.stan` | 강한 사전 분포 Stan 모형 | 기존 파일 |

---

## 실행 순서

### 단계 1 — CS-OLS vs LV-OLS 몬테카를로 (필수, CmdStan 불필요, ~5분)

```bash
cd miyamoto/t1
python t1_monte_carlo.py
```

생성 파일:
- `t1_mc_results.csv`  — 반복별 원시 추정치 (CS-OLS, LV-OLS)
- `t1_mc_summary.csv`  — 조건별 편향/RMSE/포함확률/검출력 요약

---

### 단계 2 — MAP+Laplace 몬테카를로 (선택 B, CmdStan 필요, ~90–120분)

단계 1 완료 후 실행. 각 조건 50회 반복.

```bash
python t1_mc_laplace.py
```

옵션 (시간 단축):
```bash
python t1_mc_laplace.py --reps 30      # 30회 반복 (~60분)
python t1_mc_laplace.py --reps 10      # 10회 반복 (빠른 테스트)
python t1_mc_laplace.py --n 86         # N=86 단일 조건만
```

생성 파일:
- `t1_mc_laplace_results.csv`  — 반복별 원시 추정치 (CS-OLS, MAP+Laplace, LV-OLS)
- `t1_mc_laplace_summary.csv`  — 조건별 편향/RMSE/포함확률/검출력 요약 (3-방향)

---

### 단계 3 — MCMC 10회 반복 분석 (선택 C, CmdStan 필요, ~150분)

단계 1 완료 후 독립적으로 실행 가능 (단계 2와 병행 가능).
10개 독립 데이터셋(N=86, medium 효과 크기)에 각각 4체인 MCMC 적용.

```bash
python t1_mcmc_10reps.py
```

빠른 테스트 (3회, 2체인):
```bash
python t1_mcmc_10reps.py --reps 3 --chains 2
```

생성 파일:
- `t1_mcmc_10reps.csv`  — 반복별·파라미터별 사후 요약 (mean, SD, 95% CrI, Rhat, ESS)

---

### 단계 4 — 그림 생성 (단계 1 완료 후 실행 가능)

```bash
python t1_figures.py
```

생성 파일:
- `t1_fig_mc_combined.png`      — 그림 1: CS-OLS vs LV-OLS 몬테카를로 비교 (4패널)
- `t1_fig_mc_beta2.png`         — 그림 2: β₂ 및 간접 효과 검출력
- `t1_fig_mcmc_example.png`     — 그림 3: MCMC 예시 사후 분포
  (상위 폴더 `ss_mcmc_weakprior_N86.csv`, `ss_mcmc_strongprior_N86.csv` 사용;
   파일 없으면 그림 3 생략)
- `t1_fig_3way_compare.png`     — 그림 4: 3-방향 비교 (단계 2 완료 후)
  (t1_mc_laplace_summary.csv 필요)
- `t1_fig_mcmc_10reps.png`      — 그림 5: MCMC 10회 반복 에러바 플롯
  (t1_mcmc_10reps.csv 필요)
- `t1_fig_mcmc_violin.png`      — 그림 6: MCMC 10회 반복 바이올린 플롯
  (t1_mcmc_10reps.csv 필요)

---

### 단계 5 — (선택) MCMC 예시 실행

단계 4에서 그림 3이 생성되지 않은 경우에만 필요.

상위 폴더에서:
```bash
cd miyamoto
python ss_run_with_prior.py
```

완료 후 t1/ 에서 다시:
```bash
python t1_figures.py
```

---

### 단계 6 — LaTeX 컴파일

```bash
cd miyamoto/t1
xelatex t1_paper.tex
xelatex t1_paper.tex
```

---

## 권장 실행 시나리오

### 최소 실행 (CmdStan 불필요, ~15분)
```
단계 1 → 단계 4 → 단계 6
```
그림 1, 2 생성 + 논문 컴파일. 그림 3, 4, 5, 6은 생략.

### 완전 실행 (CmdStan 필요, ~5–6시간)
```
단계 1 → [단계 2 병행 단계 3] → 단계 4 → 단계 6
```
모든 그림 생성. 단계 2와 단계 3은 별도 터미널에서 병행 실행 가능.

---

## 생성 파일 목록 (최종)

| 파일 | 설명 | CmdStan |
|------|------|---------|
| `t1_monte_carlo.py` | CS-OLS vs LV-OLS 몬테카를로 스크립트 | 불필요 |
| `t1_mc_laplace.py` | MAP+Laplace 포함 3-방향 몬테카를로 스크립트 | **필요** |
| `t1_mcmc_10reps.py` | MCMC 10회 반복 분석 스크립트 | **필요** |
| `t1_figures.py` | 그림 생성 스크립트 (그림 1–6) | 불필요 |
| `t1_mc_results.csv` | CS-OLS/LV-OLS MC 원시 결과 (단계 1 생성) | — |
| `t1_mc_summary.csv` | CS-OLS/LV-OLS MC 요약 결과 (단계 1 생성) | — |
| `t1_mc_laplace_results.csv` | 3-방향 MC 원시 결과 (단계 2 생성) | — |
| `t1_mc_laplace_summary.csv` | 3-방향 MC 요약 결과 (단계 2 생성) | — |
| `t1_mcmc_10reps.csv` | MCMC 10회 반복 요약 (단계 3 생성) | — |
| `t1_fig_mc_combined.png` | 그림 1 (단계 4 생성) | — |
| `t1_fig_mc_beta2.png` | 그림 2 (단계 4 생성) | — |
| `t1_fig_mcmc_example.png` | 그림 3 (단계 4 또는 5 생성) | — |
| `t1_fig_3way_compare.png` | 그림 4 (단계 4, 단계 2 필요) | — |
| `t1_fig_mcmc_10reps.png` | 그림 5 (단계 4, 단계 3 필요) | — |
| `t1_fig_mcmc_violin.png` | 그림 6 (단계 4, 단계 3 필요) | — |
| `t1_paper.md` | 논문 마크다운 원본 | — |
| `t1_paper.tex` | 논문 LaTeX | — |
| `t1_paper.pdf` | 최종 PDF (단계 6 생성) | — |
