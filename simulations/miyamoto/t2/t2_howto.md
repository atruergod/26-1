# t2 연구 실행 가이드

## 연구 개요

**제목**: 중국인 유학생의 한국어 쓰기 태도 인과 구조:
베이지안 부분 신용 모형–구조 방정식 모형(PCM-SEM) 분석

**목적**: 실제 설문 데이터(N≈100)를 사용하여 중국인 유학생의
쓰기인식→쓰기반응→수행태도 인과 경로를 Bayesian PCM-SEM으로 규명한다.
시뮬레이션은 모형 사전 보정 및 타당화 도구로만 사용한다.

**CmdStan 필요 여부**: **필요** (단계 3, 4)

---

## 실행 순서

### 단계 1 — 설문 데이터 수집 (사전 작업)

설문지: 주월랑(2022)의 21개 한국어 쓰기 태도 문항 (동일 문항)
대상: 한국 대학교 재학 중국인 유학생 N≈100명
출력 형식: 엑셀 또는 CSV → `t2_data.csv` 로 저장 (아래 형식 참조)

**t2_data.csv 형식**:
```
student_id, gender, y1, y2, y3, y4, y5, y6, y7, y8, y9, y10,
            y11, y12, y13, y14, y15, y16, y17, y18, y19, y20, y21
1, 1, 4, 3, 5, 4, 3, 4, 2, 3, 4, 3, 3, 4, 2, 3, 4, 3, 4, 3, 2, 3, 4
...
```

- `gender`: 여성=1, 남성=0
- `y1–y4`: 쓰기인식 (X) 구인 문항
- `y5–y15`: 쓰기반응 (M) 구인 문항
- `y16–y21`: 수행태도 (Y) 구인 문항
- 응답 범주: 1–5 (Likert 5점)

---

### 단계 2 — 사전 분포 보정 시뮬레이션 (CmdStan 불필요, ~3분)

t1 연구를 먼저 실행하거나, 아래를 독립 실행:

```bash
cd miyamoto/t2
python t2_model_check.py
```

생성 파일:
- `t2_prior_check.png`      — 사전 예측 분포 점검 그림
- `t2_calibration.json`     — PCM 임계값 보정 결과 (t2_analysis.py에서 사용)

---

### 단계 3 — MCMC 본 분석 (CmdStan 필요, ~20–40분)

`t2_data.csv`가 준비된 후 실행:

```bash
python t2_analysis.py
```

생성 파일:
- `t2_mcmc_samples.csv`     — MCMC 사후 샘플 (약한 사전 분포)
- `t2_mcmc_strong_samples.csv` — MCMC 사후 샘플 (강한 사전 분포)
- `t2_results_summary.csv`  — 경로 계수 사후 요약 (mean, SD, 95% CrI)
- `t2_convergence.csv`      — R̂, ESS 수렴 진단표

---

### 단계 4 — 그림 생성 (단계 3 완료 후)

```bash
python t2_figures.py
```

생성 파일:
- `t2_fig_posterior.png`    — 그림 1: 경로 계수 사후 분포
- `t2_fig_path_estimates.png` — 그림 2: 경로 계수 점추정 + 95% CrI
- `t2_fig_mediation.png`    — 그림 3: 간접 효과 사후 분포 및 매개 비율

---

### 단계 5 — 논문 완성 및 LaTeX 컴파일

단계 3–4 결과를 `t2_paper.md`의 [결과], [논의], [결론] 플레이스홀더에 채워 넣은 후:

```bash
xelatex t2_paper.tex
xelatex t2_paper.tex
```

---

## 생성 파일 목록 (최종)

| 파일 | 설명 |
|------|------|
| `t2_data.csv` | 실측 설문 데이터 (연구자 수집) |
| `t2_model_check.py` | 사전 예측 점검 + 보정 스크립트 |
| `t2_analysis.py` | MCMC 본 분석 스크립트 |
| `t2_figures.py` | 그림 생성 스크립트 |
| `t2_calibration.json` | 임계값 보정값 (단계 2 생성) |
| `t2_mcmc_samples.csv` | 약한 사전 MCMC 샘플 (단계 3 생성) |
| `t2_mcmc_strong_samples.csv` | 강한 사전 MCMC 샘플 (단계 3 생성) |
| `t2_results_summary.csv` | 결과 요약 (단계 3 생성) |
| `t2_fig_posterior.png` | 그림 1 (단계 4 생성) |
| `t2_fig_path_estimates.png` | 그림 2 (단계 4 생성) |
| `t2_fig_mediation.png` | 그림 3 (단계 4 생성) |
| `t2_paper.md` | 논문 마크다운 원본 |
| `t2_paper.tex` | 논문 LaTeX |
| `t2_paper.pdf` | 최종 PDF (단계 5 생성) |
