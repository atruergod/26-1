"""difbayes — Bayesian DIF Detection Utilities (Rasch 1PL/2PL).

공통 모듈 모음:
  - simulate    : Rasch 1PL/2PL + DIF 응답 자료 생성기
  - visualize   : ICC, 사후분포, shrinkage plot
  - frequentist : MH, 로지스틱 회귀 DIF 검정 (비교 기준선)
  - diagnostics : R-hat, ESS, 사후 요약
  - models      : 백엔드(stan/numpyro) 디스패치
"""

from . import simulate
from . import visualize
from . import frequentist
from . import diagnostics
from . import models

__all__ = ["simulate", "visualize", "frequentist", "diagnostics", "models"]
__version__ = "0.1.0"
