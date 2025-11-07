import os
import sys

BASE = os.path.dirname(os.path.dirname(__file__))
if BASE not in sys.path:
    sys.path.append(BASE)

from skillpilot.core.scorer import score_fit


def test_basic():
    jd = "Нужен Python, pandas, numpy, scikit-learn; плюсом Docker."
    cv = "Навыки: Python, pandas, numpy, matplotlib."
    s, strengths, gaps, _ = score_fit(jd, cv)
    assert isinstance(s, int) and 0 <= s <= 100
