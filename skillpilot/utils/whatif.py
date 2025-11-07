# skillpilot/utils/whatif.py
from typing import List, Tuple
from ..core.scorer import score_fit

def delta_scores(jd: str, resume: str, add_terms: List[str]) -> Tuple[int, List[Tuple[str,int,int]]]:
    """Возвращает базовый score и список (term, base, with_term)."""
    base, _, _, _ = score_fit(jd, resume)
    out = []
    for t in add_terms:
        mod_resume = (resume or "") + f"\nSkills: {t}"
        sc, _, _, _ = score_fit(jd, mod_resume)
        out.append((t, base, sc))
    return base, out
