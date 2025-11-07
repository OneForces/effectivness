from typing import List, Tuple
from sklearn.metrics.pairwise import cosine_similarity
from .extractor import extract_keywords, detect_lang
from .embedder import embed


ALIASES = {
    "sklearn": "scikit-learn",
}


def jaccard(a, b) -> float:
    """Лексическое пересечение множеств (0..1)."""
    a, b = set(a), set(b)
    return 0.0 if not a and not b else len(a & b) / len(a | b)


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def _clamp(v: float, lo: float = 0.0, hi: float = 100.0) -> int:
    return int(max(lo, min(hi, round(v))))


def _normalize_terms(terms: List[str]) -> List[str]:
    out = []
    seen = set()
    for t in terms:
        k = ALIASES.get(t.lower().strip(), t.lower().strip())
        if k not in seen:
            seen.add(k)
            out.append(k)
    return out


def score_fit(jd: str, resume: str) -> Tuple[int, List[str], List[str], str]:
    """
    Возвращает:
      - score: 0..100
      - strengths: пересечение навыков резюме с JD
      - gaps: навыки из JD, которых нет в резюме
      - msg: диагностическая строка (semantic/jaccard + coverage)
    """
    # Ранние проверки
    if not jd.strip() or not resume.strip():
        return 0, [], [], "Нет данных для оценки (пустой JD или резюме)."

    # 1) извлекаем ключевые слова
    jd_kw_raw = extract_keywords(jd, 25)
    cv_kw_raw = extract_keywords(resume, 25)

    # Нормализуем для сравнения (lower + алиасы), но для вывода оставим «красивые» JD-формы
    jd_norm = _normalize_terms(jd_kw_raw)
    cv_norm = _normalize_terms(cv_kw_raw)

    # 2) семантика (устойчиво к ошибкам модели/сети)
    sem = 0.0
    try:
        vec = embed([jd, resume])
        cos = float(cosine_similarity([vec[0]], [vec[1]])[0][0])  # [-1..1]
        sem = _clamp01((cos + 1.0) / 2.0)  # [0..1]
    except Exception:
        sem = 0.0  # фоллбек на лексический скор

    # 3) лексическое перекрытие (по нормализованным терминам)
    jac = jaccard(jd_norm, cv_norm)

    # 4) финальный скор (если семантика недоступна — используем только джаккар)
    score = _clamp((0.7 * sem + 0.3 * jac) * 100) if sem > 0 else _clamp(jac * 100)

    # 5) объяснимость: сильные/пробелы (возвращаем красивые JD-термины)
    cv_set = set(cv_norm)
    # карта нормализованное → исходное из JD (для понятного вывода)
    pretty_map = {t.lower(): src for src in jd_kw_raw for t in [_normalize_terms([src])[0]]}

    strengths_norm = [t for t in cv_norm if t in set(jd_norm)]
    strengths = [pretty_map.get(t, t) for t in strengths_norm][:8]

    gaps_norm = [t for t in jd_norm if t not in cv_set]
    gaps = [pretty_map.get(t, t) for t in gaps_norm][:8]

    # coverage по топ-навыкам JD (используем «красивые» названия)
    coverage_marks = []
    for t_raw in jd_kw_raw[:12]:
        t_norm = _normalize_terms([t_raw])[0]
        coverage_marks.append(f"{t_raw}:{'✅' if t_norm in cv_set else '—'}")

    sem_str = "off" if sem == 0 else f"{sem:.2f}"
    msg = (
        f"Semantic={sem_str}, Overlap={jac:.2f}. Язык JD: {detect_lang(jd)}. "
        f"(JD terms: {len(jd_norm)}, CV terms: {len(cv_norm)})"
        + (f"\nCoverage: {', '.join(coverage_marks)}" if coverage_marks else "")
    )

    return score, strengths, gaps, msg
