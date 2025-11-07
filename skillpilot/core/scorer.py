from typing import List, Tuple
import re
from sklearn.metrics.pairwise import cosine_similarity
from .extractor import extract_keywords, detect_lang
from .embedder import embed


# Больше синонимов/нормализаций для устойчивых сравнений
ALIASES = {
    "sklearn": "scikit-learn",
    "scikit learn": "scikit-learn",
    "tf": "tensorflow",
    "tfidf": "tf-idf",
    "py torch": "pytorch",
    "pytorch": "pytorch",
    "js": "javascript",
    "ts": "typescript",
    "postgres": "postgresql",
    "ms excel": "excel",
    "excel": "excel",
    "nlp": "natural language processing",
    "gpt": "llm",
    "llama": "llm",
    "xgboost": "xgboost",
    "catboost": "catboost",
    "lightgbm": "lightgbm",
    "scipy": "scipy",
    "plt": "matplotlib",
    "git": "git",
    "docker": "docker",
    "k8s": "kubernetes",
    "kubernetes": "kubernetes",
    "sql": "sql",
    "postgresql": "postgresql",
    "mysql": "mysql",
    "nosql": "nosql",
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
        base = (t or "").lower().strip()
        k = ALIASES.get(base, base)
        if k and k not in seen:
            seen.add(k)
            out.append(k)
    return out


# Эвристика: вычленяем «обязательные» навыки из JD по контексту
# Смотрим фразы must/required/обязательно/требуется рядом с терминами
_CRIT_PAT = re.compile(
    r"(must(?:\s*have)?|required|mandatory|обязательн\w*|строго|требуется|необходим\w+)",
    flags=re.IGNORECASE,
)

def _extract_critical_terms(jd_text: str, jd_terms_raw: List[str]) -> List[str]:
    if not jd_text or not jd_terms_raw:
        return []
    text = jd_text.lower()
    # окна вокруг «триггеров»
    crit_spans = [m.span() for m in _CRIT_PAT.finditer(text)]
    if not crit_spans:
        return []
    norm_map = { _normalize_terms([src])[0]: src for src in jd_terms_raw if src.strip() }
    crit_norm = []
    for norm, src in norm_map.items():
        # ищем упоминание исходной формы рядом с триггерами
        # берём +- 120 символов как близость
        for a,b in crit_spans:
            left = max(0, a-120); right = min(len(text), b+120)
            win = text[left:right]
            if src.lower() in win or norm in win:
                crit_norm.append(norm); break
    # если эвристика ничего не нашла, можно подсветить базовые критические (демо-режим)
    if not crit_norm:
        baseline = []
        for t in ["python", "sql"]:
            if t in norm_map: baseline.append(t)
        crit_norm = baseline
    # uniq, preserve order
    seen=set(); out=[]
    for t in crit_norm:
        if t not in seen:
            seen.add(t); out.append(t)
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

    # 4) критичные навыки и штраф
    crit_norm = _extract_critical_terms(jd, jd_kw_raw)
    cv_set = set(cv_norm)
    missing_crit = [t for t in crit_norm if t not in cv_set]
    # мягкий штраф: по 0.1 за навык, но потолок 0.3 (чтобы не «убить» семантику)
    penalty = min(0.3, 0.1 * len(missing_crit))

    # 5) финальный скор:
    # - если семантика есть → 60% семантика + 40% яккард - штраф
    # - иначе только яккард (без штрафа, чтобы не двойной негатив на слабом сигнале)
    if sem > 0:
        raw = max(0.0, (0.6 * sem + 0.4 * jac) - penalty)
        score = _clamp(100 * raw)
    else:
        score = _clamp(100 * jac)

    # 6) объяснимость: сильные/пробелы (возвращаем красивые JD-термины)
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

    # диагностическая строка
    sem_str = "off" if sem == 0 else f"{sem:.2f}"
    lang = detect_lang(jd)
    crit_disp = ", ".join(pretty_map.get(t, t) for t in crit_norm) or "—"
    miss_disp = ", ".join(pretty_map.get(t, t) for t in missing_crit) or "—"

    msg = (
        f"Semantic={sem_str}, Overlap={jac:.2f}, Penalty={penalty:.2f}. "
        f"Язык JD: {lang}. (JD terms: {len(jd_norm)}, CV terms: {len(cv_norm)})"
        + (f"\nCritical: {crit_disp}" if crit_norm else "\nCritical: —")
        + (f"\nMissing critical: {miss_disp}" if missing_crit else "")
        + (f"\nCoverage: {', '.join(coverage_marks)}" if coverage_marks else "")
    )

    return score, strengths, gaps, msg
