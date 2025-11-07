import re
import yake

# Языковая эвристика по кириллице (учтём Ё/ё)
CYRIL = re.compile(r"[А-Яа-яЁё]")

def detect_lang(text: str) -> str:
    return "ru" if CYRIL.search(text or "") else "en"


# Базовые стоп-слова и «шум» (минимальные, чтобы не выкидывать полезные термины)
STOP_COMMON = {
    "-", "+", "#", ".", ",", ";", ":", "/", "\\", "|", "&",
    "etc", "etc.", "and", "or", "vs", "plus",
    "junior", "middle", "senior", "lead", "jr", "sr",
    "stack", "project", "product", "team",
}

STOP_RU = {
    "опыт", "задачи", "обязанности", "обязанность", "требования", "требование",
    "ответственность", "ответственности", "умение", "навыки", "навык",
    "работа", "работать", "команда", "команды", "продукт", "проекты", "проект",
    "офис", "гибкий", "формат", "условия", "возможность", "участие",
}

STOP_EN = {
    "experience", "responsibility", "responsibilities", "requirements",
    "requirement", "skills", "skill", "ability", "team", "product", "project",
    "work", "office", "flexible", "format", "conditions", "opportunity",
    "participation", "role",
}

# Минимальная длина полезного токена
_MIN_LEN = 2

# Разрешённые символы в терминах (цифры, латиница, кириллица, дефис/плюс/решётка/точка/пробел)
_CLEAN_RE = re.compile(r"[^0-9A-Za-zА-Яа-яЁё\-\+\#\. ]+")

def _normalize_token(t: str) -> str:
    t = (t or "").strip().lower()
    t = _CLEAN_RE.sub("", t)
    t = re.sub(r"\s{2,}", " ", t).strip()
    return t

def _is_meaningful(t: str, lang: str) -> bool:
    if not t or len(t) < _MIN_LEN:
        return False
    if t in STOP_COMMON:
        return False
    if lang == "ru" and t in STOP_RU:
        return False
    if lang == "en" and t in STOP_EN:
        return False
    # односимвольные/только знаки уже отфильтруем регуляркой/длиной
    return True


def extract_keywords(text: str, top_k: int = 20):
    """Возвращает до top_k уникальных нормализованных терминов (в исходном порядке важности)."""
    lang = detect_lang(text)
    kw = yake.KeywordExtractor(lan=lang, n=1, top=max(top_k, 20))
    pairs = kw.extract_keywords(text or "")
    # отсортируем по возрастанию score (у YAKE чем меньше, тем важнее)
    pairs_sorted = sorted(pairs, key=lambda x: x[1])

    out = []
    seen = set()
    for term, _score in pairs_sorted:
        t = _normalize_token(term)
        if not _is_meaningful(t, lang):
            continue
        if t not in seen:
            seen.add(t)
            out.append(t)
        if len(out) >= top_k:
            break

    return out
