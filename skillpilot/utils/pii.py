# skillpilot/utils/pii.py
import re

_EMAIL = re.compile(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b")
_PHONE = re.compile(r"(?:\+?\d[\d\-\s\(\)]{7,}\d)")
# Осторожная эвристика имён (латиница/кириллица), исключаем популярные tech-термины
_NAME  = re.compile(r"\b([А-ЯЁ][а-яё]+(?:\s+[А-ЯЁ][а-яё]+){0,2}|[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2})\b")
_TECH_SAFE = re.compile(r"^(Python|Pandas|NumPy|SQL|Docker|Kubernetes|TensorFlow|PyTorch|LLM|NLP|XGBoost|CatBoost|LightGBM)$", re.I)

def anonymize(text: str) -> str:
    if not text:
        return text
    t = _EMAIL.sub("[email hidden]", text)
    t = _PHONE.sub("[phone hidden]", t)

    def _name_sub(m):
        s = m.group(0)
        return s if _TECH_SAFE.match(s) else "[name]"

    t = _NAME.sub(_name_sub, t)
    return t
