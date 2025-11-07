# skillpilot/gen/star.py
from .llm import llm_stream

STAR_SYSTEM = "Ты карьерный редактор резюме. Формируешь краткие, сильные, конкретные буллеты."
STAR_USER_TMPL = """Преобразуй опыт ниже в 3–5 кратких STAR-буллетов на русском.
Требования:
- 1 строка на буллет
- Начинай с сильного глагола (Достиг, Внедрил, Снизил, Ускорил…)
- Укажи метрики/цифры, если возможно
- Фокус на результате и вкладе

Текст опыта:
<<<
{raw}
>>>"""

def starify(raw_text: str, temperature: float = 0.4, max_tokens: int = 400) -> str:
    """Вернёт 3–5 STAR-буллетов. Совместимо с вызовом starify(t) из UI."""
    if not (raw_text or "").strip():
        return "— нет входного текста —"
    acc = []
    try:
        for chunk in llm_stream(
            STAR_SYSTEM,
            STAR_USER_TMPL.format(raw=raw_text),
            temperature=temperature,
            max_tokens=int(max_tokens),
        ):
            if chunk:
                acc.append(chunk)
    except Exception as e:
        return f"[STAR ERROR] {type(e).__name__}: {e}"
    return "".join(acc).strip()
