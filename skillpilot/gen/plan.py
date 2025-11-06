from .llm import llm
def make_7day_plan(gaps: list, role_hint: str = ""):
    gaps_txt = ", ".join(gaps[:6]) if gaps else "нет явных пробелов"
    system = "Ментор. План на 7 дней, задачи ≤60 минут, с артефактами."
    prompt = f"""Цель: роль '{role_hint or "под JD"}'. Ключевые пробелы: {gaps_txt}.
Сделай таблицу: День | Задача | Артефакт | Ресурсы."""
    return llm(system, prompt, max_tokens=700)
