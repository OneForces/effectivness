from .llm import llm

def _truncate(text: str, limit: int) -> str:
    if not text:
        return ""
    text = text.strip()
    return text if len(text) <= limit else text[:limit] + "…"

def _offline_plan(gaps: list, role: str) -> str:
    g = gaps[:6] if gaps else []
    need = ", ".join(g) if g else "систематизация текущих навыков"
    lines = [
        f"День | Задача (≤60 мин) | Артефакт | Ресурсы",
        f"1 | Разбор JD и чек-лист компетенций | checklist.md | вакансия/JD, заметки",
        f"2 | Мини-репо под роль ({need}) | репозиторий с README | GitHub, шаблон проекта",
        f"3 | 1 практическая задача по {g[0] if g else 'основному стеку'} | ноутбук/скрипт | документация/статья",
        f"4 | Документация/тесты для мини-репо | tests + README обновлены | pytest, md",
        f"5 | Подготовка ответов STAR (3–4 кейса) | draft.md | шпаргалка STAR",
        f"6 | Мок-интервью (самопроверка) | список вопросов/ответов | вопросы из UI",
        f"7 | Итоговый пакет (резюме+cover+план) | zip-пакет | кнопка «Скачать пакет»",
    ]
    return "\n".join(lines)

def make_7day_plan(gaps: list, role_hint: str = ""):
    role = role_hint or "под JD"
    gaps_txt = ", ".join(gaps[:6]) if gaps else "нет явных пробелов"

    system = (
        "Ты ментор по развитию навыков. Всегда отвечай НА РУССКОМ ЯЗЫКЕ. "
        "Составь практичный план на 7 дней: каждая задача ≤ 60 минут, "
        "на каждый день обязателен проверяемый артефакт (что сдаём) и конкретные ресурсы."
    )
    # режем вход — этого достаточно для качественного плана и быстрее на CPU
    # 1500 символов на JD и резюме более чем достаточно
    prompt = f"""Цель: роль «{role}».
Ключевые пробелы: {gaps_txt}.

Контекст (урезанный):
- JD (фрагмент): {_truncate(gaps_txt, 200)}
- Резюме (фрагмент): {_truncate("", 0)}

СФОРМИРУЙ ТАБЛИЦУ (как обычный текст):
День | Задача (≤60 мин) | Артефакт (что сдаём) | Ресурсы (1–2 ссылки/подсказки)
"""
    out = llm(system, prompt, max_tokens=600)

    # если Ollama не ответила/таймаут — вернём копию офлайн-плана
    if isinstance(out, str) and out.startswith("[OLLAMA ERROR]"):
        return _offline_plan(gaps, role)
    return out
