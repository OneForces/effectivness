import os
from typing import Optional

from ..config import LLM_BACKEND, OPENAI_API_KEY, OPENAI_MODEL
from .llm_ollama import chat as ollama_chat, is_available as ollama_up


def _norm(s: Optional[str]) -> str:
    return "" if s is None else str(s)


def llm(system: Optional[str],
        prompt: Optional[str],
        temperature: float = 0.25,
        max_tokens: int = 800) -> str:
    """
    Унифицированный вызов LLM:

    - backend=ollama  → локальная модель (например, 'mistral'), через .llm_ollama
    - backend=openai  → при наличии OPENAI_API_KEY/OPENAI_MODEL
    - иначе           → OFFLINE-заглушка (без сетевых вызовов)

    Возвращает всегда str. Ошибки не пробрасываются наружу, а упаковываются в текст.
    """
    system = _norm(system)
    prompt = _norm(prompt)
    backend = (LLM_BACKEND or "offline").strip().lower()

    # ---------------- OLLAMA (локально) ----------------
    if backend == "ollama":
        if ollama_up():
            try:
                return ollama_chat(
                    system,
                    prompt,
                    temperature=float(temperature),
                    max_tokens=int(max_tokens),
                )
            except Exception as e:
                return f"[OLLAMA ERROR] {type(e).__name__}: {e}"
        # сервис недоступен — оффлайн фолбэк
        return "[OFFLINE] (Ollama недоступна) " + prompt[:500]

    # ---------------- OPENAI (опционально) ----------------
    if backend == "openai" and OPENAI_API_KEY:
        try:
            # поддержка совместимых эндпоинтов (напр., прокси/локальные совместимые сервисы)
            base_url = os.getenv("OPENAI_BASE_URL") or os.getenv("OPENAI_API_BASE")
            from openai import OpenAI
            client = OpenAI(api_key=OPENAI_API_KEY, base_url=base_url) if base_url else OpenAI(api_key=OPENAI_API_KEY)

            model = (OPENAI_MODEL or "").strip() or "gpt-4o-mini"
            resp = client.chat.completions.create(
                model=model,
                temperature=float(temperature),
                max_tokens=int(max_tokens),
                timeout=60,  # чуть больше времени по умолчанию
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt},
                ],
            )
            out = (resp.choices[0].message.content or "").strip()
            return out or "[LLM EMPTY]"
        except Exception as e:
            return f"[LLM ERROR] {type(e).__name__}: {e}"

    # ---------------- OFFLINE fallback ----------------
    return f"[OFFLINE]\n{prompt[:500]}"
