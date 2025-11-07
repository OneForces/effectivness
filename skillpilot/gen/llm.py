# skillpilot/gen/llm.py
import os
from typing import Optional, Generator, Iterable

from ..config import LLM_BACKEND, OPENAI_API_KEY, OPENAI_MODEL
from .llm_ollama import (
    chat as ollama_chat,
    chat_stream as ollama_chat_stream,
    is_available as ollama_up,
)


def _norm(s: Optional[str]) -> str:
    return "" if s is None else str(s)


def _openai_client():
    """
    Возвращает OpenAI-клиент с учётом кастомного BASE_URL (совместимые эндпоинты).
    """
    base_url = os.getenv("OPENAI_BASE_URL") or os.getenv("OPENAI_API_BASE")
    from openai import OpenAI  # импорт тут, чтобы не тянуть либу без надобности
    return OpenAI(api_key=OPENAI_API_KEY, base_url=base_url) if base_url else OpenAI(api_key=OPENAI_API_KEY)


def _extract_openai_chunk(event) -> Optional[str]:
    """
    Универсально вытаскивает контент из стрим-чанка OpenAI (новый SDK / словари / совместимые эндпоинты).
    Порядок проверки:
      1) choices[0].delta.content
      2) choices[0].message.content
      3) choices[0].text  (совместимые API)
    """
    try:
        choices = getattr(event, "choices", None)
        if choices is None and isinstance(event, dict):
            choices = event.get("choices")
        if not choices:
            return None

        ch0 = choices[0]

        # delta.content
        delta = getattr(ch0, "delta", None)
        if delta is None and isinstance(ch0, dict):
            delta = ch0.get("delta")
        if delta:
            content = getattr(delta, "content", None)
            if content is None and isinstance(delta, dict):
                content = delta.get("content")
            if content:
                return content

        # message.content
        msg = getattr(ch0, "message", None)
        if msg is None and isinstance(ch0, dict):
            msg = ch0.get("message")
        if msg:
            content = getattr(msg, "content", None)
            if content is None and isinstance(msg, dict):
                content = msg.get("content")
            if content:
                return content

        # text (совместимые эндпоинты)
        text = getattr(ch0, "text", None)
        if text:
            return text
        if isinstance(ch0, dict) and ch0.get("text"):
            return ch0["text"]

    except Exception:
        return None
    return None


def llm(
    system: Optional[str],
    prompt: Optional[str],
    temperature: float = 0.25,
    max_tokens: int = 800,
    *,
    top_p: Optional[float] = None,
    stop: Optional[Iterable[str]] = None,
) -> str:
    """
    Унифицированный синхронный вызов LLM.

    - backend=ollama  → локальная модель (напр., 'mistral') через .llm_ollama.chat
    - backend=openai  → при наличии OPENAI_API_KEY/OPENAI_MODEL (+ поддержка совместимых BASE_URL)
    - иначе           → OFFLINE-заглушка

    Всегда возвращает str; сетевые ошибки инкапсулируются в текст.
    """
    system = _norm(system)
    prompt = _norm(prompt)
    backend = (LLM_BACKEND or "offline").strip().lower()

    # ---- OLLAMA (локально)
    if backend == "ollama":
        if ollama_up():
            try:
                # НЕ передаём top_p/stop — текущая обёртка их не принимает
                return ollama_chat(
                    system,
                    prompt,
                    temperature=float(temperature),
                    max_tokens=int(max_tokens),
                )
            except Exception as e:
                return f"[OLLAMA ERROR] {type(e).__name__}: {e}"
        return "[OFFLINE] (Ollama недоступна) " + prompt[:500]

    # ---- OPENAI (или совместимый эндпоинт)
    if backend == "openai" and OPENAI_API_KEY:
        try:
            client = _openai_client()
            model = (OPENAI_MODEL or "").strip() or "gpt-4o-mini"

            kwargs = {
                "model": model,
                "temperature": float(temperature),
                "max_tokens": int(max_tokens),
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt},
                ],
            }
            if top_p is not None:
                kwargs["top_p"] = float(top_p)
            if stop:
                kwargs["stop"] = list(stop)

            resp = client.chat.completions.create(**kwargs)
            out = (resp.choices[0].message.content or "").strip()
            return out or "[LLM EMPTY]"
        except Exception as e:
            return f"[LLM ERROR] {type(e).__name__}: {e}"

    # ---- OFFLINE fallback
    return f"[OFFLINE]\n{prompt[:500]}"


def llm_stream(
    system: Optional[str],
    prompt: Optional[str],
    temperature: float = 0.25,
    max_tokens: int = 800,
    *,
    top_p: Optional[float] = None,
    stop: Optional[Iterable[str]] = None,
) -> Generator[str, None, None]:
    """
    Стриминговый вызов LLM. Возвращает генератор, выдающий части ответа.

    - Для Ollama — .llm_ollama.chat_stream(...)
    - Для OpenAI — stream=True и отдаём delta-контент (устойчиво к разным форматам)
    - В offline — единичный кусок с оффлайн-заглушкой
    """
    system = _norm(system)
    prompt = _norm(prompt)
    backend = (LLM_BACKEND or "offline").strip().lower()

    # ---- OLLAMA (локально)
    if backend == "ollama":
        if ollama_up():
            try:
                # НЕ передаём top_p/stop — текущая обёртка их не принимает
                for chunk in ollama_chat_stream(
                    system,
                    prompt,
                    temperature=float(temperature),
                    max_tokens=int(max_tokens),
                ):
                    if chunk:
                        yield chunk
                return
            except Exception as e:
                yield f"[OLLAMA STREAM ERROR] {type(e).__name__}: {e}"
                return
        else:
            yield "[OFFLINE] (Ollama недоступна) " + prompt[:500]
            return

    # ---- OPENAI (или совместимый эндпоинт)
    if backend == "openai" and OPENAI_API_KEY:
        try:
            client = _openai_client()
            model = (OPENAI_MODEL or "").strip() or "gpt-4o-mini"

            kwargs = {
                "model": model,
                "temperature": float(temperature),
                "max_tokens": int(max_tokens),
                "stream": True,  # ключ
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt},
                ],
            }
            if top_p is not None:
                kwargs["top_p"] = float(top_p)
            if stop:
                kwargs["stop"] = list(stop)

            stream = client.chat.completions.create(**kwargs)

            for event in stream:
                chunk = _extract_openai_chunk(event)
                if chunk:
                    yield chunk
            return
        except Exception as e:
            yield f"[LLM STREAM ERROR] {type(e).__name__}: {e}"
            return

    # ---- OFFLINE fallback
    yield f"[OFFLINE]\n{prompt[:500]}"
