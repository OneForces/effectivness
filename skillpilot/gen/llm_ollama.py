# skillpilot/gen/llm_ollama.py
import time
import json
import requests
from typing import Generator, Optional, Iterable

from ..config import (
    OLLAMA_HOST, OLLAMA_MODEL,
    OLLAMA_TIMEOUT, OLLAMA_KEEP_ALIVE, OLLAMA_NUM_CTX
)

UA = "skillpilot/ollama-client"
_RETRIES = 3          # базовое число попыток для нестримовых вызовов
_STREAM_RETRIES = 2   # число попыток для стрима


def _url(path: str) -> str:
    return f"{OLLAMA_HOST}{path}"


def is_available(timeout: int = 3) -> bool:
    try:
        r = requests.get(_url("/api/tags"), timeout=timeout, headers={"User-Agent": UA})
        return r.ok
    except Exception:
        return False


def _wake() -> None:
    """Лёгкий пинг /api/tags, чтобы «разбудить» демон/модель."""
    try:
        requests.get(_url("/api/tags"), timeout=5, headers={"User-Agent": UA})
    except Exception:
        pass


def _payload(
    system: Optional[str],
    prompt: Optional[str],
    temperature: float,
    max_tokens: int,
    stream: bool,
    top_p: Optional[float] = None,
    stop: Optional[Iterable[str]] = None,
) -> dict:
    opts: dict = {
        "temperature": float(temperature),
        "num_ctx": int(OLLAMA_NUM_CTX),
        "num_predict": int(max_tokens),
    }
    if top_p is not None:
        opts["top_p"] = float(top_p)
    if stop:
        # Ollama принимает строку или массив строк
        opts["stop"] = list(stop)

    return {
        "model": OLLAMA_MODEL,
        "messages": [
            {"role": "system", "content": (system or "")},
            {"role": "user",   "content": (prompt or "")},
        ],
        "options": opts,
        "stream": stream,
        "keep_alive": OLLAMA_KEEP_ALIVE,
    }


def _extract_text(obj: dict) -> str:
    """
    Унифицируем разные форматы ответов Ollama:
    - chat: {"message":{"role":"assistant","content":"..."},"done":...}
    - generate: {"response":"...","done":...}
    """
    msg = (obj.get("message") or {}).get("content")
    if isinstance(msg, str) and msg:
        return msg
    resp = obj.get("response")
    if isinstance(resp, str) and resp:
        return resp
    return ""


def chat(
    system: str,
    prompt: str,
    temperature: float = 0.25,
    max_tokens: int = 800,
    *,
    top_p: Optional[float] = None,
    stop: Optional[Iterable[str]] = None,
) -> str:
    """
    Нестримйнг, с ретраями и экспоненциальным бэкоффом.
    Возвращает финальный ответ целиком.
    """
    payload = _payload(system, prompt, temperature, max_tokens, stream=False, top_p=top_p, stop=stop)

    last_err = None
    for attempt in range(_RETRIES):
        try:
            r = requests.post(
                _url("/api/chat"),
                json=payload,
                timeout=OLLAMA_TIMEOUT,
                headers={"User-Agent": UA},
            )
            r.raise_for_status()
            data = r.json()
            text = _extract_text(data).strip()
            return text if text else "[OLLAMA ERROR] Unexpected response format"
        except Exception as e:
            last_err = e
            if attempt == 0:
                _wake()
            time.sleep(2 ** attempt)  # 1s, 2s, 4s
    return f"[OLLAMA ERROR] {type(last_err).__name__}: {last_err}"


def chat_stream(
    system: str,
    prompt: str,
    temperature: float = 0.25,
    max_tokens: int = 800,
    *,
    top_p: Optional[float] = None,
    stop: Optional[Iterable[str]] = None,
) -> Generator[str, None, None]:
    """
    Стриминг токенов — генератор возвращает кусочки текста по мере прихода.
    С ретраями: если первая попытка не удалась, «будим» демон и повторяем.
    """
    payload = _payload(system, prompt, temperature, max_tokens, stream=True, top_p=top_p, stop=stop)

    last_err = None
    for attempt in range(_STREAM_RETRIES):
        try:
            # timeout=None для stream: держим соединение сколько нужно;
            # при желании можно заменить на большой таймаут (например, 600).
            with requests.post(
                _url("/api/chat"),
                json=payload,
                stream=True,
                timeout=None,
                headers={"User-Agent": UA},
            ) as r:
                r.raise_for_status()
                for raw in r.iter_lines(decode_unicode=True):
                    if not raw:
                        continue
                    try:
                        obj = json.loads(raw)
                    except Exception:
                        # иногда приходят служебные строки — пропускаем
                        continue

                    chunk = _extract_text(obj)
                    if chunk:
                        # Ollama обычно шлёт дельты — отдаём их как есть
                        yield chunk

                    if obj.get("done"):
                        return
            # если вышли из with без done — завершаем
            return
        except Exception as e:
            last_err = e
            if attempt == 0:
                _wake()
            time.sleep(1)

    # если не удалось — финальное сообщение-объяснение
    yield f"[OLLAMA STREAM ERROR] {type(last_err).__name__}: {last_err}"


# --- совместимость со старым импортом ---
# ранее модуль экспортировал функцию с именем stream_chat
stream_chat = chat_stream
