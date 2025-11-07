import time
import requests
from ..config import (
    OLLAMA_HOST, OLLAMA_MODEL,
    OLLAMA_TIMEOUT, OLLAMA_KEEP_ALIVE, OLLAMA_NUM_CTX
)

def _url(path: str) -> str:
    return f"{OLLAMA_HOST}{path}"

def is_available(timeout: int = 3) -> bool:
    try:
        r = requests.get(_url("/api/tags"), timeout=timeout)
        return r.ok
    except Exception:
        return False

def chat(system: str, prompt: str, temperature: float = 0.25, max_tokens: int = 800) -> str:
    """
    Нестримйнг, с ретраями и увеличенным таймаутом.
    """
    payload = {
        "model": OLLAMA_MODEL,
        "messages": [
            {"role": "system", "content": system or ""},
            {"role": "user", "content": prompt or ""},
        ],
        "options": {
            "temperature": float(temperature),
            "num_ctx": int(OLLAMA_NUM_CTX),
        },
        "stream": False,
        "keep_alive": OLLAMA_KEEP_ALIVE,
    }

    # до 3 попыток с экспоненциальным бэкоффом
    last_err = None
    for attempt in range(3):
        try:
            r = requests.post(_url("/api/chat"), json=payload, timeout=OLLAMA_TIMEOUT)
            r.raise_for_status()
            data = r.json()
            msg = (data.get("message") or {}).get("content")
            if isinstance(msg, str) and msg.strip():
                return msg.strip()
            if isinstance(data.get("response"), str):
                return data["response"].strip()
            return "[OLLAMA ERROR] Unexpected response format"
        except Exception as e:
            last_err = e
            # после 1-й попытки попробуем «разбудить» модель
            if attempt == 0:
                try:
                    requests.get(_url("/api/tags"), timeout=5)
                except Exception:
                    pass
            time.sleep(2 ** attempt)  # 1s, 2s
    return f"[OLLAMA ERROR] {type(last_err).__name__}: {last_err}"
