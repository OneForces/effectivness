import os
import io
import json
import hashlib
import shelve
import threading
from typing import Iterable, List, Sequence, Union

import numpy as np
from sentence_transformers import SentenceTransformer
from ..config import EMB_MODEL

# --------- globals & cache ----------

_MODEL: SentenceTransformer | None = None
_MODEL_LOCK = threading.Lock()
_DB_LOCK = threading.Lock()

CACHE_DIR = os.getenv("SKILLPILOT_CACHE_DIR", os.path.expanduser("~/.cache/skillpilot"))
os.makedirs(CACHE_DIR, exist_ok=True)
CACHE_PATH = os.path.join(CACHE_DIR, "embeddings.db")

# Версионирование формата кэша (на случай будущих изменений)
CACHE_SCHEMA_VER = "v2"  # v2 = per-item, npy-bytes float32


def _get(model_name: str | None = None) -> SentenceTransformer:
    """Ленивая загрузка модели эмбеддингов (один инстанс на процесс)."""
    global _MODEL
    name = model_name or EMB_MODEL
    if _MODEL is None:
        with _MODEL_LOCK:
            if _MODEL is None:
                _MODEL = SentenceTransformer(name)
    return _MODEL


def _key_item(text: str, model_name: str) -> str:
    """Ключ кэша на один текст (лучше для повторного использования)."""
    payload = {"ver": CACHE_SCHEMA_VER, "m": model_name, "t": text}
    return hashlib.sha1(
        json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()


def _arr_to_bytes(arr: np.ndarray) -> bytes:
    """Сериализация ndarray в npy-байты (portable)."""
    bio = io.BytesIO()
    np.save(bio, np.asarray(arr, dtype=np.float32))  # compact
    return bio.getvalue()


def _bytes_to_arr(b: bytes) -> np.ndarray | None:
    try:
        bio = io.BytesIO(b)
        arr = np.load(bio)
        # гарантируем float32 (экономит память/диск, достаточно для косинуса)
        return np.asarray(arr, dtype=np.float32)
    except Exception:
        return None


def _shelve_open():
    # writeback=False чтобы не плодить избыточные pickle-объекты
    return shelve.open(CACHE_PATH, flag="c", writeback=False)


def embed(
    texts: Union[str, Iterable[str]],
    name: str | None = None
) -> np.ndarray:
    """
    Возвращает матрицу эмбеддингов shape=(n, d), L2-нормированных.
    Дисковый кэш (per-item): ~/.cache/skillpilot/embeddings.db

    Поведение:
      • Один раз открывает shelve и читает все совпадения.
      • Считает только «промахи» батчем и дописывает в кэш.
      • Потокобезопасная запись (глобальный _DB_LOCK).
    """
    model_name = name or EMB_MODEL

    # нормализация входа
    if isinstance(texts, str):
        items: List[str] = [texts]
    else:
        items = [t if isinstance(t, str) else str(t) for t in texts]

    n = len(items)
    if n == 0:
        return np.empty((0, 0), dtype=np.float32)

    # попытка загрузить из кэша
    keys = [_key_item(t or "", model_name) for t in items]
    cached: list[np.ndarray | None] = [None] * n
    missing_idx: list[int] = []
    missing_texts: list[str] = []

    with _shelve_open() as db:
        for i, k in enumerate(keys):
            raw = db.get(k)
            if isinstance(raw, (bytes, bytearray)):
                arr = _bytes_to_arr(raw)
                if arr is not None:
                    cached[i] = arr
                    continue
            # промах
            missing_idx.append(i)
            missing_texts.append(items[i])

    # если нужны дорасчёты — считаем батчем
    if missing_idx:
        model = _get(model_name)
        # normalize_embeddings=True вернёт уже L2-нормированные векторы
        vecs: np.ndarray = model.encode(
            missing_texts,
            normalize_embeddings=True,
            show_progress_bar=False
        )
        vecs = np.asarray(vecs, dtype=np.float32)

        # записываем в кэш под lock, но одним открытием
        with _DB_LOCK:
            with _shelve_open() as db:
                for j, i in enumerate(missing_idx):
                    v = vecs[j]
                    cached[i] = v
                    try:
                        db[keys[i]] = _arr_to_bytes(v)
                    except Exception:
                        # если драйвер shelve/dbm капризничает — тихо пропускаем запись;
                        # на функциональность это не влияет
                        pass

    # склейка результата
    # все элементы cached к этому моменту должны быть np.ndarray
    out = np.vstack([np.asarray(v, dtype=np.float32) for v in cached])
    return out
