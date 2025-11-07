import os
import io
import re
from typing import Optional

def _clean(txt: str) -> str:
    txt = txt.replace("\x00", " ")
    txt = re.sub(r"\r\n?", "\n", txt)
    txt = re.sub(r"\n{3,}", "\n\n", txt)
    return txt.strip()

def read_any(file_path_or_bytes, filename: Optional[str] = None) -> str:
    """
    Принимает путь или bytes + имя файла. Возвращает чистый текст.
    Поддержка: .pdf, .docx, .md, .txt
    """
    name = (filename or "").lower()
    if isinstance(file_path_or_bytes, (bytes, bytearray)):
        b = io.BytesIO(file_path_or_bytes)
    else:
        # path
        with open(file_path_or_bytes, "rb") as f:
            b = io.BytesIO(f.read())

    # Определяем по расширению; pdf/docx требуют бинарь
    if name.endswith(".pdf"):
        try:
            from pypdf import PdfReader
            reader = PdfReader(b)
            pages = [p.extract_text() or "" for p in reader.pages]
            return _clean("\n\n".join(pages))
        except Exception:
            pass

    if name.endswith(".docx"):
        try:
            import docx  # python-docx
            b.seek(0)
            doc = docx.Document(b)
            text = "\n".join(p.text for p in doc.paragraphs)
            return _clean(text)
        except Exception:
            pass

    # markdown / txt
    if name.endswith(".md"):
        try:
            b.seek(0)
            raw = b.read().decode("utf-8", "ignore")
            # очень лёгкая очистка md
            raw = re.sub(r"`{1,3}.*?`{1,3}", "", raw, flags=re.S)
            raw = re.sub(r"^#+\s*", "", raw, flags=re.M)
            return _clean(raw)
        except Exception:
            pass

    # как простой txt (по умолчанию)
    b.seek(0)
    txt = b.read().decode("utf-8", "ignore")
    return _clean(txt)
