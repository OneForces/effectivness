# skillpilot/utils/batch.py
import os, io, zipfile, csv, tempfile, re
from typing import List, Tuple
from ..core.scorer import score_fit

_TXT_RE = re.compile(rb"[\x09\x0A\x0D\x20-\x7E\xA0-\xFF]+")

def _read_file_bytes(fp) -> bytes:
    try:
        with open(fp, "rb") as f: return f.read()
    except Exception:
        return b""

def _guess_text(data: bytes) -> str:
    if not data: return ""
    # быстрый фоллбек без внешних либ: вытаскиваем печатные байты и декодируем в utf-8
    try:
        return data.decode("utf-8", "ignore")
    except Exception:
        m = _TXT_RE.findall(data)
        return b" ".join(m).decode("utf-8", "ignore")

def _iter_zip_texts(zip_path: str):
    with zipfile.ZipFile(zip_path, "r") as z:
        for name in z.namelist():
            if name.endswith("/") or name.startswith("__MACOSX/"): 
                continue
            data = z.read(name)
            yield name, _guess_text(data)

def batch_score(jd_text: str, resumes: List[Tuple[str, str]]):
    """
    resumes: список (display_name, text)
    return: rows(list), csv_path(str), top_zip(None пока не формируем)
    """
    rows = []
    for name, text in resumes:
        score, strengths, gaps, msg = score_fit(jd_text, text)
        rows.append({
            "resume": name, "score": score,
            "strengths": ", ".join(strengths),
            "gaps": ", ".join(gaps),
        })
    rows.sort(key=lambda r: r["score"], reverse=True)

    # экспорт CSV
    outdir = tempfile.mkdtemp(prefix="skillpilot_batch_")
    csv_path = os.path.join(outdir, "batch_scores.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["resume","score","strengths","gaps"])
        w.writeheader(); w.writerows(rows)

    return rows, csv_path

def read_any_to_text(filepath: str) -> str:
    return _guess_text(_read_file_bytes(filepath))
