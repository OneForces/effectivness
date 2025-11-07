import os
import time
from typing import Optional

def export_md(dir_path: str, name: str, content: str) -> str:
    os.makedirs(dir_path, exist_ok=True)
    path = os.path.join(dir_path, f"{name}.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write(content.strip() + "\n")
    return path

def export_pdf(dir_path: str, name: str, content: str, title: Optional[str] = None) -> str:
    """
    Простой PDF без внешних системных либ (ReportLab).
    """
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas
    from reportlab.lib.units import mm

    os.makedirs(dir_path, exist_ok=True)
    path = os.path.join(dir_path, f"{name}.pdf")
    c = canvas.Canvas(path, pagesize=A4)
    width, height = A4
    x, y = 20*mm, height - 20*mm

    if title:
        c.setFont("Helvetica-Bold", 14)
        c.drawString(x, y, title)
        y -= 10*mm

    c.setFont("Helvetica", 10)
    # примитивная разбивка по строкам
    for para in content.splitlines():
        for line in _wrap_line(para, max_chars=95):
            if y < 20*mm:
                c.showPage()
                y = height - 20*mm
                c.setFont("Helvetica", 10)
            c.drawString(x, y, line)
            y -= 5*mm
        y -= 3*mm
    c.showPage()
    c.save()
    return path

def _wrap_line(s: str, max_chars: int):
    # очень простая разбивка по словам
    words = s.split()
    cur = []
    ln = 0
    for w in words:
        if ln + len(w) + (1 if cur else 0) > max_chars:
            yield " ".join(cur)
            cur = [w]
            ln = len(w)
        else:
            cur.append(w)
            ln += len(w) + (1 if cur[:-1] else 0)
    if cur:
        yield " ".join(cur)
