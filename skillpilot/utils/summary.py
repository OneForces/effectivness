# skillpilot/utils/summary.py
import os, tempfile
from .viz import radar_coverage, heat_coverage

TEMPLATE = """# Executive Summary
**Role/JD summary:** {role}

**Overall fit:** {score}/100

**Strengths:** {strengths}
**Gaps:** {gaps}

**ATS hygiene:** email={has_email}, phone={has_phone}, old_dates={old_dates}
"""

def build_summary_md(role: str, score: int, strengths, gaps, ats, diag_text: str):
    md = TEMPLATE.format(
        role=role or "—",
        score=score,
        strengths=", ".join(strengths or []) or "—",
        gaps=", ".join(gaps or []) or "—",
        has_email=ats["contacts"]["has_email"],
        has_phone=ats["contacts"]["has_phone"],
        old_dates=ats["hygiene"]["old_dates_flag"],
    )
    # парсим coverage для картинок
    skills, marks = [], []
    if "Coverage:" in (diag_text or ""):
        cov = diag_text.split("Coverage:",1)[1].strip()
        for p in [x.strip() for x in cov.split(",")]:
            if ":" in p:
                s, f = p.split(":",1)
                skills.append(s.strip()); marks.append("✅" in f)
    rimg = radar_coverage(skills[:10], marks[:10]) if skills else None
    himg = heat_coverage(skills[:16], marks[:16]) if skills else None
    return md, rimg, himg
