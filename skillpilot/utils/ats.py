# skillpilot/utils/ats.py
import re
from typing import Dict, Any

_EMAIL = re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}")
_PHONE = re.compile(r"(?:\+?\d[\d\-\s\(\)]{7,}\d)")
_BULLET = re.compile(r"^\s*[\-\*\u2022]\s+", re.M)
_ACTION = re.compile(r"\b(achieved|built|led|designed|implemented|improved|reduced|increased|optimized|launched|migrated|delivered|automated)\b", re.I)
_DATE_NOISE = re.compile(r"\b(201[0-5]|200\d)\b")  # слишком старые даты — повод обновить

def ats_check(text: str) -> Dict[str, Any]:
    bullets = _BULLET.findall(text or "")
    actions = _ACTION.findall(text or "")
    return {
        "contacts": {
            "has_email": bool(_EMAIL.search(text or "")),
            "has_phone": bool(_PHONE.search(text or "")),
        },
        "structure": {
            "bullets_count": len(bullets),
            "action_verbs": len(actions),
            "recommendation": "Используйте буллеты с глаголами действия и цифрами результата (STAR).",
        },
        "hygiene": {
            "old_dates_flag": bool(_DATE_NOISE.search(text or "")),
        }
    }
