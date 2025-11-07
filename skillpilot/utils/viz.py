# skillpilot/utils/viz.py
import os, tempfile, math
from typing import List, Tuple
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def _norm(v: float) -> float:
    v = max(0.0, min(100.0, float(v)))
    return v

def radar_coverage(skills: List[str], marks: List[bool], title: str="Покрытие навыков") -> str:
    """
    skills: названия (<=10 желательно)
    marks: True/False наличие
    Рисуем радар: 100 для True, 30 для False.
    """
    N = len(skills) or 1
    vals = [100 if m else 30 for m in marks]
    angles = [n / float(N) * 2 * math.pi for n in range(N)]
    vals += vals[:1]; angles += angles[:1]  # замкнуть

    fig = plt.figure(figsize=(5,5))
    ax = plt.subplot(111, polar=True)
    ax.set_theta_offset(math.pi / 2); ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1]); ax.set_xticklabels(skills, fontsize=9)
    ax.set_rlabel_position(0); ax.set_yticks([20,40,60,80,100]); ax.set_ylim(0,100)
    ax.plot(angles, vals); ax.fill(angles, vals, alpha=0.15)
    ax.set_title(title)
    out = os.path.join(tempfile.mkdtemp(prefix="skillpilot_viz_"), "radar.png")
    plt.tight_layout(); fig.savefig(out, dpi=160); plt.close(fig)
    return out

def heat_coverage(skills: List[str], marks: List[bool], title: str="Heatmap покрытия") -> str:
    import numpy as np
    import matplotlib.pyplot as plt
    data = np.array([[1 if m else 0 for m in marks]])
    fig = plt.figure(figsize=(min(10, max(4, len(skills)*0.6)), 1.6))
    ax = plt.subplot(111)
    im = ax.imshow(data, aspect="auto")
    ax.set_yticks([]); ax.set_xticks(range(len(skills))); ax.set_xticklabels(skills, rotation=45, ha="right", fontsize=9)
    ax.set_title(title)
    for (j,k),v in np.ndenumerate(data):
        ax.text(k, j, "✓" if v==1 else "—", ha="center", va="center", fontsize=9)
    out = os.path.join(tempfile.mkdtemp(prefix="skillpilot_viz_"), "heat.png")
    plt.tight_layout(); fig.savefig(out, dpi=160); plt.close(fig)
    return out
