# skillpilot/graph/skill_graph.py
import os, time, tempfile
from typing import Tuple, List, Any

# аккуратно тянем зависимости: если нет, используем PIL-заглушку
try:
    import networkx as nx  # type: ignore
except Exception:
    nx = None  # noqa

try:
    import matplotlib
    matplotlib.use("Agg")  # headless
    import matplotlib.pyplot as plt  # type: ignore
except Exception:
    plt = None  # noqa

try:
    from PIL import Image, ImageDraw, ImageFont  # type: ignore
except Exception:
    Image = ImageDraw = ImageFont = None  # noqa

# вытаскиваем ключевые слова тем же экстрактором, что и скорер
from ..core.extractor import extract_keywords


def _tmp_png(name: str = "skillgraph") -> str:
    d = tempfile.mkdtemp(prefix="skillgraph_")
    return os.path.join(d, f"{name}_{int(time.time())}.png")


def _placeholder_png(msg: str) -> str:
    """Фоллбек-изображение, если нет зависимостей."""
    path = _tmp_png("placeholder")
    if Image is None:
        # вообще без PIL: создаём пустой файл, чтобы Gradio не падал
        open(path, "wb").close()
        return path
    img = Image.new("RGB", (900, 500), (247, 249, 252))
    dr = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 20)
    except Exception:
        font = None
    dr.text((24, 24), "SkillGraph", fill=(12, 18, 32), font=font)
    dr.text((24, 70), msg, fill=(32, 46, 80), font=font)
    img.save(path)
    return path


def demo_graph_reco(jd_text: str, cv_text: str) -> Tuple[Any, List[str], List[str]]:
    """
    Возвращает (G, have, recs).
    - G: networkx.Graph, если доступен, иначе простая структура (dict) — но render_graph_png умеет и без него.
    - have: навыки из JD, которые найдены в резюме
    - recs: навыки из JD, которых нет в резюме (рекомендации)
    """
    jd_terms = extract_keywords(jd_text or "", 20)
    cv_terms = extract_keywords(cv_text or "", 20)

    jd_set, cv_set = set(jd_terms), set(cv_terms)
    have = sorted(list(jd_set & cv_set))
    recs = sorted([t for t in jd_terms if t not in cv_set])[:12]

    # если есть networkx — строим граф
    if nx is not None:
        G = nx.Graph()
        center = "Вы"
        role = "Роль (JD)"
        G.add_node(center, kind="person")
        G.add_node(role, kind="role")
        G.add_edge(center, role)

        for s in have:
            G.add_node(s, kind="have")
            G.add_edge(center, s)
        for s in recs:
            G.add_node(s, kind="need")
            G.add_edge(role, s)
        return G, have, recs

    # без networkx — вернём простую структуру
    G = {"center": "Вы", "role": "Роль (JD)", "have": have, "need": recs}
    return G, have, recs


def render_graph_png(G_or_struct: Any, target_role: str = "под JD") -> str:
    """
    Принимает networkx.Graph ИЛИ упрощённую структуру из demo_graph_reco.
    Возвращает путь к PNG. Никогда не возвращает None (даже на фоллбеке).
    """
    # Фоллбек без matplotlib/networkx — рисуем заглушку
    if plt is None or (nx is None and isinstance(G_or_struct, dict) is False):
        return _placeholder_png("Установите зависимости: networkx, matplotlib")

    # Если пришла «структура», преобразуем в граф при наличии networkx
    if nx is not None and isinstance(G_or_struct, dict):
        G = nx.Graph()
        center = G_or_struct.get("center", "Вы")
        role = G_or_struct.get("role", "Роль (JD)")
        G.add_node(center, kind="person")
        G.add_node(role, kind="role")
        G.add_edge(center, role)
        for s in G_or_struct.get("have", []):
            G.add_node(s, kind="have"); G.add_edge(center, s)
        for s in G_or_struct.get("need", []):
            G.add_node(s, kind="need"); G.add_edge(role, s)
    else:
        G = G_or_struct

    # Если тут всё равно нет networkx — окончательный фоллбек
    if nx is None:
        return _placeholder_png("networkx недоступен — показываю заглушку")

    # Рендер через matplotlib
    pos = nx.spring_layout(G, seed=42, k=0.6)
    path = _tmp_png("skillgraph")

    plt.figure(figsize=(9, 5), dpi=120)
    kinds = nx.get_node_attributes(G, "kind")

    # Узлы по типам
    center_nodes = [n for n, k in kinds.items() if k == "person"]
    role_nodes   = [n for n, k in kinds.items() if k == "role"]
    have_nodes   = [n for n, k in kinds.items() if k == "have"]
    need_nodes   = [n for n, k in kinds.items() if k == "need"]

    nx.draw_networkx_edges(G, pos, alpha=0.25, width=1.0)

    if center_nodes:
        nx.draw_networkx_nodes(G, pos, nodelist=center_nodes, node_size=900, node_color="#3B82F6")
    if role_nodes:
        nx.draw_networkx_nodes(G, pos, nodelist=role_nodes, node_size=800, node_color="#10B981")
    if have_nodes:
        nx.draw_networkx_nodes(G, pos, nodelist=have_nodes, node_size=600, node_color="#93C5FD")
    if need_nodes:
        nx.draw_networkx_nodes(G, pos, nodelist=need_nodes, node_size=600, node_color="#FCA5A5")

    # подписи
    labels = {n: (n if len(str(n)) <= 26 else (str(n)[:23] + "…")) for n in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=9)

    plt.title(f"SkillGraph {target_role}", fontsize=12)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    return path
