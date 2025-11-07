import os
import tempfile
import networkx as nx
import matplotlib.pyplot as plt
from ..core.extractor import extract_keywords

def _build_graph():
    G = nx.DiGraph()
    G.add_edges_from([
        ("Python", "pandas"), ("Python", "numpy"), ("pandas", "feature engineering"),
        ("numpy", "linear algebra"), ("feature engineering", "logistic regression"),
        ("scikit-learn", "logistic regression"), ("scikit-learn", "xgboost"),
        ("matplotlib", "reporting"), ("sql", "ab testing"), ("git", "team workflow"),
        ("docker", "reproducibility"), ("airflow", "pipelines")
    ])
    return G

def _normalize_to_graph(terms, G):
    """
    Приводит найденные термины к существующим узлам графа (регистронезависимо),
    учитывает простые алиасы вроде sklearn -> scikit-learn.
    """
    aliases = {"sklearn": "scikit-learn"}
    name_map = {str(n).lower(): n for n in G.nodes()}  # 'python' -> 'Python'
    out = set()
    for t in terms:
        key = aliases.get(str(t).lower().strip(), str(t).lower().strip())
        if key in name_map:
            out.add(name_map[key])
    return out

def demo_graph_reco(resume: str, target_role: str = ""):
    G = _build_graph()
    have_raw = set(extract_keywords(resume, 30))
    have = _normalize_to_graph(have_raw, G)

    cand = [v for u, v in G.edges() if u in have and v not in have]
    ranked = sorted({c: G.in_degree(c) for c in cand}.items(), key=lambda x: -x[1])
    top = [s for s, _ in ranked[:5]]

    return "Граф не дал явных рекомендаций." if not top else \
        f"Путь к роли {target_role or 'под JD'}:\n- " + "\n- ".join(top)

def render_graph_png(resume: str, target_role: str = "под JD") -> str:
    """Рисует PNG графа навыков, возвращает путь к файлу."""
    G = _build_graph()

    # нормализуем термины к узлам графа
    have_raw = set(extract_keywords(resume, 30))
    have = _normalize_to_graph(have_raw, G)
    recs = {v for u, v in G.edges() if u in have and v not in have}

    pos = nx.spring_layout(G, seed=42)

    plt.figure(figsize=(8, 6))
    # базовые рёбра/узлы/лейблы
    nx.draw_networkx_edges(G, pos, alpha=0.25)
    nx.draw_networkx_nodes(G, pos, nodelist=list(G.nodes()), node_size=120, alpha=0.35)
    nx.draw_networkx_labels(G, pos, font_size=8)

    # рисуем только те, что реально есть в pos
    have_draw = [n for n in have if n in pos]
    recs_draw = [n for n in recs if n in pos]

    if have_draw:
        nx.draw_networkx_nodes(G, pos, nodelist=have_draw, node_size=240)
    if recs_draw:
        nx.draw_networkx_nodes(G, pos, nodelist=recs_draw, node_size=280)

    plt.title(f"SkillGraph → {target_role}")
    plt.axis("off")

    tmpdir = tempfile.mkdtemp(prefix="skillgraph_")
    out_path = os.path.join(tmpdir, "graph.png")
    plt.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close()
    return out_path
