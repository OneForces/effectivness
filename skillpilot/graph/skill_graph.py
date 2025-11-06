import networkx as nx
from ..core.extractor import extract_keywords

# новый импорт для PNG-рендера
import os, tempfile
import matplotlib.pyplot as plt

def demo_graph_reco(resume: str, target_role: str = ""):
    G = _build_graph()
    have = set(extract_keywords(resume, 30))
    aliases = {"sklearn":"scikit-learn"}; have = {aliases.get(s,s) for s in have}
    cand = [v for u,v in G.edges() if u in have and v not in have]
    ranked = sorted({c: G.in_degree(c) for c in cand}.items(), key=lambda x:-x[1])
    top = [s for s,_ in ranked[:5]]
    return "Граф не дал явных рекомендаций." if not top else \
        f"Путь к роли {target_role or 'под JD'}:\n- " + "\n- ".join(top)

def render_graph_png(resume: str, target_role: str = "под JD") -> str:
    """Рисует PNG графа навыков, возвращает путь к файлу."""
    G = _build_graph()
    have = set(extract_keywords(resume, 30))
    aliases = {"sklearn":"scikit-learn"}; have = {aliases.get(s,s) for s in have}
    cand = [v for u,v in G.edges() if u in have and v not in have]
    recs = set(cand)

    pos = nx.spring_layout(G, seed=42)
    plt.figure(figsize=(8, 6))
    # базовые узлы/рёбра
    nx.draw_networkx_edges(G, pos, alpha=0.25)
    nx.draw_networkx_labels(G, pos, font_size=8)

    # подсветка: имеющиеся и рекомендованные
    nx.draw_networkx_nodes(G, pos, nodelist=list(G.nodes()), node_size=120, alpha=0.4)
    if have:
        nx.draw_networkx_nodes(G, pos, nodelist=list(have), node_size=220)
    if recs:
        nx.draw_networkx_nodes(G, pos, nodelist=list(recs), node_size=260)

    plt.title(f"SkillGraph → {target_role}")
    plt.axis("off")

    tmpdir = tempfile.mkdtemp(prefix="skillgraph_")
    out_path = os.path.join(tmpdir, "graph.png")
    plt.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close()
    return out_path

def _build_graph():
    G = nx.DiGraph()
    G.add_edges_from([
        ("Python","pandas"),("Python","numpy"),("pandas","feature engineering"),
        ("numpy","linear algebra"),("feature engineering","logistic regression"),
        ("scikit-learn","logistic regression"),("scikit-learn","xgboost"),
        ("matplotlib","reporting"),("sql","ab testing"),("git","team workflow"),
        ("docker","reproducibility"),("airflow","pipelines")
    ])
    return G
