from sklearn.metrics.pairwise import cosine_similarity
from .extractor import extract_keywords, detect_lang
from .embedder import embed
def jaccard(a,b):
    a,b=set(a),set(b)
    return 0.0 if not a and not b else len(a & b)/len(a | b)
def score_fit(jd: str, resume: str):
    jd_kw = extract_keywords(jd, 25); cv_kw = extract_keywords(resume, 25)
    vec = embed([jd, resume]); sem = float(cosine_similarity([vec[0]],[vec[1]])[0][0])
    jac = jaccard(jd_kw, cv_kw); score = int(round((0.7*sem+0.3*jac)*100))
    strengths = [t for t in cv_kw if t in jd_kw][:8]
    gaps = [t for t in jd_kw if t not in cv_kw][:8]
    msg = f"Semantic={sem:.2f}, Overlap={jac:.2f}. Язык JD: {detect_lang(jd)}."
    return score, strengths, gaps, msg
