from sentence_transformers import SentenceTransformer
_model = None
def _get(name: str = "all-MiniLM-L6-v2"):
    global _model
    if _model is None:
        _model = SentenceTransformer(name)
    return _model
def embed(texts, name: str = "all-MiniLM-L6-v2"):
    if isinstance(texts, str): texts=[texts]
    return _get(name).encode(texts, normalize_embeddings=True)
