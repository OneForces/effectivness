import re, yake
CYRIL = re.compile(r"[А-Яа-я]")
def detect_lang(text: str) -> str: return "ru" if CYRIL.search(text or "") else "en"
def extract_keywords(text: str, top_k: int = 20):
    kw = yake.KeywordExtractor(lan=detect_lang(text), n=1, top=top_k)
    pairs = kw.extract_keywords(text or "")
    terms = [k.strip().lower() for k,_ in sorted(pairs, key=lambda x:x[1])]
    clean = []
    for t in terms:
        t = re.sub(r"[^0-9a-zA-Zа-яА-Я\-\+\#\. ]","",t).strip()
        if len(t)>=2 and t not in clean: clean.append(t)
    return clean[:top_k]
