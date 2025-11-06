from .llm import llm
def make_cover(resume: str, jd: str):
    system = "Копирайтер. 170–220 слов, конкретика."
    prompt = f"""Сгенерируй сопроводительное письмо под вакансию, сошлись на 3–4 совпадения с JD и 1 кейс из резюме.
Резюме: {resume}
JD: {jd}"""
    return llm(system, prompt, max_tokens=600)
