from .llm import llm
def make_tailored_resume(resume: str, jd: str):
    system = "Карьерный консультант. Стиль STAR, буллеты."
    prompt = f"""Исходное резюме:\n{resume}\n\nВакансия:\n{jd}\n
Сформируй: 1) 3 ключевых навыка; 2) 5–7 буллетов STAR с метриками."""
    return llm(system, prompt, max_tokens=900)
