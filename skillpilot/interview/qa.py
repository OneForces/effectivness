from ..core.extractor import detect_lang
from .rubric import RUBRIC
from ..gen.llm import llm
def gen_questions(jd: str, n: int = 5):
    if not jd: return []
    system = "Интервьюер. Краткие вопросы по JD."
    txt = llm(system, f"Сгенерируй {n} вопросов по JD: {jd}", max_tokens=400)
    qs = [q.strip('- •0123456789. ') for q in txt.split('\n') if q.strip()]
    if qs: return qs[:n]
    lang = detect_lang(jd)
    return (["Расскажите о проекте и оценке качества?","Какие метрики и почему?",
             "Опишите пайплайн фичеризации.","Что делать при переобучении?","Как проведёте A/B тест?"]
            if lang=='ru' else
            ["Describe a project and evaluation.","Which metrics and why?",
             "Feature engineering pipeline.","How to handle overfitting?","How to run an A/B test?"])
def grade_answer(question: str, answer: str):
    system = "Оценщик. Разбор по рубрике и балл 0-100."
    prompt = f"Вопрос: {question}\nОтвет: {answer}\nРубрика: {RUBRIC}"
    return llm(system, prompt, max_tokens=400)
