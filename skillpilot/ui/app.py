import os, tempfile, time, zipfile
import gradio as gr
from ..core.scorer import score_fit
from ..gen.resume import make_tailored_resume
from ..gen.cover import make_cover
from ..gen.plan import make_7day_plan
from ..interview.qa import gen_questions, grade_answer
from ..graph.skill_graph import demo_graph_reco, render_graph_png

def _bundle_artifacts(tailored_text: str, cover_text: str, plan_text: str, jd_text: str, resume_text: str) -> str | None:
    """Собирает zip с артефактами и возвращает путь к файлу (или None, если пусто)."""
    if not any([tailored_text, cover_text, plan_text]):
        return None
    tmpdir = tempfile.mkdtemp(prefix="skillpilot_pkg_")
    files = []
    def _write(name: str, content: str):
        path = os.path.join(tmpdir, name)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content.strip() + "\n")
        files.append(path)

    if jd_text: _write("job_description.txt", jd_text)
    if resume_text: _write("resume_source.txt", resume_text)
    if tailored_text: _write("resume_tailored.txt", tailored_text)
    if cover_text: _write("cover_letter.txt", cover_text)
    if plan_text: _write("7day_plan.txt", plan_text)

    zip_path = os.path.join(tmpdir, f"skillpilot_package_{int(time.time())}.zip")
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for p in files:
            z.write(p, arcname=os.path.basename(p))
    return zip_path

def ui():
    with gr.Blocks(title="SkillPilot") as demo:
        gr.Markdown("# SkillPilot — карьерный помощник (Python only)")

        with gr.Row():
            jd = gr.Textbox(label="Вакансия (JD)", lines=12, placeholder="Вставьте описание вакансии")
            resume = gr.Textbox(label="Резюме (свободный текст)", lines=12, placeholder="Вставьте резюме")

        score_out = gr.Number(label="Job-Fit (0–100)")
        strengths = gr.JSON(label="Сильные стороны")
        gaps = gr.JSON(label="Пробелы")
        diag = gr.Textbox(label="Диагностика", lines=2)

        with gr.Row():
            btn_fit = gr.Button("Оценить соответствие")
            btn_tailor = gr.Button("Адаптировать резюме")
            btn_cover = gr.Button("Сгенерировать Cover")
            btn_plan = gr.Button("7-дневный план")

        tailored = gr.Textbox(label="Адаптированное резюме", lines=10)
        cover = gr.Textbox(label="Cover Letter", lines=10)
        plan = gr.Textbox(label="План на 7 дней", lines=12)

        # граф: текст + картинка
        path_text = gr.Textbox(label="SkillGraph рекомендации (текст)", lines=6)
        graph_img = gr.Image(label="SkillGraph (PNG)", type="filepath")

        # пакет на скачивание
        with gr.Row():
            btn_bundle = gr.Button("Скачать пакет (zip)")
            bundle_file = gr.File(label="Готовый ZIP", interactive=False)

        with gr.Accordion("Мини-интервью", open=False):
            qlist = gr.Textbox(label="Вопросы", lines=8)
            q = gr.Textbox(label="Вопрос", lines=2)
            a = gr.Textbox(label="Ваш ответ", lines=4)
            btn_grade = gr.Button("Оценить ответ")
            grade = gr.Textbox(label="Фидбек", lines=8)

        # --- handlers ---
        def do_fit(jd_text, cv_text):
            s, st, gp, msg = score_fit(jd_text, cv_text)
            return s, st, gp, msg

        def do_tailor(jd_text, cv_text):
            return make_tailored_resume(cv_text, jd_text)

        def do_cover(jd_text, cv_text):
            return make_cover(cv_text, jd_text)

        def do_plan(jd_text, cv_text):
            _, _, gp, _ = score_fit(jd_text, cv_text)
            return make_7day_plan(gp, role_hint="под JD")

        def do_graph_text(jd_text, cv_text):
            return demo_graph_reco(cv_text, target_role="под JD")

        def do_graph_img(cv_text):
            return render_graph_png(cv_text, target_role="под JD")

        def do_bundle(tailored_text, cover_text, plan_text, jd_text, cv_text):
            return _bundle_artifacts(tailored_text, cover_text, plan_text, jd_text, cv_text)

        # --- wire ---
        btn_fit.click(do_fit, inputs=[jd, resume], outputs=[score_out, strengths, gaps, diag])
        btn_tailor.click(do_tailor, inputs=[jd, resume], outputs=tailored)
        btn_cover.click(do_cover, inputs=[jd, resume], outputs=cover)
        btn_plan.click(do_plan, inputs=[jd, resume], outputs=plan)

        # граф — по кнопке Fit
        btn_fit.click(do_graph_text, inputs=[jd, resume], outputs=path_text)
        btn_fit.click(do_graph_img, inputs=[resume], outputs=graph_img)

        # пакет на скачивание
        btn_bundle.click(do_bundle, inputs=[tailored, cover, plan, jd, resume], outputs=bundle_file)

        # интервью
        btn_fit.click(lambda jd_text: "\n".join(gen_questions(jd_text, 5)), inputs=[jd], outputs=qlist)
        btn_grade.click(lambda q_text, a_text: grade_answer(q_text, a_text), inputs=[q, a], outputs=grade)

    return demo

def main():
    demo = ui()
    demo.launch()

if __name__ == "__main__":
    main()
