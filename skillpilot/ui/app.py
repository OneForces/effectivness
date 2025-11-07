import os, io, re, tempfile, time, zipfile
import gradio as gr

from ..config import LLM_BACKEND, OLLAMA_MODEL, EMB_MODEL
from ..core.scorer import score_fit
from ..gen.resume import make_tailored_resume
from ..gen.cover import make_cover
from ..gen.plan import make_7day_plan
from ..interview.qa import gen_questions, grade_answer
from ..graph.skill_graph import demo_graph_reco, render_graph_png
from ..utils.export import export_md, export_pdf

THEME_MODE = os.getenv("THEME", "light").strip().lower()  # light | dark


# ---------------- helpers ----------------

def _bundle_artifacts(tailored_text: str, cover_text: str, plan_text: str, jd_text: str, resume_text: str) -> str | None:
    if not any([tailored_text, cover_text, plan_text, jd_text, resume_text]):
        return None
    tmpdir = tempfile.mkdtemp(prefix="skillpilot_pkg_")
    files = []

    def _write(name: str, content: str | None):
        if not content:
            return
        path = os.path.join(tmpdir, name)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content.strip() + "\n")
        files.append(path)

    _write("job_description.txt", jd_text)
    _write("resume_source.txt", resume_text)
    _write("resume_tailored.txt", tailored_text)
    _write("cover_letter.txt", cover_text)
    _write("7day_plan.txt", plan_text)

    if not files:
        return None

    zip_path = os.path.join(tmpdir, f"skillpilot_package_{int(time.time())}.zip")
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for p in files:
            z.write(p, arcname=os.path.basename(p))
    return zip_path


def _load_demo():
    def _safe_read(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return f.read()
        except Exception:
            return ""
    jd_text = _safe_read("sample_data/jd_ru.txt")
    cv_text = _safe_read("sample_data/resume_ru.txt")
    return jd_text, cv_text


def _can_run(jd_text: str, cv_text: str) -> bool:
    return bool(jd_text and jd_text.strip()) and bool(cv_text and cv_text.strip())


# ---------- lightweight file ingest (PDF/DOCX/TXT/MD) ----------

def _clean_txt(txt: str) -> str:
    txt = txt.replace("\x00", " ")
    txt = re.sub(r"\r\n?", "\n", txt)
    txt = re.sub(r"\n{3,}", "\n\n", txt)
    return txt.strip()

def _read_any(file_obj) -> str:
    if not file_obj:
        return ""
    name = (os.path.basename(file_obj.name) if hasattr(file_obj, "name") else "").lower()
    try:
        with open(file_obj.name, "rb") as f:
            data = f.read()
    except Exception:
        try:
            data = file_obj.read()
        except Exception:
            return "[ERROR] Не удалось прочитать файл."

    if name.endswith(".pdf"):
        try:
            from pypdf import PdfReader
            reader = PdfReader(io.BytesIO(data))
            pages = [p.extract_text() or "" for p in reader.pages]
            return _clean_txt("\n\n".join(pages))
        except Exception:
            return "[WARN] Для PDF нужен пакет pypdf. Установите его в образ (requirements) либо конвертируйте в TXT."

    if name.endswith(".docx"):
        try:
            import docx  # python-docx
            doc = docx.Document(io.BytesIO(data))
            return _clean_txt("\n".join(p.text for p in doc.paragraphs))
        except Exception:
            return "[WARN] Для DOCX нужен пакет python-docx. Установите его или загрузите TXT/MD."

    if name.endswith(".md"):
        raw = data.decode("utf-8", "ignore")
        raw = re.sub(r"`{1,3}.*?`{1,3}", "", raw, flags=re.S)
        raw = re.sub(r"^#+\s*", "", raw, flags=re.M)
        return _clean_txt(raw)

    try:
        return _clean_txt(data.decode("utf-8", "ignore"))
    except Exception:
        return "[ERROR] Не удалось декодировать файл как UTF-8."


# ---------------- UI ----------------

def ui():
    theme = gr.themes.Soft(primary_hue="blue", secondary_hue="slate", neutral_hue="slate")

    CSS_LIGHT = """
    :root {
      --sp-bg:#F7F9FC; --sp-text:#0B1220;
      --sp-card-bg:#FFFFFF; --sp-card-border:#E5EAF3;
      --sp-pill-bg:#EEF3FF; --sp-pill-text:#22315A;
      --sp-accent:#3B82F6; --sp-muted:#6B7280;

      --body-background-fill:#F7F9FC !important;
      --background-fill-primary:#FFFFFF !important;
      --background-fill-secondary:#FFFFFF !important;
      --block-background-fill:#FFFFFF !important;
      --border-color-primary:#E5EAF3 !important;
      --color-text:#0B1220 !important;
    }
    .gradio-container { max-width:1160px !important; margin:0 auto !important; }
    html, body { background:var(--sp-bg) !important; color:var(--sp-text) !important; }
    .prose, .gr-prose, .markdown, .markdown * { color:var(--sp-text) !important; }

    .sp-card,
    .gradio-container .sp-card.gr-box,
    .gradio-container .sp-card .gr-box {
      border:1px solid var(--sp-card-border) !important;
      background:var(--sp-card-bg) !important;
      border-radius:14px !important; padding:14px !important; color:var(--sp-text) !important;
    }

    .sp-pill { display:inline-block; padding:4px 10px; border-radius:999px;
      background:var(--sp-pill-bg) !important; color:var(--sp-pill-text) !important; margin-right:8px; font-size:12px }

    .sp-hero h1 { margin:0; font-size:26px }
    .sp-hero small { opacity:.75 }

    .gr-button,.btn { border-radius:12px !important; }

    .sp-card .gr-input, .sp-card .gr-textbox, .sp-card .gr-textbox > div,
    textarea,input,.gr-textbox textarea{
      background:var(--sp-card-bg) !important; color:var(--sp-text) !important; border-color:var(--sp-card-border) !important;
    }
    .gr-textbox textarea::placeholder{ color:var(--sp-muted) !important; }

    .tabs{ margin-top:6px; }
    .tabs > div[role="tablist"] button[aria-selected="true"] { box-shadow: inset 0 -2px 0 0 var(--sp-accent) !important; }
    textarea:focus, input:focus { outline:1px solid var(--sp-accent) !important; }

    ::-webkit-scrollbar{ height:10px; width:10px; }
    ::-webkit-scrollbar-thumb{ background:#D2D8E4; border-radius:8px; }
    ::-webkit-scrollbar-track{ background:#EFF2F8; }
    """

    # Тёмная: полностью прибиваем любые white bg (включая tailwind'овские bg-white)
    CSS_DARK = """
    :root {
      --sp-bg:#0B1220; --sp-text:#E6EAF2;
      --sp-card-bg:#101826; --sp-card-border:#293247;
      --sp-pill-bg:#172542; --sp-pill-text:#DCE6FF;
      --sp-accent:#60A5FA; --sp-muted:#9CA3AF;

      --body-background-fill:#0B1220 !important;
      --background-fill-primary:#101826 !important;
      --background-fill-secondary:#101826 !important;
      --block-background-fill:#101826 !important;
      --border-color-primary:#293247 !important;
      --color-text:#E6EAF2 !important;
    }

    .gradio-container { max-width:1160px !important; margin:0 auto !important; }
    html, body { background:var(--sp-bg) !important; color:var(--sp-text) !important; }
    .prose, .gr-prose, .markdown, .markdown * { color:var(--sp-text) !important; }

    /* Карточки и ВСЕ типовые контейнеры */
    .sp-card,
    .gradio-container .gr-box,
    .gradio-container .gr-panel,
    .gradio-container .gr-group,
    .gradio-container .form,
    .gradio-container .tabitem,
    .gradio-container .tabs,
    .gradio-container .container,
    .gradio-container .wrap,
    .gradio-container .block,
    .gradio-container .panel,
    .gradio-container .border,
    .gradio-container .shadow {
      background:var(--sp-card-bg) !important;
      border-color:var(--sp-card-border) !important;
      color:var(--sp-text) !important;
    }

    /* убиваем все "bg-white" и прозрачные наливки, которые Gradio/TAILWIND кидает внутри */
    .gradio-container .bg-white,
    .gradio-container .bg-white\\/0,
    .gradio-container .bg-white\\/50,
    .gradio-container [style*="background: white"],
    .gradio-container [style*="background-color: white"] {
      background:var(--sp-card-bg) !important;
      background-color:var(--sp-card-bg) !important;
    }

    /* Текстовые поля — перекрываем все уровни вложенности */
    .gradio-container textarea,
    .gradio-container input[type="text"],
    .gradio-container .gr-input,
    .gradio-container .gr-textbox,
    .gradio-container .gr-textbox > div,
    .gradio-container .gr-textbox > div > textarea,
    .gradio-container .gr-textbox textarea,
    .gradio-container .gr-input textarea,
    .gradio-container .gr-textbox .container,
    .gradio-container .gr-textbox .wrap,
    .gradio-container [data-testid="textbox"] textarea {
      background:var(--sp-card-bg) !important;
      color:var(--sp-text) !important;
      border-color:var(--sp-card-border) !important;
      box-shadow:none !important;
    }

    /* disabled / focus */
    .gradio-container textarea:disabled,
    .gradio-container input[type="text"]:disabled {
      background:var(--sp-card-bg) !important; color:var(--sp-text) !important; opacity:0.9;
    }
    .gradio-container textarea:focus,
    .gradio-container input[type="text"]:focus {
      outline:1px solid var(--sp-accent) !important;
    }

    /* лейблы/подписи */
    .gradio-container label, .gradio-container .label, .gradio-container .label-wrap,
    .gradio-container .md, .gradio-container .prose {
      color:var(--sp-text) !important; background:transparent !important;
    }

    /* upload-зоны и предпросмотр файлов */
    .gradio-container .file-wrap, .gradio-container .upload-box, .gradio-container .file-preview {
      background:var(--sp-card-bg) !important; border-color:var(--sp-card-border) !important; color:var(--sp-text) !important;
    }

    .sp-pill { display:inline-block; padding:4px 10px; border-radius:999px;
      background:var(--sp-pill-bg) !important; color:var(--sp-pill-text) !important; margin-right:8px; font-size:12px }

    .sp-hero h1 { margin:0; font-size:26px }
    .sp-hero small { opacity:.85 }

    .gr-button,.btn { border-radius:12px !important; }

    .tabs{ margin-top:6px; }
    .tabs > div[role="tablist"] button[aria-selected="true"] { box-shadow: inset 0 -2px 0 0 var(--sp-accent) !important; }

    ::-webkit-scrollbar{ height:10px; width:10px; }
    ::-webkit-scrollbar-thumb{ background:#2D3B55; border-radius:8px; }
    ::-webkit-scrollbar-track{ background:#0E1522; }
    """

    css = CSS_DARK if THEME_MODE == "dark" else CSS_LIGHT

    with gr.Blocks(title="SkillPilot", theme=theme, css=css) as demo:
        # Header
        with gr.Row():
            with gr.Column(scale=3):
                gr.Markdown(
                    """
                    <div class="sp-hero sp-card">
                      <h1>🚀 SkillPilot — карьерный помощник</h1>
                      <small>Python-only · локальная LLM через Ollama · артефакты под вакансию</small>
                    </div>
                    """,
                    elem_classes=["sp-card"]
                )
            with gr.Column(scale=2):
                backend = (f"Ollama · {OLLAMA_MODEL}" if LLM_BACKEND == "ollama" else LLM_BACKEND.upper())
                gr.Markdown(
                    f"""
                    <div class="sp-card">
                        <div class="sp-pill">LLM: {backend}</div>
                        <div class="sp-pill">Embeddings: {EMB_MODEL}</div>
                        <div class="sp-pill">UI: Gradio</div>
                    </div>
                    """,
                    elem_classes=["sp-card"]
                )

        with gr.Tabs():
            # ----- Данные
            with gr.Tab("📄 Данные"):
                with gr.Row():
                    jd = gr.Textbox(label="Вакансия (JD)", lines=14, placeholder="Вставьте описание вакансии", elem_classes=["sp-card"])
                    resume = gr.Textbox(label="Резюме (текст)", lines=14, placeholder="Вставьте резюме", elem_classes=["sp-card"])
                with gr.Row():
                    jd_file = gr.File(label="Загрузить JD (PDF/DOCX/TXT/MD)", file_types=[".pdf", ".docx", ".txt", ".md"])
                    cv_file = gr.File(label="Загрузить Резюме (PDF/DOCX/TXT/MD)", file_types=[".pdf", ".docx", ".txt", ".md"])
                with gr.Row():
                    btn_file2text = gr.Button("📥 Прочитать файлы → в поля", variant="secondary")
                with gr.Row():
                    btn_demo = gr.Button("Загрузить демоданные", variant="secondary")
                    btn_clear = gr.Button("Очистить поля", variant="stop")

            # ----- Анализ
            with gr.Tab("🧮 Анализ"):
                with gr.Row():
                    score_out = gr.Number(label="Job-Fit (0–100)", interactive=False, elem_classes=["sp-card"])
                    strengths = gr.JSON(label="Сильные стороны", elem_classes=["sp-card"])
                    gaps = gr.JSON(label="Пробелы", elem_classes=["sp-card"])
                diag = gr.Textbox(label="Диагностика / Coverage", lines=8, elem_classes=["sp-card"])
                with gr.Row():
                    btn_fit = gr.Button("Оценить соответствие", variant="primary", interactive=False)
                    btn_graph = gr.Button("Построить рекомендации навыков", variant="secondary", interactive=False)

            # ----- Генерация
            with gr.Tab("✍️ Генерация"):
                with gr.Row():
                    btn_tailor = gr.Button("Адаптировать резюме (STAR)", variant="primary", interactive=False)
                    btn_cover = gr.Button("Сгенерировать сопроводительное", variant="primary", interactive=False)
                    btn_plan = gr.Button("План на 7 дней", variant="secondary", interactive=False)
                tailored = gr.Textbox(label="Адаптированное резюме", lines=12, elem_classes=["sp-card"])
                cover = gr.Textbox(label="Сопроводительное письмо", lines=12, elem_classes=["sp-card"])
                plan = gr.Textbox(label="План на 7 дней", lines=12, elem_classes=["sp-card"])

                with gr.Row():
                    btn_bundle = gr.Button("⬇️ Скачать пакет (ZIP)", variant="secondary")
                    bundle_file = gr.File(label="Готовый ZIP", interactive=False)

                with gr.Row():
                    btn_exp_md = gr.Button("⬇️ Export MD (резюме+cover+план)", variant="secondary")
                    btn_exp_pdf = gr.Button("⬇️ Export PDF (резюме)", variant="secondary")
                with gr.Row():
                    md_pkg = gr.File(label="MD-пакет (ZIP)", interactive=False)
                    pdf_file = gr.File(label="PDF-файл", interactive=False)

            # ----- Навыки / Граф
            with gr.Tab("🧭 Навыки / Граф"):
                with gr.Row():
                    path_text = gr.Textbox(label="SkillGraph рекомендации (текст)", lines=8, elem_classes=["sp-card"])
                    graph_img = gr.Image(label="SkillGraph (PNG)", type="filepath", elem_classes=["sp-card"])

            # ----- Мини-интервью
            with gr.Tab("🎤 Мини-интервью"):
                qlist = gr.Textbox(label="Сгенерированные вопросы (5 шт.)", lines=8, elem_classes=["sp-card"])
                with gr.Row():
                    q = gr.Textbox(label="Вопрос", lines=2, elem_classes=["sp-card"])
                    a = gr.Textbox(label="Ваш ответ", lines=4, elem_classes=["sp-card"])
                btn_grade = gr.Button("Оценить ответ", variant="primary")
                grade = gr.Textbox(label="Фидбек по рубрике", lines=10, elem_classes=["sp-card"])

            # ----- Настройки
            with gr.Tab("⚙️ Настройки"):
                gr.Markdown(
                    f"""
                    <div class="sp-card">
                      <p><b>Backend:</b> {LLM_BACKEND or '—'}</p>
                      <p><b>Ollama model:</b> {OLLAMA_MODEL or '—'}</p>
                      <p><b>Embeddings:</b> {EMB_MODEL or '—'}</p>
                      <p>Тему можно задать через переменную окружения <code>THEME=dark|light</code>
                      или параметр в урле <code>__theme=dark</code>.</p>
                    </div>
                    """,
                    elem_classes=["sp-card"]
                )

        # handlers
        btn_demo.click(_load_demo, inputs=None, outputs=[jd, resume])

        btn_file2text.click(
            lambda f1, f2: (_read_any(f1), _read_any(f2)),
            inputs=[jd_file, cv_file],
            outputs=[jd, resume]
        )

        def _update_buttons(jd_text, cv_text):
            ok = _can_run(jd_text, cv_text)
            upd = gr.update(interactive=ok)
            return upd, upd, upd, upd, upd

        jd.change(_update_buttons, inputs=[jd, resume], outputs=[btn_fit, btn_graph, btn_tailor, btn_cover, btn_plan])
        resume.change(_update_buttons, inputs=[jd, resume], outputs=[btn_fit, btn_graph, btn_tailor, btn_cover, btn_plan])

        btn_clear.click(lambda: ("", ""), inputs=None, outputs=[jd, resume])
        btn_clear.click(lambda: (gr.update(interactive=False),)*5, inputs=None, outputs=[btn_fit, btn_graph, btn_tailor, btn_cover, btn_plan])
        btn_clear.click(lambda: ("", "", "", "", ""), inputs=None, outputs=[tailored, cover, plan, qlist, diag])

        def do_fit(jd_text, cv_text):
            if not _can_run(jd_text, cv_text):
                return 0, [], [], "Сначала вставьте JD и резюме."
            s, st, gp, msg = score_fit(jd_text, cv_text)
            return s, st, gp, msg
        btn_fit.click(do_fit, inputs=[jd, resume], outputs=[score_out, strengths, gaps, diag])

        def _guarded(gen_fn, j, r):
            if not _can_run(j, r):
                return "Сначала заполните JD и резюме."
            return gen_fn(r, j)

        btn_tailor.click(lambda j, r: _guarded(make_tailored_resume, j, r), inputs=[jd, resume], outputs=tailored)
        btn_cover.click(lambda j, r: _guarded(make_cover, j, r), inputs=[jd, resume], outputs=cover)
        btn_plan.click(lambda j, r: (make_7day_plan(score_fit(j, r)[2], role_hint="под JD")
                                     if _can_run(j, r) else "Сначала заполните JD и резюме."),
                       inputs=[jd, resume], outputs=plan)

        def _graph_text(j, r):
            if not _can_run(j, r):
                return "Сначала заполните JD и резюме."
            return demo_graph_reco(r, target_role="под JD")
        btn_graph.click(_graph_text, inputs=[jd, resume], outputs=[path_text])
        btn_graph.click(lambda r: render_graph_png(r, target_role="под JD"), inputs=[resume], outputs=[graph_img])

        btn_bundle.click(lambda t, c, p, j, r: _bundle_artifacts(t, c, p, j, r),
                         inputs=[tailored, cover, plan, jd, resume], outputs=bundle_file)

        def _export_md_all(t, c, p):
            if not any([t, c, p]):
                return None
            out_dir = os.path.join("/tmp", f"skillpilot_md_{int(time.time())}")
            os.makedirs(out_dir, exist_ok=True)
            paths = []
            if t: paths.append(export_md(out_dir, "resume_tailored", t))
            if c: paths.append(export_md(out_dir, "cover_letter", c))
            if p: paths.append(export_md(out_dir, "plan_7days", p))
            zip_path = os.path.join(out_dir, "md_package.zip")
            with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as z:
                for pth in paths:
                    z.write(pth, arcname=os.path.basename(pth))
            return zip_path

        btn_exp_md.click(_export_md_all, inputs=[tailored, cover, plan], outputs=md_pkg)

        def _export_pdf_resume(t):
            if not t:
                return None
            out_dir = os.path.join("/tmp", f"skillpilot_pdf_{int(time.time())}")
            path = export_pdf(out_dir, "resume_tailored", t, title="Адаптированное резюме")
            return path

        btn_exp_pdf.click(_export_pdf_resume, inputs=[tailored], outputs=pdf_file)

        btn_fit.click(lambda jd_text: ("\n".join(gen_questions(jd_text, 5)) if jd_text.strip() else "Сначала вставьте JD."),
                      inputs=[jd], outputs=qlist)
        btn_grade.click(lambda q_text, a_text: grade_answer(q_text, a_text), inputs=[q, a], outputs=grade)

        gr.Markdown(
            "⌛️ Подсказка: первая генерация с локальной LLM может дольше просыпаться на «холодной» модели. "
            "Повторы быстрее благодаря keep_alive.",
            elem_classes=["sp-card"]
        )

    return demo


def main():
    app = ui()
    app.launch()


if __name__ == "__main__":
    main()
