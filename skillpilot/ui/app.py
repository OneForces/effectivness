# skillpilot/ui/app.py
import os, io, re, tempfile, time, zipfile, json, datetime
import gradio as gr

from ..config import LLM_BACKEND, OLLAMA_MODEL, EMB_MODEL
from ..core.scorer import score_fit
from ..gen.resume import make_tailored_resume
from ..gen.cover import make_cover
from ..gen.plan import make_7day_plan
from ..interview.qa import gen_questions, grade_answer
from ..graph.skill_graph import demo_graph_reco, render_graph_png
from ..utils.export import export_md, export_pdf
from ..gen.llm_ollama import is_available as ollama_up  # индикатор статуса
from ..gen.llm import llm_stream  # настоящий стрим для песочницы
from ..utils.pii import anonymize  # режим приватности (PII)

# ⬇️ новые утилиты (вау-фичи)
from ..utils.batch import batch_score, read_any_to_text
from ..utils.viz import radar_coverage, heat_coverage
from ..utils.ats import ats_check

THEME_MODE = (os.getenv("THEME", "light") or "light").strip().lower()  # light | dark


# ---------------- paths for sessions ----------------
def _sess_dir() -> str:
    d = os.path.expanduser("~/.skillpilot/sessions")
    os.makedirs(d, exist_ok=True)
    return d

def _sess_path(name: str) -> str:
    safe = re.sub(r"[^a-zA-Z0-9._-]+", "_", (name or "").strip()) or "session"
    return os.path.join(_sess_dir(), f"{safe}.json")

def _list_sessions() -> list[str]:
    try:
        return sorted([f for f in os.listdir(_sess_dir()) if f.endswith(".json")])
    except Exception:
        return []


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
            f.write((content or "").strip() + "\n")
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
    txt = (txt or "").replace("\x00", " ")
    txt = re.sub(r"\r\n?", "\n", txt)
    txt = re.sub(r"\n{3,}", "\n\n", txt)
    return txt.strip()

def _read_any(file_obj) -> str:
    if not file_obj:
        return ""
    name = (os.path.basename(getattr(file_obj, "name", "") or "")).lower()
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


# ---------- streaming helper (визуальный стрим) ----------
def _yield_chunks(text: str, chunk: int = 48, delay: float = 0.015):
    """Простой «визуальный» стрим (без реального онлайнового токен-стрима)."""
    text = (text or "").strip()
    acc = ""
    for i in range(0, len(text), chunk):
        acc = text[: i + chunk]
        time.sleep(delay)
        yield acc


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

    /* Делаем чекбокс гарантированно кликабельным */
    #stream_out input[type="checkbox"]{
      appearance:auto !important;
      accent-color: var(--sp-accent) !important;
      width:20px; height:20px;
      border:1.5px solid var(--sp-card-border) !important;
      border-radius:6px;
      background: var(--sp-card-bg) !important;
      position: relative; z-index: 5; pointer-events: auto !important;
    }
    #stream_out, #stream_out * { position: relative; z-index: 5; pointer-events: auto !important; }
    """

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
      --color-text:#E6EAF2 !импортант;
    }

    .gradio-container { max_width:1160px !important; margin:0 auto !important; }
    html, body { background:var(--sp-bg) !important; color:var(--sp-text) !important; }
    .prose, .gr-prose, .markdown, .markdown * { color:var(--sp-text) !important; }

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
    .gradio-container .shadow,
    .gradio-container .file-wrap,
    .gradio-container .upload-box,
    .gradio-container .file-preview,
    .gradio-container .gr-accordion,
    .gradio-container .gr-accordion .gr-accordion-header,
    .gradio-container .gr-accordion .gr-accordion-content,
    .gradio-container .gr-dataframe,
    .gradio-container .gr-dataframe thead tr th,
    .gradio-container .gr-dataframe tbody tr td {
      background:var(--sp-card-bg) !important;
      border-color:var(--sp-card-border) !important;
      color:var(--sp-text) !important;
    }

    .gradio-container .bg-white,
    .gradio-container .bg-white\\/0,
    .gradio-container .bg-white\\/50,
    .gradio-container [style*="background: white"],
    .gradio-container [style*="background-color: white"] {
      background:var(--sp-card-bg) !important;
      background-color:var(--sp-card-bg) !important;
    }

    .gradio-container pre, .gradio-container code, .gradio-container kbd, .gradio-container samp {
      background:rgba(255,255,255,0.03) !important;
      color:var(--sp-text) !important;
      border:1px solid var(--sp-card-border) !important;
    }

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

    .gradio-container textarea:disabled,
    .gradio-container input[type="text"]:disabled {
      background:var(--sp-card-bg) !important; color:var(--sp-text) !important; opacity:0.9;
    }
    .gradio-container textarea:focus,
    .gradio-container input[type="text"]:focus {
      outline:1px solid var(--sp-accent) !important;
    }

    .gradio-container label, .gradio-container .label, .gradio-container .label-wrap,
    .gradio-container .md, .gradio-container .prose {
      color:var(--sp-text) !important; background:transparent !important;
    }

    .gr-button,.btn { border-radius:12px !important; }

    .tabs{ margin_top:6px; }
    .tabs > div[role="tablist"] button[aria-selected="true"] { box-shadow: inset 0 -2px 0 0 var(--sp-accent) !important; }

    ::-webkit-scrollbar{ height:10px; width:10px; }
    ::-webkit-scrollbar-thumb{ background:#2D3B55; border-radius:8px; }
    ::-webkit-scrollbar-track{ background:#0E1522; }

    /* Делаем чекбокс гарантированно кликабельным */
    #stream_out input[type="checkbox"]{
      appearance:auto !important;
      accent-color: var(--sp-accent) !important;
      width:20px; height:20px;
      border:1.5px solid var(--sp-card-border) !important;
      border-radius:6px;
      background: var(--sp-card-bg) !important;
      position: relative; z-index: 5; pointer-events: auto !important;
    }
    #stream_out, #stream_out * { position: relative; z-index: 5; pointer-events: auto !important; }
    """

    css = CSS_DARK if THEME_MODE == "dark" else CSS_LIGHT

    with gr.Blocks(title="SkillPilot", theme=theme, css=css) as demo:
        # Header + LLM статус
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
                backend = (f"Ollama · {OLLAMA_MODEL}" if (LLM_BACKEND or "").lower() == "ollama" else (LLM_BACKEND or "—").upper())
                status_html = gr.HTML(elem_classes=["sp-card"])

                def _llm_status():
                    if (LLM_BACKEND or "").lower() == "ollama":
                        ok = ollama_up()
                        dot = "🟢" if ok else "🔴"
                        return f"""
                        <div class="sp-card">
                          <div class="sp-pill">LLM: {backend}</div>
                          <div class="sp-pill">Embeddings: {EMB_MODEL}</div>
                          <div class="sp-pill">UI: Gradio</div>
                          <div style="margin-top:8px">{dot} <b>Ollama</b> status: {"online" if ok else "offline"}</div>
                        </div>"""
                    else:
                        return f"""
                        <div class="sp-card">
                          <div class="sp-pill">LLM: {backend}</div>
                          <div class="sp-pill">Embeddings: {EMB_MODEL}</div>
                          <div class="sp-pill">UI: Gradio</div>
                          <div style="margin-top:8px">ℹ️ Для OpenAI-совместимых эндпоинтов прогресс виден в баре действий.</div>
                        </div>"""
                status_html.value = _llm_status()
                gr.Button("↻ Проверить LLM").click(lambda: _llm_status(), outputs=status_html)

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

            # ----- Пакетная проверка (новая)
            with gr.Tab("📦 Пакетная проверка"):
                gr.Markdown("Сравнить пачку резюме с текущим JD. Загрузите ZIP или выберите несколько файлов.")
                with gr.Row():
                    zip_in = gr.File(label="ZIP с резюме (txt/pdf/docx/md)", file_types=[".zip"])
                    files_in = gr.Files(label="Или выберите несколько файлов", type="filepath")
                with gr.Row():
                    btn_batch = gr.Button("Скоринг пачки", variant="primary")
                batch_table = gr.Dataframe(headers=["resume","score","strengths","gaps"], interactive=False, wrap=True)
                csv_out = gr.File(label="Экспорт CSV", interactive=False)

            # ----- Анализ
            with gr.Tab("🧮 Анализ"):
                with gr.Row():
                    score_out = gr.Number(label="Job-Fit (0–100)", interactive=False, elem_classes=["sp-card"])
                    strengths = gr.JSON(label="Сильные стороны", elem_classes=["sp-card"])
                    gaps = gr.JSON(label="Пробелы", elem_classes=["sp-card"])
                diag = gr.Textbox(label="Диагностика / Coverage", lines=8, elem_classes=["sp-card"])
                with gr.Row():
                    hide_pii = gr.Checkbox(value=True, label="Скрывать PII (имена/email/телефон) перед обработкой", elem_classes=["sp-card"])
                with gr.Row():
                    btn_fit = gr.Button("Оценить соответствие", variant="primary", interactive=False)
                    btn_graph = gr.Button("Построить рекомендации навыков", variant="secondary", interactive=False)

            # ----- Генерация
            with gr.Tab("✍️ Генерация"):
                with gr.Row():
                    btn_tailor = gr.Button("Адаптировать резюме (STAR)", variant="primary", interactive=False)
                    btn_cover = gr.Button("Сгенерировать сопроводительное", variant="primary", interactive=False)
                    btn_plan = gr.Button("План на 7 дней", variant="secondary", interactive=False)
                stream_out = gr.Checkbox(value=True, label="⚡️ Стримить вывод (визуально)", interactive=True, elem_id="stream_out")
                btn_stop = gr.Button("⏹️ Остановить генерацию", variant="stop")

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

            # ----- Визуализация (новая)
            with gr.Tab("📊 Визуализация"):
                gr.Markdown("Radar и Heatmap по покрытию топ-навыков JD из блока «Диагностика / Coverage».")
                viz_radar = gr.Image(label="Radar", type="filepath")
                viz_heat  = gr.Image(label="Heatmap", type="filepath")
                btn_viz   = gr.Button("Построить графики", variant="secondary")

            # ----- ATS-чекер (новая)
            with gr.Tab("🧾 ATS-чекер"):
                gr.Markdown("Быстрый аудит структуры резюме под ATS.")
                ats_json = gr.JSON(label="Отчёт")
                btn_ats  = gr.Button("Проверить текущее резюме", variant="secondary")

            # ----- Мини-интервью
            with gr.Tab("🎤 Мини-интервью"):
                qlist = gr.Textbox(label="Сгенерированные вопросы (5 шт.)", lines=8, elem_classes=["sp-card"])
                with gr.Row():
                    q = gr.Textbox(label="Вопрос", lines=2, elem_classes=["sp-card"])
                    a = gr.Textbox(label="Ваш ответ", lines=4, elem_classes=["sp-card"])
                btn_grade = gr.Button("Оценить ответ", variant="primary")
                grade = gr.Textbox(label="Фидбек по рубрике", lines=10, elem_classes=["sp-card"])

            # ----- Prompt-песочница
            with gr.Tab("🧪 Prompt-песочница"):
                sys_box = gr.Textbox(label="System", lines=4, placeholder="Кем модель должна быть")
                usr_box = gr.Textbox(label="User", lines=8, placeholder="Ваш запрос/промпт")
                with gr.Row():
                    temp = gr.Slider(0.0, 1.5, value=0.25, step=0.05, label="temperature")
                    mxtok = gr.Slider(64, 4096, value=800, step=32, label="max_tokens")
                btn_run_pp = gr.Button("▶️ Запустить (реальный стрим)")
                btn_stop_pp = gr.Button("⏹️ Стоп", variant="stop")
                out_pp = gr.Textbox(label="Ответ (stream)", lines=14)

            # ----- Настройки / Сессии
            with gr.Tab("⚙️ Настройки / Сессии"):
                gr.Markdown(
                    f"""
                    <div class="sp-card">
                      <p><b>Backend:</b> {LLM_BACKEND or '—'}; <b>Ollama model:</b> {OLLAMA_MODEL or '—'}; <b>Embeddings:</b> {EMB_MODEL or '—'}</p>
                      <p>Тема: <code>THEME=dark|light</code> или URL параметр <code>__theme=dark</code>.</p>
                    </div>
                    """,
                    elem_classes=["sp-card"]
                )
                with gr.Row():
                    sess_name = gr.Textbox(label="Имя сессии", placeholder="напр. vacancy_senior_ds")
                    btn_save_sess = gr.Button("💾 Сохранить сессию")
                with gr.Row():
                    sess_list = gr.Dropdown(choices=_list_sessions(), label="Загрузить сессию", allow_custom_value=False)
                    btn_refresh_sess = gr.Button("🔄 Обновить список")

        # -------- handlers --------

        btn_demo.click(_load_demo, inputs=None, outputs=[jd, resume])

        btn_file2text.click(lambda f1, f2: (_read_any(f1), _read_any(f2)), inputs=[jd_file, cv_file], outputs=[jd, resume])

        def _update_buttons(jd_text, cv_text):
            ok = _can_run(jd_text, cv_text)
            upd = gr.update(interactive=ok)
            return upd, upd, upd, upd, upd

        jd.change(_update_buttons, inputs=[jd, resume], outputs=[btn_fit, btn_graph, btn_tailor, btn_cover, btn_plan])
        resume.change(_update_buttons, inputs=[jd, resume], outputs=[btn_fit, btn_graph, btn_tailor, btn_cover, btn_plan])

        btn_clear.click(lambda: ("", ""), inputs=None, outputs=[jd, resume])
        btn_clear.click(lambda: (gr.update(interactive=False),)*5, inputs=None, outputs=[btn_fit, btn_graph, btn_tailor, btn_cover, btn_plan])
        btn_clear.click(lambda: ("", "", "", "", ""), inputs=None, outputs=[tailored, cover, plan, qlist, diag])

        # ---- Пакетная проверка (handler)
        def _do_batch(jd_text, zip_file, file_list, hide):
            if not (jd_text or "").strip():
                return [], None
            J = anonymize(jd_text) if hide else jd_text
            resumes = []
            # ZIP
            if zip_file is not None:
                try:
                    with zipfile.ZipFile(zip_file.name, "r") as z:
                        for nm in z.namelist():
                            if nm.endswith("/") or nm.startswith("__MACOSX/"): 
                                continue
                            data = z.read(nm)
                            try:
                                txt = data.decode("utf-8", "ignore")
                            except Exception:
                                txt = ""
                            resumes.append((nm, anonymize(txt) if hide else txt))
                except Exception:
                    pass
            # files
            if file_list:
                for fp in file_list:
                    path = fp if isinstance(fp, str) else getattr(fp, "name", "")
                    name = os.path.basename(path)
                    txt = read_any_to_text(path)
                    resumes.append((name, anonymize(txt) if hide else txt))

            if not resumes:
                return [], None
            rows, csv_path = batch_score(J, resumes)
            table = [[r["resume"], r["score"], r["strengths"], r["gaps"]] for r in rows]
            return table, csv_path

        btn_batch.click(_do_batch, inputs=[jd, zip_in, files_in, hide_pii], outputs=[batch_table, csv_out])

        # ---- Анализ
        def do_fit(jd_text, cv_text, hide, progress=gr.Progress(track_tqdm=True)):
            if not _can_run(jd_text, cv_text):
                return 0, [], [], "Сначала вставьте JD и резюме."
            J = anonymize(jd_text) if hide else jd_text
            R = anonymize(cv_text) if hide else cv_text
            progress(0.12, desc="🔎 Извлекаем ключевые фразы…")
            time.sleep(0.05)
            progress(0.35, desc="🧠 Считаем эмбеддинги…")
            s, st, gp, msg = score_fit(J, R)
            progress(0.9, desc="🧾 Формируем отчёт…")
            time.sleep(0.05)
            progress(1.0)
            return s, st, gp, msg

        btn_fit.click(do_fit, inputs=[jd, resume, hide_pii], outputs=[score_out, strengths, gaps, diag])

        # ---- Генерация (визуальный стрим)
        def _gen_stream_wrapper(text: str):
            yield from _yield_chunks(text)

        def _guarded(gen_fn, j, r, do_stream: bool, hide: bool, progress=gr.Progress(track_tqdm=True)):
            if not _can_run(j, r):
                return "Сначала заполните JD и резюме."
            J = anonymize(j) if hide else j
            R = anonymize(r) if hide else r
            progress(0.08, desc="🔧 Готовим промпт…")
            time.sleep(0.05)
            progress(0.28, desc="🤖 Вызываем LLM…")
            out = gen_fn(R, J)  # сигнатуры генераторов: (resume, jd)
            progress(0.9, desc="✍️ Заполняем поле…")
            return (_gen_stream_wrapper(out) if do_stream else out)

        tailor_evt = btn_tailor.click(lambda j, r, st, hide: _guarded(make_tailored_resume, j, r, st, hide),
                                      inputs=[jd, resume, stream_out, hide_pii], outputs=[tailored])

        cover_evt = btn_cover.click(lambda j, r, st, hide: _guarded(make_cover, j, r, st, hide),
                                    inputs=[jd, resume, stream_out, hide_pii], outputs=[cover])

        def _make_plan(j, r, do_stream: bool, hide: bool, progress=gr.Progress(track_tqdm=True)):
            if not _can_run(j, r):
                return "Сначала заполните JD и резюме."
            J = anonymize(j) if hide else j
            R = anonymize(r) if hide else r
            progress(0.2, desc="📊 Анализируем JD/резюме…")
            out = make_7day_plan(J, R)  # исходная сигнатура генератора плана
            progress(0.9, desc="✍️ Заполняем поле…")
            return (_gen_stream_wrapper(out) if do_stream else out)

        plan_evt = btn_plan.click(_make_plan, inputs=[jd, resume, stream_out, hide_pii], outputs=[plan])

        # Кнопка «Стоп» реально останавливает любые три генерации выше
        btn_stop.click(lambda: None, inputs=None, outputs=None, cancels=[tailor_evt, cover_evt, plan_evt])

        # ---- Prompt-песочница (настоящий стрим из llm_stream)
        def _run_prompt(system, user, temperature, max_tokens):
            acc = ""
            try:
                for chunk in llm_stream(system, user, temperature=temperature, max_tokens=int(max_tokens)):
                    if chunk:
                        acc += chunk
                        yield acc
            except Exception as e:
                yield f"[STREAM ERROR] {type(e).__name__}: {e}"

        pp_evt = btn_run_pp.click(_run_prompt, inputs=[sys_box, usr_box, temp, mxtok], outputs=[out_pp])
        btn_stop_pp.click(lambda: None, inputs=None, outputs=None, cancels=[pp_evt])

        # ---- Навыки / Граф
        def _graph_text(j, r, hide, progress=gr.Progress()):
            if not _can_run(j, r):
                return "Сначала заполните JD и резюме."
            J = anonymize(j) if hide else j
            R = anonymize(r) if hide else r
            progress(0.4, desc="🌐 Строим рекомендации…")
            res = demo_graph_reco(J, R)
            try:
                G, have, recs = res
                txt = "Имеющиеся навыки: " + ", ".join(sorted(have)) + "\n" + \
                      "Рекомендуемые навыки: " + ", ".join(sorted(recs))
            except Exception:
                txt = str(res)
            progress(1.0)
            return txt

        def _graph_img(j, r, hide):
            J = anonymize(j) if hide else j
            R = anonymize(r) if hide else r
            res = demo_graph_reco(J, R)
            try:
                G, _, _ = res
            except Exception:
                return None
            return render_graph_png(G, target_role="под JD")

        btn_graph.click(_graph_text, inputs=[jd, resume, hide_pii], outputs=[path_text])
        btn_graph.click(_graph_img, inputs=[jd, resume, hide_pii], outputs=[graph_img])

        # ---- Визуализация (парсим coverage из diag)
        def _parse_marks(diag_text: str):
            marks, skills = [], []
            if "Coverage:" in (diag_text or ""):
                try:
                    cov = diag_text.split("Coverage:",1)[1].strip()
                    parts = [p.strip() for p in cov.split(",")]
                    for p in parts:
                        if ":" in p:
                            skill, flag = p.split(":",1)
                            skill = skill.strip().strip("-• ")
                            ok = "✅" in flag
                            if skill:
                                skills.append(skill)
                                marks.append(ok)
                except Exception:
                    pass
            return skills[:12], marks[:12]

        def _viz(diag_text):
            skills, marks = _parse_marks(diag_text)
            if not skills:
                return None, None
            return radar_coverage(skills, marks), heat_coverage(skills, marks)

        btn_viz.click(_viz, inputs=[diag], outputs=[viz_radar, viz_heat])

        # ---- ATS-чекер
        btn_ats.click(lambda txt, hide: ats_check(anonymize(txt) if hide else txt),
                      inputs=[resume, hide_pii], outputs=[ats_json])

        # ---- Пакет ZIP
        btn_bundle.click(lambda t, c, p, j, r: _bundle_artifacts(t, c, p, j, r),
                         inputs=[tailored, cover, plan, jd, resume], outputs=[bundle_file])

        # ---- Экспорт
        def _export_md_all(t, c, p, progress=gr.Progress()):
            if not any([t, c, p]): return None
            progress(0.2, desc="📦 Готовим файлы…")
            out_dir = os.path.join("/tmp", f"skillpilot_md_{int(time.time())}")
            os.makedirs(out_dir, exist_ok=True)
            paths = []
            if t: paths.append(export_md(out_dir, "resume_tailored", t))
            if c: paths.append(export_md(out_dir, "cover_letter", c))
            if p: paths.append(export_md(out_dir, "plan_7days", p))
            progress(0.65, desc="🧩 Упаковываем ZIP…")
            zip_path = os.path.join(out_dir, "md_package.zip")
            with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as z:
                for pth in paths: z.write(pth, arcname=os.path.basename(pth))
            progress(1.0); return zip_path

        btn_exp_md.click(_export_md_all, inputs=[tailored, cover, plan], outputs=[md_pkg])

        def _export_pdf_resume(t, progress=gr.Progress()):
            if not t: return None
            progress(0.4, desc="🖨 Рендерим PDF…")
            out_dir = os.path.join("/tmp", f"skillpilot_pdf_{int(time.time())}")
            path = export_pdf(out_dir, "resume_tailored", t, title="Адаптированное резюме")
            progress(1.0); return path

        btn_exp_pdf.click(_export_pdf_resume, inputs=[tailored], outputs=[pdf_file])

        # ---- Q&A
        btn_fit.click(lambda jd_text: ("\n".join(gen_questions(jd_text, 5)) if (jd_text or "").strip() else "Сначала вставьте JD."),
                      inputs=[jd], outputs=[qlist])

        btn_grade.click(lambda q_text, a_text: grade_answer(q_text, a_text), inputs=[q, a], outputs=[grade])

        # ---- Сохранение/загрузка сессий
        def _save_session(name, jd_text, resume_text, tailored_text, cover_text, plan_text):
            path = _sess_path(name)
            payload = dict(
                name=name,
                ts=datetime.datetime.utcnow().isoformat() + "Z",
                jd=jd_text, resume=resume_text,
                tailored=tailored_text, cover=cover_text, plan=plan_text,
            )
            with open(path, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
            return gr.update(choices=_list_sessions())

        btn_save_sess.click(_save_session,
                            inputs=[sess_name, jd, resume, tailored, cover, plan],
                            outputs=[sess_list])

        btn_refresh_sess.click(lambda: gr.update(choices=_list_sessions()), inputs=None, outputs=[sess_list])

        def _load_session(sel_name):
            if not sel_name:
                return gr.update(), gr.update(), gr.update(), gr.update(), gr.update()
            path = os.path.join(_sess_dir(), sel_name)
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                return data.get("jd",""), data.get("resume",""), data.get("tailored",""), data.get("cover",""), data.get("plan","")
            except Exception as e:
                return f"[LOAD ERROR] {e}", "", "", "", ""

        sess_list.change(_load_session, inputs=[sess_list], outputs=[jd, resume, tailored, cover, plan])

        # Footer
        gr.Markdown(
            "⌛️ Первая генерация на «холодной» модели может быть дольше; повторы быстрее благодаря keep_alive.",
            elem_classes=["sp-card"]
        )

    return demo


def main():
    app = ui()
    app.launch()


if __name__ == "__main__":
    main()
