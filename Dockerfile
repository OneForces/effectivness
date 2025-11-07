# SkillPilot (Python-only) — Dockerfile
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/opt/hf \
    TRANSFORMERS_CACHE=/opt/hf

# системные зависимости (git нужен sentence-transformers для некоторых моделей)
RUN apt-get update && apt-get install -y --no-install-recommends \
    git build-essential gcc libgomp1 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Ставим зависимости до копирования всего кода — лучший кеш
COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

# Предзагрузка эмбеддингов — ОДИН раз при сборке
# Важно: используем ПОЛНОЕ имя модели
RUN python -c "from sentence_transformers import SentenceTransformer as S; S('sentence-transformers/all-MiniLM-L6-v2'); print('✅ prewarmed all-MiniLM-L6-v2')"

# Копируем исходники после прогрева, чтобы правки кода не инвалидировали слой с моделью
COPY . /app

EXPOSE 7860
ENV HOST=0.0.0.0 PORT=7860

CMD ["python", "run.py"]
