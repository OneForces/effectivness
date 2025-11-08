# syntax=docker/dockerfile:1.6

FROM python:3.11-slim

# Системные зависимости (редко меняются → стабильный кэш)
RUN apt-get update && apt-get install -y --no-install-recommends \
    git build-essential gcc libgomp1 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Куда кэшировать модели HF внутри образа
ENV HF_HOME=/opt/hf \
    PIP_ROOT_USER_ACTION=ignore

# 1) Зависимости — отдельным слоем (кэш pip через BuildKit)
COPY requirements.txt /app/requirements.txt
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -U pip && \
    pip install -r requirements.txt

# 2) Предзагрузка эмбед-модели при СБОРКЕ (один раз)
#    EMB_MODEL можно переопределить build-аргументом или через compose build args
ARG EMB_MODEL=sentence-transformers/all-MiniLM-L6-v2
ENV EMB_MODEL=${EMB_MODEL}
RUN python - <<'PY'
import os
from sentence_transformers import SentenceTransformer as S

m = os.environ.get("EMB_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
# поддержка короткой формы из .env: all-MiniLM-L6-v2
if "/" not in m:
    m = f"sentence-transformers/{m}"
c = os.environ.get("HF_HOME", "/opt/hf")
print(f"Pre-warming model '{m}' into '{c}' ...")
S(m, cache_folder=c)
print("Done.")
PY

# 3) Копируем исходники (изменения кода НЕ ломают слои выше)
COPY . /app

EXPOSE 7860
# Убедись, что точка входа совпадает с именем файла
CMD ["python", "run.py"]
