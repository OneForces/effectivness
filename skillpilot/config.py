import os
from dotenv import load_dotenv

load_dotenv()

LLM_BACKEND = os.getenv("LLM_BACKEND", "ollama").strip().lower()

# Ollama
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://ollama:11434").rstrip("/")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "mistral")
# новые настройки
OLLAMA_TIMEOUT = int(os.getenv("OLLAMA_TIMEOUT", "600"))          # секунд
OLLAMA_KEEP_ALIVE = os.getenv("OLLAMA_KEEP_ALIVE", "15m")         # не выгружать модель
OLLAMA_NUM_CTX = int(os.getenv("OLLAMA_NUM_CTX", "4096"))

# OpenAI (опционально)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

# Embeddings
EMB_MODEL = os.getenv("EMB_MODEL", "all-MiniLM-L6-v2")
