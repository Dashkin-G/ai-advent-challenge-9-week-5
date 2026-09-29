"""Настройки проекта: откуда брать документы, куда класть индекс, как резать текст."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = Path(os.getenv("DATA_DIR", ROOT / "data"))

# --- Документы: выгружаются один раз ---
# ПДД РФ со всеми приложениями одной страницей, действующая редакция. Текст
# правил — официальный документ, авторским правом не охраняется; примечания
# Гаранта при разборе выбрасываются.
PDD_URL = "https://base.garant.ru/1305770/"
# Экзаменационные билеты ГИБДД (категории A, B). У вопросов есть подсказка со
# ссылкой на пункт правил — из них собирается эталон для проверки поиска.
TICKETS_REPO = "etspring/pdd_russia"
TICKETS_DIR = "questions/A_B/tickets"

RAW_DIR = DATA / "raw"                          # как скачано: HTML и JSON
DOC_PATH = DATA / "docs" / "pdd.md"             # очищенный текст — вход для нарезки
QUESTIONS_PATH = DATA / "eval" / "questions.json"

# --- Индекс ---
INDEX_DIR = DATA / "index"                      # FAISS на каждую стратегию + SQLite
EMBED_MODEL = os.getenv("EMBED_MODEL", "BAAI/bge-m3")
# На процессоре с AVX-512 bfloat16 считает в 1,75 раза быстрее float32, а векторы
# совпадают с точностью до косинуса 0,998. На старом процессоре — float32.
EMBED_DTYPE = os.getenv("EMBED_DTYPE", "bfloat16")

# --- Нарезка, в знаках ---
# Размер общий для обеих стратегий, иначе сравнивали бы размер, а не способ нарезки.
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
FIXED_OVERLAP = int(os.getenv("FIXED_OVERLAP", "150"))   # только у фиксированной

# --- Сеть ---
HOST = os.getenv("HOST", "127.0.0.1")
PORT = int(os.getenv("PORT", "8000"))
