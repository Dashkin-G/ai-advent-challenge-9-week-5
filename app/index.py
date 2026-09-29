"""Индекс: эмбеддинги bge-m3, векторы в FAISS, текст и метаданные чанков в SQLite.

На каждую стратегию свой файл FAISS (`fixed.faiss`, `structure.faiss`); номер
вектора в нём совпадает с полем `row` чанка в SQLite. Векторы нормированы, поэтому
скалярное произведение в IndexFlatIP — это косинусная близость. Поиск точный,
полным перебором: на трёхстах векторах приближённые индексы не нужны.

Эмбеддинги кэшируются по хешу текста: пересборка без изменений ничего не
пересчитывает, а после смены размера чанка считаются только новые куски.
"""
import hashlib
import json
import os
import sqlite3
import threading
import time
from datetime import datetime
from typing import Callable

import faiss
import numpy as np

from . import config
from .chunking import Chunk

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

DB = config.INDEX_DIR / "index.db"
BATCH = 16

SCHEMA = """
CREATE TABLE IF NOT EXISTS chunks (
    chunk_id TEXT PRIMARY KEY,
    strategy TEXT NOT NULL,
    row      INTEGER NOT NULL,          -- номер вектора в FAISS этой стратегии
    source   TEXT, title TEXT, section TEXT,
    points   TEXT,                      -- JSON: номера пунктов
    start    INTEGER, end INTEGER, chars INTEGER, tokens INTEGER,
    text     TEXT NOT NULL
);
CREATE UNIQUE INDEX IF NOT EXISTS chunks_row ON chunks(strategy, row);
CREATE TABLE IF NOT EXISTS embeddings (key TEXT PRIMARY KEY, vector BLOB NOT NULL);
CREATE TABLE IF NOT EXISTS builds (strategy TEXT PRIMARY KEY, info TEXT NOT NULL);
"""

_model = None
_model_lock = threading.Lock()
_encode_lock = threading.Lock()
_indexes: dict[str, faiss.Index] = {}


def _db() -> sqlite3.Connection:
    config.INDEX_DIR.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(DB)
    con.executescript(SCHEMA)
    return con


def _path(strategy: str):
    return config.INDEX_DIR / f"{strategy}.faiss"


# --- модель эмбеддингов --------------------------------------------------------

def model():
    """bge-m3 грузится один раз (~2,3 ГБ с диска, 10–20 с) и живёт в памяти процесса."""
    global _model
    with _model_lock:
        if _model is None:
            import torch                                             # тяжёлый импорт — только по делу
            from sentence_transformers import SentenceTransformer
            options = {"device": "cpu", "model_kwargs": {"torch_dtype": getattr(torch, config.EMBED_DTYPE)}}
            try:        # уже скачана — без похода в сеть за обновлениями
                _model = SentenceTransformer(config.EMBED_MODEL, local_files_only=True, **options)
            except Exception:
                _model = SentenceTransformer(config.EMBED_MODEL, **options)
    return _model


def loaded() -> bool:
    return _model is not None


def count_tokens(texts: list[str]) -> list[int]:
    ids = model().tokenizer(texts, add_special_tokens=False)["input_ids"]
    return [len(x) for x in ids]


def embed(texts: list[str], progress: Callable[[int, int, int], None] | None = None) -> np.ndarray:
    """Векторы для текстов. progress(готово, всего, из кэша)."""
    tag = f"{config.EMBED_MODEL}|{config.EMBED_DTYPE}"
    keys = [hashlib.sha256(f"{tag}\n{t}".encode()).hexdigest() for t in texts]
    with _db() as con:
        marks = ",".join("?" * len(keys))
        cached = dict(con.execute(f"SELECT key, vector FROM embeddings WHERE key IN ({marks})", keys))
    vectors: dict[str, np.ndarray] = {k: np.frombuffer(v, dtype=np.float32) for k, v in cached.items()}
    first: dict[str, int] = {}                  # одинаковые тексты считаем один раз
    for i, k in enumerate(keys):
        if k not in vectors:
            first.setdefault(k, i)
    todo = list(first.values())
    hits = len(texts) - len(todo)
    if progress:
        progress(hits, len(texts), hits)
    for n in range(0, len(todo), BATCH):
        part = todo[n:n + BATCH]
        with _encode_lock:                      # поиск и сборка могут прийти из разных потоков
            out = model().encode([texts[i] for i in part], batch_size=BATCH, normalize_embeddings=True,
                                 convert_to_numpy=True).astype(np.float32)
        with _db() as con:
            con.executemany("INSERT OR REPLACE INTO embeddings VALUES (?, ?)",
                            [(keys[i], v.tobytes()) for i, v in zip(part, out)])
        for i, v in zip(part, out):
            vectors[keys[i]] = v
        if progress:
            progress(hits + n + len(part), len(texts), hits)
    return np.stack([vectors[k] for k in keys])


# --- сборка и поиск -------------------------------------------------------------

def save(strategy: str, chunks: list[Chunk], vectors: np.ndarray, tokens: list[int]) -> dict:
    """Записать индекс стратегии: векторы — в FAISS, чанки с метаданными — в SQLite."""
    started = time.perf_counter()
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    # Через байты, а не faiss.write_index: тот на Windows спотыкается о кириллицу в пути.
    _path(strategy).write_bytes(faiss.serialize_index(index).tobytes())
    info = {
        "strategy": strategy,
        "chunks": len(chunks),
        "model": config.EMBED_MODEL,
        "dim": int(vectors.shape[1]),
        "size": config.CHUNK_SIZE,
        "overlap": config.FIXED_OVERLAP if strategy == "fixed" else 0,
        "built": datetime.now().isoformat(timespec="seconds"),
        "faiss_bytes": _path(strategy).stat().st_size,
    }
    with _db() as con:
        con.execute("DELETE FROM chunks WHERE strategy = ?", (strategy,))
        con.executemany(
            "INSERT INTO chunks VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [(c.chunk_id, strategy, row, c.source, c.title, c.section,
              json.dumps(c.points, ensure_ascii=False), c.start, c.end, c.end - c.start, n, c.text)
             for row, (c, n) in enumerate(zip(chunks, tokens))])
        info["seconds"] = round(time.perf_counter() - started, 2)
        con.execute("INSERT OR REPLACE INTO builds VALUES (?, ?)", (strategy, json.dumps(info)))
    _indexes[strategy] = index
    return info


def _index(strategy: str) -> faiss.Index:
    if strategy not in _indexes:
        raw = _path(strategy).read_bytes()
        _indexes[strategy] = faiss.deserialize_index(np.frombuffer(raw, dtype=np.uint8))
    return _indexes[strategy]


def search(strategy: str, queries: np.ndarray, k: int) -> list[list[dict]]:
    """Ближайшие чанки для каждого вектора запроса: [{score, chunk_id, …, text}]."""
    scores, rows = _index(strategy).search(queries, k)
    with _db() as con:
        con.row_factory = sqlite3.Row
        by_row = {r["row"]: dict(r) for r in con.execute(
            f"SELECT * FROM chunks WHERE strategy = ? AND row IN ({','.join('?' * rows.size)})",
            [strategy, *map(int, rows.ravel())])}
    out = []
    for q_scores, q_rows in zip(scores, rows):
        hits = []
        for score, row in zip(q_scores, q_rows):
            if row < 0:
                continue
            chunk = dict(by_row[int(row)])
            chunk["points"] = json.loads(chunk["points"])
            chunk["score"] = round(float(score), 4)
            hits.append(chunk)
        out.append(hits)
    return out


def builds() -> dict[str, dict]:
    """Что сейчас лежит в индексе — по стратегиям."""
    if not DB.exists():
        return {}
    with _db() as con:
        return {s: json.loads(info) for s, info in con.execute("SELECT strategy, info FROM builds")
                if _path(s).exists()}


def chunk_rows(strategy: str) -> list[dict]:
    with _db() as con:
        con.row_factory = sqlite3.Row
        return [dict(r) for r in con.execute(
            "SELECT chunk_id, start, end, chars, tokens, section, points FROM chunks "
            "WHERE strategy = ? ORDER BY row", (strategy,))]


def files() -> list[dict]:
    """Файлы индекса на диске — для карточки «Индекс»."""
    if not config.INDEX_DIR.is_dir():
        return []
    paths = [config.INDEX_DIR / f"{s}.faiss" for s in ("fixed", "structure")] + [DB]
    return [{"name": p.name, "bytes": p.stat().st_size} for p in paths if p.exists()]
