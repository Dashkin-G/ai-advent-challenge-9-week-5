"""Второй этап поиска: переписанный вопрос, реранкер и порог.

Векторный поиск быстрый, но грубый: эмбеддинги вопроса и чанка считаются порознь,
и нужный пункт нередко оказывается не первым. Кросс-энкодер bge-reranker-v2-m3
читает вопрос и чанк вместе и оценивает от 0 до 1, отвечает ли чанк на вопрос, —
точнее, но в сотни раз медленнее, поэтому он перечитывает только 20 кандидатов.

Вопрос → модель переписывает его терминами правил → 20 ближайших чанков по косинусу →
реранкер читает пары «вопрос + переписанный вопрос — чанк» → в контекст идут чанки
с оценкой не ниже порога, не больше трёх. Не прошёл ни один — в правилах ответа нет,
модель можно не звать.

Проверка — на двух наборах: вопросы с ответом (130 билетов и 9 контрольных) и вопросы,
ответа на которые в правилах нет (20 вне базы и один контрольный). Оценки реранкера
и переписанные вопросы кэшируются в SQLite индекса: повторный прогон занимает секунды.
"""
import hashlib
import json
import sqlite3
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import Callable

from . import config, corpus, evaluate, index, llm, pipeline

SCHEMA = """
CREATE TABLE IF NOT EXISTS rerank (key TEXT PRIMARY KEY, score REAL NOT NULL);
CREATE TABLE IF NOT EXISTS rewrites (key TEXT PRIMARY KEY, question TEXT NOT NULL, text TEXT NOT NULL,
                                     tokens INTEGER, seconds REAL);
"""
STRATEGY = "structure"
BATCH = 16
REPORT = config.DATA / "eval" / "rerank.json"
REWRITE = ("Перепиши вопрос водителя терминами Правил дорожного движения РФ для поиска по их тексту. "
           "Не отвечай на вопрос и не добавляй нового. Ответь одной строкой — только запрос.")

# Настройки агента; меняются ползунками на вкладке «Фильтр» и живут до перезапуска.
DEFAULTS = {"candidates": config.CANDIDATES, "threshold": config.THRESHOLD, "top": config.CONTEXT}
settings = dict(DEFAULTS)

_model = None
_model_lock = threading.Lock()
_predict_lock = threading.Lock()


def _db() -> sqlite3.Connection:
    config.INDEX_DIR.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(index.DB)
    con.executescript(SCHEMA)
    return con


# --- реранкер ---------------------------------------------------------------------

def model():
    """bge-reranker-v2-m3 (~2,3 ГБ) грузится один раз и живёт в памяти процесса."""
    global _model
    with _model_lock:
        if _model is None:
            import torch                                             # тяжёлый импорт — только по делу
            from sentence_transformers import CrossEncoder
            options = {"device": "cpu", "max_length": 512,
                       "model_kwargs": {"torch_dtype": getattr(torch, config.EMBED_DTYPE)}}
            try:        # уже скачана — без похода в сеть за обновлениями
                _model = CrossEncoder(config.RERANK_MODEL, local_files_only=True, **options)
            except Exception:
                _model = CrossEncoder(config.RERANK_MODEL, **options)
    return _model


def loaded() -> bool:
    return _model is not None


def score(pairs: list[tuple[str, str]], progress: Callable[[int, int], None] | None = None) -> list[float]:
    """Оценка реранкера для пар (запрос, текст чанка). progress(готово, всего)."""
    tag = f"{config.RERANK_MODEL}|{config.EMBED_DTYPE}"
    keys = [hashlib.sha256(f"{tag}\n{q}\n{t}".encode()).hexdigest() for q, t in pairs]
    cached: dict[str, float] = {}
    with _db() as con:
        for n in range(0, len(keys), 900):                  # предел параметров SQLite
            part = keys[n:n + 900]
            cached.update(con.execute(f"SELECT key, score FROM rerank WHERE key IN ({','.join('?' * len(part))})", part))
    first: dict[str, int] = {}
    for i, k in enumerate(keys):
        if k not in cached:
            first.setdefault(k, i)
    # Похожие по длине пары — в одну пачку: в пачке всё добивается до самой длинной.
    todo = sorted(first.values(), key=lambda i: len(pairs[i][0]) + len(pairs[i][1]))
    done = len(pairs) - len(todo)
    if progress:
        progress(done, len(pairs))
    for n in range(0, len(todo), BATCH):
        part = todo[n:n + BATCH]
        with _predict_lock:                     # живой вопрос и прогон проверки — из разных потоков
            out = model().predict([pairs[i] for i in part], batch_size=BATCH, show_progress_bar=False)
        with _db() as con:
            con.executemany("INSERT OR REPLACE INTO rerank VALUES (?, ?)",
                            [(keys[i], float(s)) for i, s in zip(part, out)])
        cached.update((keys[i], float(s)) for i, s in zip(part, out))
        if progress:
            progress(done + n + len(part), len(pairs))
    return [round(cached[k], 4) for k in keys]


# --- переписанный вопрос --------------------------------------------------------------

def rewrite(question: str) -> dict:
    """Вопрос терминами правил. Думать модели дают совсем немного: задача простая, а без
    ограничения она размышляет по тысяче токенов. Одинаковый вопрос переписывается один раз."""
    key = hashlib.sha256(f"{config.LLM_MODEL}|{config.REWRITE_BUDGET}|{REWRITE}\n{question}".encode()).hexdigest()
    with _db() as con:
        row = con.execute("SELECT text, tokens, seconds FROM rewrites WHERE key = ?", (key,)).fetchone()
    if row:
        return {"text": row[0], "tokens": row[1], "seconds": row[2], "cached": True}
    started, parts, usage = time.perf_counter(), [], {}
    messages = [{"role": "system", "content": REWRITE}, {"role": "user", "content": question}]
    for kind, value in llm.stream(messages, budget=config.REWRITE_BUDGET):
        if kind == "text":
            parts.append(value)
        elif kind == "usage":
            usage = value
    got = {"text": " ".join("".join(parts).split()) or question, "tokens": usage.get("total_tokens", 0),
           "seconds": round(time.perf_counter() - started, 1)}
    with _db() as con:
        con.execute("INSERT OR REPLACE INTO rewrites VALUES (?, ?, ?, ?, ?)",
                    (key, question, got["text"], got["tokens"], got["seconds"]))
    return {**got, "cached": False}


def query(question: str, rewritten: str) -> str:
    """Что читает реранкер: вопрос как задан и он же терминами правил. По одному исходному
    вопросу у разговорной формулировки («права по QR-коду») верный пункт получает 0,01,
    по одному переписанному вопрос вне базы («штраф за красный свет») выглядит ответимым."""
    return f"{question} {rewritten}"


# --- второй этап ------------------------------------------------------------------

def candidates(question: str, k: int) -> list[dict]:
    """Первый этап: k ближайших чанков по косинусу; vrank — их место в этой выдаче."""
    hits = index.search(STRATEGY, index.embed([question]), k)[0]
    for vrank, h in enumerate(hits, 1):
        h["cos"], h["vrank"] = h.pop("score"), vrank
    return hits


def stage(question: str, rewritten: str, hits: list[dict], threshold: float, top: int) -> tuple[list, list]:
    """Переоценить кандидатов реранкером и отсечь: (в контекст, отсеяно) — по убыванию оценки.
    У отсеянного причина: ниже порога или не вошёл в первые top."""
    q = query(question, rewritten)
    for h, s in zip(hits, score([(q, h["text"]) for h in hits])):
        h["score"] = s
    order = sorted(hits, key=lambda h: -h["score"])
    passed = [h for h in order if h["score"] >= threshold][:top]
    dropped = [{**h, "why": "threshold" if h["score"] < threshold else "top"} for h in order if h not in passed]
    return passed, dropped


# --- проверка на вопросах с ответом и без --------------------------------------------

revision = 0            # растёт на каждое изменение — интерфейс перерисовывается только тогда
_run = {"running": False, "stage": "", "done": 0, "total": 0, "started": None, "finished": None, "error": None}


def state() -> dict:
    return {"revision": revision, **_run, "loaded": loaded()}


def running() -> bool:
    return _run["running"]


def _step(stage: str, done: int = 0, total: int = 0) -> None:
    global revision
    _run.update(stage=stage, done=done, total=total)
    revision += 1


def _questions(text: str, units: list[corpus.Unit]) -> list[dict]:
    """С ответом — билеты (эталон дня 21) и контрольные с источником; без ответа — вопросы
    вне базы и контрольный без источника. golden — где в тексте лежит ответ."""
    from . import rag                       # rag сам опирается на этот модуль
    out = []
    for q in json.loads(config.QUESTIONS_PATH.read_text(encoding="utf-8")):
        out.append({"question": q["question"], "group": "tickets", "label": ", ".join(map(pipeline.ref_label, q["refs"])),
                    "golden": [corpus.span(units, text, r) for r in q["refs"]]})
    control = rag.control()
    for i, (q, spans) in enumerate(zip(control, rag._spans(control)), 1):
        out.append({"question": q["question"], "group": "control", "n": i,     # без источника — вопрос про штраф
                    "label": "; ".join(s["label"] for s in q["sources"]) or "КоАП", "golden": spans})
    for q in json.loads(config.OUTSIDE_PATH.read_text(encoding="utf-8")):
        out.append({"question": q["question"], "group": "outside", "label": q["where"], "golden": []})
    return out


def run() -> None:
    """Прогон проверки: переписать вопросы, найти кандидатов, оценить их реранкером дважды —
    по одному вопросу и по вопросу с переписанным. Метрики при любых настройках считает
    интерфейс: в отчёте лежат все оценки."""
    global revision
    _run.update(running=True, started=time.time(), finished=None, error=None)
    _step("готовлю вопросы")
    try:
        text = config.DOC_PATH.read_text(encoding="utf-8")
        items = _questions(text, corpus.structure(text))
        tokens = spent = 0                              # цена переписывания всего и в этом прогоне
        with ThreadPoolExecutor(5) as pool:             # модель отвечает секунды — параллельно быстрее
            for i, got in enumerate(pool.map(rewrite, [it["question"] for it in items]), 1):
                items[i - 1]["rewrite"] = got["text"]
                tokens += got["tokens"]
                spent += 0 if got["cached"] else got["tokens"]
                _step("переписываю вопросы", i, len(items))
        _step("ищу кандидатов")
        found = index.search(STRATEGY, index.embed([it["question"] for it in items]), config.CANDIDATES)
        if not loaded():
            _step("загружаю реранкер")
        pairs = [(it["question"], h["text"]) for it, hs in zip(items, found) for h in hs]
        pairs += [(query(it["question"], it["rewrite"]), h["text"]) for it, hs in zip(items, found) for h in hs]
        scores = score(pairs, lambda done, total: _step("реранкер читает пары", done, total))
        half, at = len(pairs) // 2, 0
        for it, hs in zip(items, found):
            golden = it.pop("golden")
            it["answerable"] = bool(golden)
            # Кандидат: косинус, оценка по вопросу, оценка по вопросу с переписанным, нужный ли, знаков.
            it["c"] = []
            for h in hs:
                relevant = bool(golden) and evaluate.relevant((h["start"], h["end"]), golden)
                it["c"].append([h["score"], scores[at], scores[half + at], int(relevant), h["chars"]])
                at += 1
        report = {"built": datetime.now().isoformat(timespec="seconds"), "model": config.RERANK_MODEL,
                  "llm": config.LLM_MODEL, "candidates": config.CANDIDATES, "tokens": tokens, "spent": spent,
                  "pairs": len(pairs), "seconds": round(time.time() - _run["started"]), "questions": items}
        REPORT.parent.mkdir(parents=True, exist_ok=True)
        REPORT.write_text(json.dumps(report, ensure_ascii=False), encoding="utf-8")
        _step("готово")
    except Exception as e:                                  # причина — словами, в строке состояния
        traceback.print_exc()
        _run["error"] = str(e)
    finally:
        _run.update(running=False, finished=time.time())
        revision += 1


def report() -> dict | None:
    return json.loads(REPORT.read_text(encoding="utf-8")) if REPORT.exists() else None
