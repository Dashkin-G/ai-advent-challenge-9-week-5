"""Агент с двумя режимами и прогон контрольных вопросов.

Без RAG модель отвечает по памяти. С RAG: вопрос → эмбеддинг → 5 ближайших
чанков из индекса «по структуре» (он выиграл сравнение нарезок) → промпт
«фрагменты правил + вопрос» → модель. Инструкция в обоих режимах одна и та же,
разница только в контексте — иначе сравнивались бы промпты, а не RAG.

Контрольные вопросы составлены вручную (control_questions.json): у каждого есть
ожидание — ключевые факты, которые должны быть в ответе, — и пункты-источники.
Прогон задаёт каждый вопрос в обоих режимах и сохраняет ответы как есть
(data/eval/control_run.json). Проверка — есть ли факты, назван ли пункт, нашёл ли
его поиск — считается при чтении: поправили ожидание — вердикты пересчитаются
без нового похода к модели.
"""
import copy
import json
import re
import statistics
import threading
import time
import traceback
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

from . import config, corpus, evaluate, index, llm, pipeline

MODES = {"plain": "Без RAG", "rag": "С RAG"}
STRATEGY = "structure"
REPORT = config.DATA / "eval" / "control_run.json"
WORKERS = 5             # ответов модели одновременно: 20 ответов по 10–30 с укладываются в пару минут

SYSTEM = (
    "Ты — справочник по Правилам дорожного движения РФ. Отвечай по-русски и коротко: первое "
    "предложение — прямой ответ на вопрос, дальше, если нужно, условия и исключения; всего не больше "
    "пяти предложений. Пиши обычным текстом, без Markdown. После каждого утверждения указывай "
    "в квадратных скобках пункт, на котором оно основано, например [п. 10.2 ПДД] или "
    "[п. 5.4 Перечня неисправностей]."
)
NOT_FOUND = "В найденных пунктах правил ответа нет."
GROUNDED = ("Отвечай только по фрагментам правил из сообщения, ничего не добавляй от себя. "
            f"Если во фрагментах ответа нет, ответь одной фразой: «{NOT_FOUND}»")


# --- агент ------------------------------------------------------------------------

def retrieve(question: str) -> list[dict]:
    """Ближайшие к вопросу чанки — ровно те, что уйдут модели."""
    return index.search(STRATEGY, index.embed([question]), evaluate.TOP)[0]


def messages(question: str, hits: list[dict] | None) -> list[dict]:
    """Промпт. Без RAG (hits=None) — та же инструкция, только без фрагментов."""
    if hits is None:
        return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": question}]
    fragments = "\n\n".join(f"Фрагмент {i} — {' › '.join(filter(None, (h['title'], h['section'])))}\n{h['text']}"
                            for i, h in enumerate(hits, 1))
    return [{"role": "system", "content": f"{SYSTEM} {GROUNDED}"},
            {"role": "user", "content": f"Фрагменты правил:\n\n{fragments}\n\nВопрос: {question}"}]


def _plain(text: str) -> str:
    """Модель иногда всё же ставит Markdown — на экране он лишний."""
    return re.sub(r"\*\*|__|^#+\s+", "", text, flags=re.MULTILINE).strip()


def answer(question: str, mode: str) -> Iterator[dict]:
    """Ответ в режиме plain или rag — событиями по мере готовности:
    sources (найденные чанки, только с RAG) → prompt → think… → text… → done."""
    hits = None
    if mode == "rag":
        started = time.perf_counter()
        hits = retrieve(question)
        yield {"type": "sources", "hits": hits, "ms": round(1000 * (time.perf_counter() - started))}
    prompt = messages(question, hits)
    yield {"type": "prompt", "messages": prompt}
    started, parts, usage = time.perf_counter(), [], {}
    for kind, value in llm.stream(prompt):
        if kind == "usage":
            usage = value
            continue
        if kind == "text":
            parts.append(value)
        yield {"type": kind, "text": value}
    yield {"type": "done", "answer": _plain("".join(parts)), "usage": usage,
           "seconds": round(time.perf_counter() - started, 1)}


# --- контрольные вопросы ------------------------------------------------------------

def control() -> list[dict]:
    """Контрольные вопросы; у источников — подпись вида «п. 10.2 ПДД»."""
    questions = json.loads(config.CONTROL_PATH.read_text(encoding="utf-8"))
    for q in questions:
        for s in q["sources"]:
            s["label"] = pipeline.ref_label(s)
    return questions


def _span(units: list[corpus.Unit], text: str, source: dict) -> tuple[int, int]:
    """Где в тексте правил лежит источник: пункт целиком или, если задана цитата, только она.
    Нет в тексте — ошибка набора: прогон остановится до того, как потратит токены."""
    span = corpus.span(units, text, source)
    if span and source.get("quote"):
        at = text.find(source["quote"], *span)
        span = (at, at + len(source["quote"])) if at >= 0 else None
    if span is None:
        raise ValueError(f"в тексте правил нет источника: {source['label']}")
    return span


revision = 0            # растёт на каждый пришедший ответ — интерфейс перерисовывается только тогда
_run = {"running": False, "done": 0, "total": 0, "started": None, "finished": None, "error": None}
_live: dict | None = None           # идущий или последний прогон этого запуска приложения
_lock = threading.Lock()


def state() -> dict:
    return {"revision": revision, **_run}


def running() -> bool:
    return _run["running"]


def _collect(question: str, mode: str) -> dict:
    """Один ответ целиком. Сбой сети или перегрузка модели — до двух повторов."""
    for attempt in range(3):
        got = {"think": ""}
        try:
            for event in answer(question, mode):
                if event["type"] == "sources":
                    got["hits"] = event["hits"]
                elif event["type"] == "think":
                    got["think"] += event["text"]
                elif event["type"] == "done":
                    got.update(answer=event["answer"], usage=event["usage"], seconds=event["seconds"])
            return got
        except llm.LLMError as e:
            if not e.transient or attempt == 2:
                raise
            time.sleep(5 * (attempt + 1))


def _judge(q: dict, got: dict, spans: list[tuple[int, int]]) -> dict:
    """Ответ против ожидания: какие факты есть, назван ли пункт, нашёл ли его поиск."""
    text = got["answer"]
    out = {**got, "facts": evaluate.facts(text, q["facts"]),
           "cited": [evaluate.cited(text, s["number"]) for s in q["sources"]]}
    if "hits" in got:
        for h in got["hits"]:
            h["relevant"] = bool(spans) and evaluate.relevant((h["start"], h["end"]), spans)
            h["marks"] = evaluate.marks((h["start"], h["end"]), spans)
        out["found"] = [any(evaluate.relevant((h["start"], h["end"]), [s]) for h in got["hits"]) for s in spans]
    return out


def _spans(questions: list[dict]) -> list[list[tuple[int, int]]]:
    text = config.DOC_PATH.read_text(encoding="utf-8")
    units = corpus.structure(text)
    return [[_span(units, text, s) for s in q["sources"]] for q in questions]


def run() -> None:
    """Прогон: каждый вопрос в обоих режимах; ответы сохраняются по мере прихода."""
    global _live, revision
    with _lock:
        _run.update(running=True, done=0, total=0, started=time.time(), finished=None, error=None)
        revision += 1
    try:
        questions = control()
        _spans(questions)                               # источники есть в тексте — иначе стоп до трат
        with _lock:
            _run["total"] = len(questions) * len(MODES)
            _live = {"model": config.LLM_MODEL, "strategy": STRATEGY, "top": evaluate.TOP,
                     "started": datetime.now().isoformat(timespec="seconds"), "finished": None,
                     "answers": {q["question"]: {} for q in questions}}
            revision += 1
        with ThreadPoolExecutor(WORKERS) as pool:
            jobs = {pool.submit(_collect, q["question"], mode): (q["question"], mode)
                    for q in questions for mode in MODES}
            for job in as_completed(jobs):
                question, mode = jobs[job]
                try:
                    result = job.result()
                except Exception as e:                  # причина — в ячейке таблицы
                    result = {"error": str(e)}
                with _lock:
                    _live["answers"][question][mode] = result
                    _run["done"] += 1
                    revision += 1
        with _lock:
            _live["finished"] = datetime.now().isoformat(timespec="seconds")
            REPORT.parent.mkdir(parents=True, exist_ok=True)
            REPORT.write_text(json.dumps(_live, ensure_ascii=False), encoding="utf-8")
    except Exception as e:
        traceback.print_exc()
        _run["error"] = str(e)
    finally:
        with _lock:
            _run.update(running=False, finished=time.time())
            revision += 1


def report() -> dict | None:
    """Идущий или последний прогон: ответы проверяются по текущему набору вопросов, плюс итог.
    Ответ ищется по тексту вопроса — изменённый вопрос честно остаётся без ответа."""
    with _lock:
        rep = copy.deepcopy(_live)
    if rep is None and REPORT.exists():
        rep = json.loads(REPORT.read_text(encoding="utf-8"))
    if rep is None:
        return None
    answers = rep.pop("answers")
    questions = control()
    for q, spans in zip(questions, _spans(questions)):
        got = answers.get(q["question"], {})
        q["answers"] = {mode: _judge(q, a, spans) if "answer" in a else a for mode, a in got.items()}
    return {**rep, "questions": questions, "summary": summary(questions)}


def _mean(values) -> int:
    values = list(values)
    return round(statistics.mean(values)) if values else 0


def correct(a: dict | None) -> bool:
    """Ответ верный — в нём все ключевые факты из ожидания."""
    return bool(a and a.get("facts")) and all(f["ok"] for f in a["facts"])


def summary(questions: list[dict]) -> dict:
    """Итог по режимам: верные ответы, найденные факты, названные и найденные пункты, цена."""
    out = {}
    for mode in MODES:
        done = [q["answers"][mode] for q in questions if "facts" in q["answers"].get(mode, {})]
        usage = [a.get("usage") or {} for a in done]
        out[mode] = {
            "answered": len(done),
            "correct": sum(correct(a) for a in done),
            "facts": sum(f["ok"] for a in done for f in a["facts"]),
            "facts_total": sum(len(q["facts"]) for q in questions),
            "cited": sum(sum(a["cited"]) for a in done),
            "found": sum(sum(a.get("found", [])) for a in done),
            "sources_total": sum(len(q["sources"]) for q in questions),
            # Цена ответа: вход (с RAG — плюс фрагменты) и выход, куда входят размышления модели.
            "tokens_in": _mean(u.get("prompt_tokens", 0) for u in usage),
            "tokens_out": _mean(u.get("completion_tokens", 0) for u in usage),
            "tokens_think": _mean((u.get("completion_tokens_details") or {}).get("reasoning_tokens", 0) for u in usage),
            "tokens_total": _mean(u.get("total_tokens", 0) for u in usage),
            "tokens": sum(u.get("total_tokens", 0) for u in usage),
            "seconds": round(statistics.mean(a["seconds"] for a in done), 1) if done else 0,
        }
    out["paired"] = evaluate.paired(*([correct(q["answers"].get(m)) for q in questions] for m in MODES))
    return out
