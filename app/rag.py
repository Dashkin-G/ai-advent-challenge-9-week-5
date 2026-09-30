"""Агент с тремя режимами и прогон контрольных вопросов.

Без RAG модель отвечает по памяти. RAG: вопрос → эмбеддинг → 5 ближайших чанков
из индекса «по структуре» (он выиграл сравнение нарезок) → промпт «фрагменты
правил + вопрос» → модель. RAG + фильтр: перед промптом второй этап (rerank.py) —
переписанный вопрос, реранкер по 20 кандидатам и порог; не прошёл ни один чанк —
«в правилах ответа нет» без вызова модели. Инструкция во всех режимах одна и та же,
разница только в контексте — иначе сравнивались бы промпты, а не поиск.

Контрольные вопросы составлены вручную (control_questions.json): у каждого есть
ожидание — ключевые факты, которые должны быть в ответе, — и пункты-источники.
Прогон задаёт вопросы во всех режимах и сохраняет ответы как есть
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
from contextlib import closing
from datetime import datetime

from . import config, corpus, evaluate, index, llm, pipeline, rerank

MODES = {"plain": "Без RAG", "rag": "RAG", "rerank": "RAG + фильтр"}
LIVE = ("rag", "rerank")    # живой вопрос — два RAG рядом: без второго этапа поиска и с ним
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


def second_stage(question: str) -> Iterator[dict]:
    """Второй этап поиска событиями: rewrite → candidates → sources (прошли порог и отсеяны).
    seconds — время реранкера, stage — всего этапа, как его прождал вопрос (с кэшем — доли секунды).
    Возвращает чанки для контекста."""
    s = dict(rerank.settings)               # ползунок сдвинут посреди ответа — ответ досчитается по старым
    begun = time.perf_counter()
    rewritten = rerank.rewrite(question)
    yield {"type": "rewrite", **rewritten}
    started = time.perf_counter()
    hits = rerank.candidates(question, s["candidates"])
    yield {"type": "candidates", "count": len(hits), "ms": round(1000 * (time.perf_counter() - started))}
    started = time.perf_counter()
    passed, dropped = rerank.stage(question, rewritten["text"], hits, s["threshold"], s["top"])
    now = time.perf_counter()
    yield {"type": "sources", "hits": passed, "dropped": dropped, "settings": s,
           "seconds": round(now - started, 1), "stage": round(now - begun, 1)}
    return passed


def answer(question: str, mode: str) -> Iterator[dict]:
    """Ответ в режиме plain, rag или rerank — событиями по мере готовности:
    [rewrite → candidates →] sources (только с RAG) → prompt → think… → text… → done."""
    hits = None
    if mode == "rag":
        started = time.perf_counter()
        hits = retrieve(question)
        yield {"type": "sources", "hits": hits, "ms": round(1000 * (time.perf_counter() - started))}
    elif mode == "rerank":
        hits = yield from second_stage(question)
        if not hits:                        # ни один чанк не прошёл порог — модель звать незачем
            yield {"type": "done", "answer": NOT_FOUND, "usage": {}, "seconds": 0, "skipped": True}
            return
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
_run = {"running": False, "done": 0, "total": 0, "started": None, "finished": None, "error": None,
        "stopping": False, "stopped": False}
_live: dict | None = None           # идущий или последний прогон этого запуска приложения
_lock = threading.Lock()
_stop = threading.Event()


class Stopped(Exception):
    """Прогон остановлен кнопкой."""


def state() -> dict:
    return {"revision": revision, **_run}


def running() -> bool:
    return _run["running"]


def stop() -> None:
    """Остановить прогон: ответы, что идут, обрываются, недоделанный прогон не сохраняется."""
    global revision
    if _run["running"]:
        _stop.set()
        _run["stopping"] = True
        revision += 1


def _collect(question: str, mode: str) -> dict:
    """Один ответ целиком. Сбой сети или перегрузка модели — до двух повторов.
    Остановили прогон — ответ обрывается; closing закрывает и соединение, модель перестаёт писать."""
    for attempt in range(3):
        got = {"think": ""}
        if _stop.is_set():                              # ответ ещё ждал своей очереди
            raise Stopped
        try:
            with closing(answer(question, mode)) as events:
                for event in events:
                    if _stop.is_set():
                        raise Stopped
                    kind = event["type"]
                    if kind == "rewrite":
                        got["rewrite"] = {k: event[k] for k in ("text", "tokens", "seconds")}
                    elif kind == "sources":
                        got["hits"] = event["hits"]
                        if "dropped" in event:
                            got.update(dropped=event["dropped"], settings=event["settings"], stage_seconds=event["stage"])
                    elif kind == "think":
                        got["think"] += event["text"]
                    elif kind == "done":
                        got.update(answer=event["answer"], usage=event["usage"], seconds=event["seconds"])
                        if event.get("skipped"):
                            got["skipped"] = True
            return got
        except llm.LLMError as e:
            if not e.transient or attempt == 2:
                raise
            if _stop.wait(5 * (attempt + 1)):           # остановили во время паузы — не повторяем
                raise Stopped


def _judge(q: dict, got: dict, spans: list[tuple[int, int]]) -> dict:
    """Ответ против ожидания: какие факты есть, назван ли пункт, нашёл ли его поиск,
    а если нашёл, но не отдал модели, — отсёк ли его фильтр (cut)."""
    text = got["answer"]
    out = {**got, "facts": evaluate.facts(text, q["facts"]),
           "cited": [evaluate.cited(text, s["number"]) for s in q["sources"]]}
    if "hits" in got:
        for h in got["hits"] + got.get("dropped", []):
            h["relevant"] = bool(spans) and evaluate.relevant((h["start"], h["end"]), spans)
            h["marks"] = evaluate.marks((h["start"], h["end"]), spans)
        has = lambda hits, s: any(evaluate.relevant((h["start"], h["end"]), [s]) for h in hits)
        out["found"] = [has(got["hits"], s) for s in spans]
        if "dropped" in got:
            out["cut"] = [not f and has(got["dropped"], s) for s, f in zip(spans, out["found"])]
    return out


def _spans(questions: list[dict]) -> list[list[tuple[int, int]]]:
    text = config.DOC_PATH.read_text(encoding="utf-8")
    units = corpus.structure(text)
    return [[_span(units, text, s) for s in q["sources"]] for q in questions]


def _saved() -> dict | None:
    """Идущий или последний прогон — из памяти или с диска."""
    with _lock:
        rep = copy.deepcopy(_live)
    if rep is None and REPORT.exists():
        rep = json.loads(REPORT.read_text(encoding="utf-8"))
    if rep is not None and "runs" not in rep:       # у прогона дня 22 дата общая на оба режима
        rep["runs"] = {m: rep["started"] for m in MODES if any(m in a for a in rep["answers"].values())}
    return rep


def pending(questions: list[dict]) -> list[tuple[str, str]]:
    """Что задать модели: ответы, которых нет, и все ответы RAG + фильтр — его настройки
    двигаются ползунками, поэтому он перепрогоняется, а два других режима — нет."""
    answers = (_saved() or {"answers": {}})["answers"]
    return [(q["question"], mode) for q in questions for mode in MODES
            if mode == "rerank" or "answer" not in answers.get(q["question"], {}).get(mode, {})]


def run() -> None:
    """Прогон: недостающие ответы и режим RAG + фильтр. Ответы видны по мере прихода,
    а на диск прогон пишется целиком в конце — остановленный не затирает прежний."""
    global _live, revision
    _stop.clear()
    with _lock:
        _run.update(running=True, done=0, total=0, started=time.time(), finished=None, error=None,
                    stopping=False, stopped=False)
        revision += 1
    try:
        questions = control()
        _spans(questions)                               # источники есть в тексте — иначе стоп до трат
        asks = pending(questions)
        rep = _saved() or {"answers": {}, "runs": {}}
        now = datetime.now().isoformat(timespec="seconds")
        for question, mode in asks:
            rep["answers"].setdefault(question, {}).pop(mode, None)
            rep["runs"][mode] = now
        rep.update(model=config.LLM_MODEL, strategy=STRATEGY, top=evaluate.TOP, started=now, finished=None)
        with _lock:
            _run["total"] = len(asks)
            _live = rep
            revision += 1
        with ThreadPoolExecutor(WORKERS) as pool:
            jobs = {pool.submit(_collect, question, mode): (question, mode) for question, mode in asks}
            for job in as_completed(jobs):
                question, mode = jobs[job]
                try:
                    result = job.result()
                except Stopped:
                    continue
                except Exception as e:                  # причина — в ячейке таблицы
                    result = {"error": str(e)}
                with _lock:
                    _live["answers"][question][mode] = result
                    _run["done"] += 1
                    revision += 1
        with _lock:
            if _stop.is_set():                          # остановлен — в таблице снова прежний прогон с диска
                _live = None
                _run["stopped"] = True
                return
            _live["finished"] = datetime.now().isoformat(timespec="seconds")
            REPORT.parent.mkdir(parents=True, exist_ok=True)
            REPORT.write_text(json.dumps(_live, ensure_ascii=False), encoding="utf-8")
    except Exception as e:
        traceback.print_exc()
        _run["error"] = str(e)
    finally:
        with _lock:
            _run.update(running=False, stopping=False, finished=time.time())
            revision += 1


def report() -> dict | None:
    """Идущий или последний прогон: ответы проверяются по текущему набору вопросов, плюс итог.
    Ответ ищется по тексту вопроса — изменённый вопрос честно остаётся без ответа."""
    rep = _saved()
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
    """Итог по режимам: верные ответы, найденные факты, названные и найденные пункты, цена.
    В цену RAG + фильтр входит переписывание вопроса (даже взятое из кэша — оно оплачено),
    во время — второй этап так, как его прождал ответ."""
    out = {}
    for mode in MODES:
        done = [q["answers"][mode] for q in questions if "facts" in q["answers"].get(mode, {})]
        usage = [a.get("usage") or {} for a in done]
        extra = [(a.get("rewrite") or {}).get("tokens", 0) for a in done]
        stage = [a.get("stage_seconds", 0) for a in done]
        out[mode] = {
            "answered": len(done),
            "correct": sum(correct(a) for a in done),
            "facts": sum(f["ok"] for a in done for f in a["facts"]),
            "facts_total": sum(len(q["facts"]) for q in questions),
            "cited": sum(sum(a["cited"]) for a in done),
            "found": sum(sum(a.get("found", [])) for a in done),
            "cut": sum(sum(a.get("cut", [])) for a in done),
            "sources_total": sum(len(q["sources"]) for q in questions),
            "chunks": round(statistics.mean(len(a["hits"]) for a in done), 1) if done and "hits" in done[0] else 0,
            "skipped": sum(bool(a.get("skipped")) for a in done),        # отказ без вызова модели
            # Цена ответа: вход (с RAG — плюс фрагменты) и выход, куда входят размышления модели.
            "tokens_in": _mean(u.get("prompt_tokens", 0) for u in usage),
            "tokens_out": _mean(u.get("completion_tokens", 0) for u in usage),
            "tokens_think": _mean((u.get("completion_tokens_details") or {}).get("reasoning_tokens", 0) for u in usage),
            "tokens_rewrite": _mean(extra),
            "tokens_total": _mean(u.get("total_tokens", 0) + x for u, x in zip(usage, extra)),
            "tokens": sum(u.get("total_tokens", 0) + x for u, x in zip(usage, extra)),
            "seconds": round(statistics.mean(a["seconds"] + s for a, s in zip(done, stage)), 1) if done else 0,
            "seconds_stage": round(statistics.mean(stage), 1) if done else 0,
        }
    # Каждый режим — против предыдущего: RAG против памяти, фильтр против RAG без него.
    right = {m: [correct(q["answers"].get(m)) for q in questions] for m in MODES}
    out["paired"] = {"rag": evaluate.paired(right["plain"], right["rag"]),
                     "rerank": evaluate.paired(right["rag"], right["rerank"])}
    return out
