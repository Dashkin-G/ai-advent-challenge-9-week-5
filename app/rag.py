"""Агент: ответ с источниками и цитатами, прогон контрольных вопросов.

Вопрос → второй этап поиска (rerank.py): переписанный вопрос, 20 ближайших чанков по
косинусу, реранкер и порог → модель. Не прошёл порог ни один чанк — агент отвечает
«не знаю» и просит уточнить вопрос, модель не зовётся.

Режимов два, контекст у них один и тот же — разница в том, что обязана вернуть модель:
- RAG + фильтр (день 23) — ответ со ссылками на пункты в квадратных скобках;
- с цитатами — JSON: ответ со сносками [1], [2] и цитаты — выдержки из чанков с их
  chunk_id. Программа проверяет, что каждая цитата слово в слово лежит в названном чанке,
  и по месту цитаты находит её пункт. Источник — чанк: chunk_id, документ, раздел, пункт.
  Ответа во фрагментах нет — модель тоже говорит «не знаю» и задаёт уточняющий вопрос.

Контрольные вопросы составлены вручную (control_questions.json): у каждого есть
ожидание — ключевые факты, которые должны быть в ответе, — и пункты-источники.
Прогон сохраняет ответы как есть (data/eval/control_run.json). Проверка — есть ли
факты, назван ли пункт, нашёл ли его поиск, дословны ли цитаты и подтверждают ли они
факты ответа — считается при чтении: поправили ожидание — вердикты пересчитаются
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

from . import config, corpus, evaluate, llm, pipeline, rerank

MODES = {"rerank": "RAG + фильтр", "cite": "С цитатами"}
LIVE = ("cite",)        # живой вопрос задаётся агенту с цитатами
STRATEGY = "structure"
REPORT = config.DATA / "eval" / "control_run.json"
WORKERS = 5             # ответов модели одновременно: 20 ответов по 10–30 с укладываются в пару минут

# День 23: ответ со ссылками на пункты.
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

# День 24: ответ, источники и цитаты; слабый контекст — «не знаю» и просьба уточнить.
CITE = (
    "Ты — справочник по Правилам дорожного движения РФ. Отвечай по-русски и только по фрагментам правил "
    "из сообщения, ничего не добавляй от себя. Верни JSON: "
    '{"answer": "…", "quotes": [{"chunk_id": "…", "text": "…"}]}. '
    "answer — первое предложение прямо отвечает на вопрос, дальше, если нужно, условия и исключения; всего "
    "не больше пяти предложений, обычным текстом. После каждого утверждения — номер цитаты, на которой оно "
    "держится: [1], [2]. quotes — цитаты в порядке номеров: text — кусок фрагмента слово в слово, без "
    "пересказа (предложение или его часть, пропуск внутри отметь «…»), chunk_id — из заголовка фрагмента. "
    "В цитатах должно быть всё, на чём держится ответ: числа, условия, перечни. "
    'Если во фрагментах ответа нет, верни {"answer": "Не знаю: <почему — одной фразой>", '
    '"clarify": "<один короткий вопрос: может быть, водитель спрашивал о том, что во фрагментах есть>", '
    '"quotes": []}. По памяти ничего не подсказывай.'
)
DONT_KNOW = "Не знаю: среди пунктов Правил не нашлось подходящего к вопросу."
CLARIFY = "Уточните, пожалуйста, вопрос: я отвечаю только по тексту Правил дорожного движения и приложений к ним."


# --- агент ------------------------------------------------------------------------

def messages(question: str, hits: list[dict], mode: str) -> list[dict]:
    """Промпт: инструкция режима, фрагменты правил с заголовком «часть › раздел» и вопрос.
    С цитатами заголовок начинается с chunk_id — им модель называет источник цитаты."""
    def head(i: int, h: dict) -> str:
        where = " › ".join(filter(None, (h["title"], h["section"])))
        return f"[{h['chunk_id']}] {where}" if mode == "cite" else f"Фрагмент {i} — {where}"
    fragments = "\n\n".join(f"{head(i, h)}\n{h['text']}" for i, h in enumerate(hits, 1))
    return [{"role": "system", "content": CITE if mode == "cite" else f"{SYSTEM} {GROUNDED}"},
            {"role": "user", "content": f"Фрагменты правил:\n\n{fragments}\n\nВопрос: {question}"}]


def _plain(text: str) -> str:
    """Модель иногда всё же ставит Markdown — на экране он лишний."""
    return re.sub(r"\*\*|__|^#+\s+", "", text, flags=re.MULTILINE).strip()


def _parse(raw: str) -> dict:
    """Ответ в режиме с цитатами. JSON не разобрался — весь текст считается ответом
    без цитат: проверка так и покажет, что источников и цитат нет."""
    try:
        got = json.loads(raw[raw.index("{"):raw.rindex("}") + 1])
    except ValueError:
        got = None
    if not isinstance(got, dict) or not isinstance(got.get("answer"), str):
        return {"answer": _plain(raw), "clarify": "", "quotes": []}
    quotes = [{"chunk_id": str(q.get("chunk_id", "")), "text": q["text"]}
              for q in got.get("quotes") or [] if isinstance(q, dict) and isinstance(q.get("text"), str)]
    return {"answer": _plain(got["answer"]), "clarify": str(got.get("clarify") or "").strip(), "quotes": quotes}


def _label(hit: dict, at: int) -> str:
    """Пункт, в котором лежит место чанка: последний номер абзаца до него, а если чанк
    начат посреди пункта, — первый пункт чанка. «п. 6.2 ПДД», «знак 3.20»."""
    number = hit["points"][0] if hit["points"] else ""
    end = hit["text"].find("\n", at)
    for line in hit["text"][:end if end >= 0 else None].split("\n"):
        m = corpus.NUMBER.match(line)
        if m:
            number = m.group(1)
    part = corpus.PART_BY_TITLE.get(hit["title"])
    return pipeline.ref_label({"part": part, "number": number}) if part and number else hit["section"]


def verify(got: dict, hits: list[dict]) -> dict:
    """Проверка ответа с цитатами по чанкам, ушедшим модели. У цитаты: в каком чанке она
    нашлась (found; None — такой фразы нет ни в одном), тот ли это чанк, что назвала модель
    (own), где она в чанке (at) и в тексте правил (start, end), в каком пункте (label).
    У ответа — какие его числа есть в найденных цитатах."""
    by_id = {h["chunk_id"]: h for h in hits}
    quotes = []
    for q in got["quotes"]:
        named = by_id.get(q["chunk_id"])
        found = None
        for h in ([named] if named else []) + [h for h in hits if h is not named]:     # сначала названный
            at = evaluate.locate(q["text"], h["text"])
            if at:
                found = h, at
                break
        if not found:
            quotes.append({**q, "found": None, "own": False})
            continue
        h, (a, b) = found
        quotes.append({**q, "found": h["chunk_id"], "own": h is named, "at": [a, b],
                       "start": h["start"] + a, "end": h["start"] + b, "label": _label(h, a)})
    real = [q["text"] for q in quotes if q["found"]]
    return {"quotes": quotes, "numbers": evaluate.numbers(got["answer"], real)}


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
    """Ответ в режиме rerank или cite — событиями по мере готовности:
    rewrite → candidates → sources → prompt → think… → text… → done.
    Порог не прошёл ни один чанк — сразу done без модели (skipped): «не знаю» и просьба
    уточнить (у режима дня 23 — «в найденных пунктах ответа нет»)."""
    hits = yield from second_stage(question)
    if not hits:
        said = {"answer": DONT_KNOW, "clarify": CLARIFY, "quotes": []} if mode == "cite" else {"answer": NOT_FOUND}
        yield {"type": "done", **said, "usage": {}, "seconds": 0, "skipped": True, "unknown": True}
        return
    prompt = messages(question, hits, mode)
    yield {"type": "prompt", "messages": prompt}
    started, parts, usage = time.perf_counter(), [], {}
    for kind, value in llm.stream(prompt, as_json=mode == "cite"):
        if kind == "usage":
            usage = value
            continue
        if kind == "text":
            parts.append(value)
        yield {"type": kind, "text": value}
    said = _parse("".join(parts)) if mode == "cite" else {"answer": _plain("".join(parts))}
    done = {"type": "done", **said, "usage": usage, "seconds": round(time.perf_counter() - started, 1),
            "unknown": unknown(said)}
    if mode == "cite":
        done.update(verify(said, hits))
    yield done


def unknown(a: dict) -> bool:
    """Агент не ответил: «не знаю» (режим дня 23 говорил «в найденных пунктах ответа нет»)."""
    text = a["answer"].lower()
    return text.startswith("не знаю") or text.startswith(NOT_FOUND.lower()[:-1])


# --- контрольные вопросы ------------------------------------------------------------

def control() -> list[dict]:
    """Контрольные вопросы; у источников — подпись вида «п. 10.2 ПДД»."""
    questions = json.loads(config.CONTROL_PATH.read_text(encoding="utf-8"))
    for q in questions:
        for s in q["sources"]:
            s["label"] = pipeline.ref_label(s)
    return questions


def _span(units: list[corpus.Unit], text: str, source: dict, whole: bool) -> tuple[int, int]:
    """Где в тексте правил лежит источник: пункт целиком или, если задана цитата и не нужен
    весь пункт (whole), только она. Нет в тексте — ошибка набора: прогон остановится до трат."""
    span = corpus.span(units, text, source)
    if span and source.get("quote") and not whole:
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
                        got.update(hits=event["hits"], dropped=event["dropped"], settings=event["settings"],
                                   stage_seconds=event["stage"])
                    elif kind == "think":
                        got["think"] += event["text"]
                    elif kind == "done":
                        got.update(answer=event["answer"], usage=event["usage"], seconds=event["seconds"])
                        if "quotes" in event:           # цитаты — как их вернула модель, проверка при чтении
                            got.update(clarify=event["clarify"],
                                       quotes=[{k: q[k] for k in ("chunk_id", "text")} for q in event["quotes"]])
                        if event.get("skipped"):
                            got["skipped"] = True
            return got
        except llm.LLMError as e:
            if not e.transient or attempt == 2:
                raise
            if _stop.wait(5 * (attempt + 1)):           # остановили во время паузы — не повторяем
                raise Stopped


def _judge(q: dict, got: dict, spans: list[tuple[int, int]], points: list[tuple[int, int]]) -> dict:
    """Ответ против ожидания: какие факты есть, назван ли пункт, нашёл ли его поиск, а если
    нашёл, но не отдал модели, — отсёк ли его фильтр (cut). У ответа с цитатами пункт не
    называется, а цитируется: цитата должна лежать в нём (points — пункты целиком); ещё
    проверяется, есть ли каждый факт ответа в его цитатах (backed; None — факта нет в ответе)."""
    text = got["answer"]
    out = {**got, "facts": evaluate.facts(text, q["facts"]), "unknown": unknown(got)}
    for h in got["hits"] + got["dropped"]:
        h["relevant"] = bool(spans) and evaluate.relevant((h["start"], h["end"]), spans)
        h["marks"] = evaluate.marks((h["start"], h["end"]), spans)
    has = lambda hits, s: any(evaluate.relevant((h["start"], h["end"]), [s]) for h in hits)
    out["found"] = [has(got["hits"], s) for s in spans]
    out["cut"] = [not f and has(got["dropped"], s) for s, f in zip(spans, out["found"])]
    if "quotes" not in got:
        out["cited"] = [evaluate.cited(text, s["number"]) for s in q["sources"]]
        return out
    out.update(verify(got, got["hits"]))
    real = [x for x in out["quotes"] if x["found"]]
    out["cited"] = [any(evaluate.relevant((x["start"], x["end"]), [p]) for x in real) for p in points]
    # Факт-вывод («с 12 лет») в тексте правил так не написан — у него quote_re: на чём он держится.
    basis = evaluate.facts(" … ".join(x["text"] for x in real),
                           [{**f, "re": f.get("quote_re", f["re"])} for f in q["facts"]])
    out["backed"] = [b["ok"] if a["ok"] and not f.get("absent") and not out["unknown"] else None
                     for f, a, b in zip(q["facts"], out["facts"], basis)]     # от «не знаю» цитат не ждём
    return out


def _spans(questions: list[dict], whole: bool = False) -> list[list[tuple[int, int]]]:
    text = config.DOC_PATH.read_text(encoding="utf-8")
    units = corpus.structure(text)
    return [[_span(units, text, s, whole) for s in q["sources"]] for q in questions]


def _saved() -> dict | None:
    """Идущий или последний прогон — из памяти или с диска."""
    with _lock:
        rep = copy.deepcopy(_live)
    if rep is None and REPORT.exists():
        rep = json.loads(REPORT.read_text(encoding="utf-8"))
    return rep


def pending(questions: list[dict]) -> list[tuple[str, str]]:
    """Что задать модели: ответы, которых нет, и все ответы с цитатами — настройки второго
    этапа двигаются ползунками, поэтому они перепрогоняются, а ответы дня 23 — нет."""
    answers = (_saved() or {"answers": {}})["answers"]
    return [(q["question"], mode) for q in questions for mode in MODES
            if mode == "cite" or "answer" not in answers.get(q["question"], {}).get(mode, {})]


def run() -> None:
    """Прогон: недостающие ответы и режим с цитатами. Ответы видны по мере прихода,
    а на диск прогон пишется целиком в конце — остановленный не затирает прежний.
    Ответы режимов дня 22 остаются в файле как были."""
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
        rep = _saved() or {"answers": {}}
        now = datetime.now().isoformat(timespec="seconds")
        for question, mode in asks:
            rep["answers"].setdefault(question, {}).pop(mode, None)
            rep.setdefault("runs", {})[mode] = now
        rep.update(model=config.LLM_MODEL, strategy=STRATEGY, started=now, finished=None)
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
    rep["runs"] = {m: at for m, at in rep.get("runs", {}).items() if m in MODES}
    questions = control()
    for q, spans, points in zip(questions, _spans(questions), _spans(questions, whole=True)):
        got = answers.get(q["question"], {})
        q["answers"] = {m: _judge(q, a, spans, points) if "answer" in a else a for m, a in got.items() if m in MODES}
    return {**rep, "questions": questions, "summary": summary(questions)}


def _mean(values) -> int:
    values = list(values)
    return round(statistics.mean(values)) if values else 0


def correct(a: dict | None) -> bool:
    """Ответ верный — в нём все ключевые факты из ожидания."""
    return bool(a and a.get("facts")) and all(f["ok"] for f in a["facts"])


def summary(questions: list[dict]) -> dict:
    """Итог по режимам: верные ответы, найденные факты, названные и найденные пункты, цена,
    у ответов с цитатами — источники, цитаты и подтверждены ли ими факты ответа.
    В цену входит переписывание вопроса (даже взятое из кэша — оно оплачено),
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
            "found": sum(sum(a["found"]) for a in done),
            "cut": sum(sum(a["cut"]) for a in done),
            "sources_total": sum(len(q["sources"]) for q in questions),
            "chunks": round(statistics.mean(len(a["hits"]) for a in done), 1) if done else 0,
            "unknown": sum(unknown(a) for a in done),
            "skipped": sum(bool(a.get("skipped")) for a in done),        # из них — без вызова модели
            # Цена ответа: вход (вопрос и фрагменты) и выход, куда входят размышления модели.
            "tokens_in": _mean(u.get("prompt_tokens", 0) for u in usage),
            "tokens_out": _mean(u.get("completion_tokens", 0) for u in usage),
            "tokens_think": _mean((u.get("completion_tokens_details") or {}).get("reasoning_tokens", 0) for u in usage),
            "tokens_rewrite": _mean(extra),
            "tokens_total": _mean(u.get("total_tokens", 0) + x for u, x in zip(usage, extra)),
            "tokens": sum(u.get("total_tokens", 0) + x for u, x in zip(usage, extra)),
            "seconds": round(statistics.mean(a["seconds"] + s for a, s in zip(done, stage)), 1) if done else 0,
            "seconds_stage": round(statistics.mean(stage), 1) if done else 0,
        }
        if mode == "cite":
            said = [a for a in done if not unknown(a)]         # «не знаю» источников и цитат не требует
            quotes = [x for a in said for x in a["quotes"]]
            backed = [b for a in said for b in a["backed"] if b is not None]
            out[mode].update(
                said=len(said),
                sourced=sum(any(x["chunk_id"] in {h["chunk_id"] for h in a["hits"]} for x in a["quotes"]) for a in said),
                quoted=sum(bool(a["quotes"]) for a in said),
                quotes=len(quotes),
                verbatim=sum(bool(x["found"]) for x in quotes),
                own=sum(x["own"] for x in quotes),
                backed=sum(backed),
                backed_total=len(backed),
                meaning=sum(all(b for b in a["backed"] if b is not None) and any(b is not None for b in a["backed"])
                            for a in said),
            )
    right = {m: [correct(q["answers"].get(m)) for q in questions] for m in MODES}
    out["paired"] = {"cite": evaluate.paired(right["rerank"], right["cite"])}
    return out
