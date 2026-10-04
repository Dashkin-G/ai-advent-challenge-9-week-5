"""Чат: история диалога, память задачи и ответ с источниками на каждое сообщение.

Сообщение → модель обновляет память задачи и переписывает реплику в самостоятельный вопрос
терминами правил («а младшего спереди?» → «перевозка ребёнка 5 лет на переднем сиденье») →
второй этап поиска по этому вопросу (rag.second_stage) → ответ с цитатами, как в rag.py.
Модель видит память и последние обмены — окно (config.WINDOW). Что было раньше, она знает
только из памяти, поэтому промпт не растёт с длиной диалога.

Память задачи — цель диалога, что водитель уточнил о себе и своей ситуации, ограничения и
термины, о которых договорились. Её обновляет тот же дешёвый вызов, что переписывает реплику, —
до поиска: искать надо уже с учётом сказанного. Водитель видит память и может удалить лишнее.

Диалоги хранятся в SQLite (data/chat.db). Два длинных сценария (scenarios.json) прогоняются
как обычные диалоги. Проверка хода — цель на месте, сказанное водителем в памяти, источники,
нужный пункт, факты ответа — считается при чтении: поправили ожидание — вердикты пересчитаются
без модели. Ходы, ответ на которые зависит от сказанного за окном, повторяются без памяти —
видно, что она даёт.
"""
import json
import re
import sqlite3
import statistics
import threading
import time
import traceback
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from datetime import datetime

from . import config, evaluate, llm, pipeline, rag

EMPTY = {"goal": "", "clarified": [], "constraints": []}

MEMORY = (
    "Ты ведёшь память задачи в диалоге водителя со справочником по Правилам дорожного движения РФ. "
    "Дано: память, последние реплики и новое сообщение водителя. Верни JSON: "
    '{"goal": "…", "clarified": ["…"], "constraints": ["…"], "query": "…"}. '
    "goal — задача всего диалога одной фразой, а не его первый вопрос: в чём водитель хочет разобраться; "
    "без подробностей ситуации; не меняй, пока он сам не сменит тему. "
    "clarified — что водитель сообщил о себе и своей ситуации: транспорт, пассажиры, стаж, маршрут, обстоятельства. "
    "constraints — условия и термины, которые водитель задал для разговора: «говорим только о грузовике», "
    "«парковка — это стоянка у дома». "
    "Записи короткие; старые переноси слово в слово. Водитель изменил факт — замени запись, а не добавляй "
    "вторую. Ответы справочника и свои выводы в память не пиши. "
    "query — то, о чём спрошено в новом сообщении, одним коротким самостоятельным вопросом терминами Правил "
    "для поиска по их тексту: подставь, о чём речь, и обстоятельства из памяти, от которых зависит ответ. "
    "На вопрос не отвечай."
)
DIALOG = (
    "Это диалог: перед вопросом могут быть прежние реплики и память задачи — цель водителя, что он уточнил "
    "о себе, ограничения и термины. Отвечай про его случай: применяй правила к тому, что он рассказал, и "
    "называй это в ответе. Факты и цитаты — только из фрагментов правил последнего сообщения; прежние "
    "ответы — не источник."
)


# --- память задачи -------------------------------------------------------------------

def _key(s: str) -> str:
    """Запись памяти для сравнения: регистр, «ё» и знаки препинания не важны."""
    return re.sub(r"\W+", " ", s.lower().replace("ё", "е")).strip()


def _clean(m: dict) -> dict:
    def items(xs) -> list[str]:
        out = []
        for x in xs if isinstance(xs, list) else []:
            x = " ".join(str(x).split())
            if x and _key(x) not in map(_key, out):
                out.append(x)
        return out
    return {"goal": " ".join(str(m.get("goal") or "").split()),
            "clarified": items(m.get("clarified")), "constraints": items(m.get("constraints"))}


def _merge(old: dict, got: dict) -> dict:
    """Память из ответа модели. Поля нет или оно не того типа, цель пустая — остаётся прежнее:
    сбой разметки не должен стирать цель или уточнения."""
    goal = got["goal"] if isinstance(got.get("goal"), str) and got["goal"].strip() else old["goal"]
    lists = {f: got[f] if isinstance(got.get(f), list) else old[f] for f in ("clarified", "constraints")}
    return _clean({"goal": goal, **lists})


def changes(before: dict, after: dict) -> dict:
    """Что изменилось в памяти за ход: новые и ушедшие записи, сменилась ли цель."""
    out = {"goal": bool(after["goal"]) and _key(before["goal"]) != _key(after["goal"])}
    for f in ("clarified", "constraints"):
        old, new = set(map(_key, before[f])), set(map(_key, after[f]))
        out[f] = {"added": [x for x in after[f] if _key(x) not in old],
                  "removed": [x for x in before[f] if _key(x) not in new]}
    return out


def said(t: dict) -> str:
    """Ответ хода для окна: без сносок [1] — у цитат нового ответа свои номера; с просьбой уточнить."""
    text = re.sub(r"\s*\[\d+(?:\s*,\s*\d+)*\]", "", t["answer"])
    return " ".join(filter(None, [text, t.get("clarify")]))


def understand(memory: dict, window: list[dict], message: str) -> dict:
    """Один вызов модели до поиска: память после сообщения и самостоятельный вопрос для поиска.
    Думать модели дают немного: задача простая, а без ограничения она размышляет по тысяче токенов.
    JSON не разобрался — память прежняя, а искать придётся по самому сообщению."""
    lines = ["Память задачи:", json.dumps(memory, ensure_ascii=False)]
    if window:
        lines += ["", "Последние реплики:"]
        for t in window:
            lines += [f"Водитель: {t['question']}", f"Справочник: {said(t)}"]
    lines += ["", f"Новое сообщение водителя: {message}"]
    prompt = [{"role": "system", "content": MEMORY}, {"role": "user", "content": "\n".join(lines)}]
    started, parts, usage = time.perf_counter(), [], {}
    for kind, value in llm.stream(prompt, budget=config.MEMORY_BUDGET, as_json=True):
        if kind == "text":
            parts.append(value)
        elif kind == "usage":
            usage = value
    raw = "".join(parts)
    try:
        got = json.loads(raw[raw.index("{"):raw.rindex("}") + 1])
    except ValueError:
        got = None
    parsed = isinstance(got, dict)
    query = " ".join(str(got.get("query") or "").split()) if parsed else ""
    return {"memory": _merge(memory, got) if parsed else memory, "query": query or message, "parsed": parsed,
            "tokens": usage.get("total_tokens", 0), "seconds": round(time.perf_counter() - started, 1)}


# --- ответ ----------------------------------------------------------------------------

def _memory_text(m: dict) -> str:
    return "\n".join([f"Цель: {m['goal'] or '—'}", f"Уточнил: {'; '.join(m['clarified']) or '—'}",
                      f"Ограничения и термины: {'; '.join(m['constraints']) or '—'}"])


def messages(message: str, query: str, hits: list[dict], window: list[dict], memory: dict | None) -> list[dict]:
    """Промпт ответа: инструкция с цитатами (rag.CITE) и про диалог; окно — прежние вопросы и ответы;
    в последнем сообщении — память задачи, фрагменты правил и вопрос вместе с его смыслом в диалоге."""
    system, last = rag.messages(f"{message}\nС учётом диалога: {query}", hits, "cite")
    history = []
    for t in window:
        history += [{"role": "user", "content": t["question"]}, {"role": "assistant", "content": said(t)}]
    head = f"Память задачи:\n{_memory_text(memory)}\n\n" if memory else ""
    return [{"role": "system", "content": f"{system['content']} {DIALOG}"}, *history,
            {"role": "user", "content": head + last["content"]}]


def reply(message: str, memory: dict | None, window: list[dict]) -> Iterator[dict]:
    """Ответ на сообщение событиями: understand → candidates → sources → prompt → think… → text….
    memory=None — без памяти (сравнение на сценариях): модель видит только окно.
    Порог не прошёл ни один чанк — «не знаю» без модели, как у агента с цитатами.
    Возвращает ход — то, что хранится в истории."""
    got = understand(memory or EMPTY, window, message)
    turn = {"query": got["query"], "understand": {k: got[k] for k in ("tokens", "seconds", "parsed")}}
    if memory is not None:
        turn.update(memory=got["memory"], changes=changes(memory, got["memory"]))
    yield {"type": "understand", **turn}
    for event in rag.second_stage(message, got["query"]):
        if event["type"] == "sources":
            turn.update(hits=event["hits"], dropped=event["dropped"], settings=event["settings"],
                        stage_seconds=event["stage"])
        yield event
    if not turn["hits"]:
        turn.update(answer=rag.DONT_KNOW, clarify=rag.CLARIFY, quotes=[], usage={}, seconds=0, skipped=True)
        return turn
    prompt = messages(message, got["query"], turn["hits"], window, turn.get("memory"))
    yield {"type": "prompt", "messages": prompt}
    answer, usage, seconds = yield from rag.ask_model(prompt, True)
    turn.update(answer, prompt=prompt, usage=usage, seconds=seconds)     # промпт — чтобы было видно окно и память
    return turn


def _verified(t: dict) -> dict:
    """Ход для экрана: цитаты сверены с чанками, ушедшими модели, «не знаю» отмечено."""
    return {**t, **rag.verify(t, t["hits"]), "unknown": rag.unknown(t)}


# --- история диалогов -------------------------------------------------------------------

SCHEMA = """
CREATE TABLE IF NOT EXISTS chats (
    id       INTEGER PRIMARY KEY AUTOINCREMENT,
    scenario TEXT,                      -- id сценария, если диалог — его прогон
    created  TEXT NOT NULL,
    memory   TEXT NOT NULL              -- JSON: память задачи сейчас
);
CREATE TABLE IF NOT EXISTS turns (
    chat_id  INTEGER NOT NULL,
    n        INTEGER NOT NULL,          -- номер обмена, с 1
    question TEXT NOT NULL,
    data     TEXT NOT NULL,             -- JSON хода: память после него, запрос, чанки, ответ, цитаты, расход
    created  TEXT NOT NULL,
    PRIMARY KEY (chat_id, n)
);
"""
revision = 0            # растёт на каждое изменение истории — интерфейс перечитывает список диалогов
_lock = threading.Lock()
_busy: set[int] = set()  # диалоги, в которых идёт ответ: два ответа сразу перепутали бы историю


def _db() -> sqlite3.Connection:
    config.CHAT_DB.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(config.CHAT_DB)
    con.executescript(SCHEMA)
    return con


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _changed() -> None:
    global revision
    revision += 1


def create(scenario: str | None = None) -> int:
    with _lock, _db() as con:
        chat_id = con.execute("INSERT INTO chats (scenario, created, memory) VALUES (?, ?, ?)",
                              (scenario, _now(), json.dumps(EMPTY))).lastrowid
    _changed()
    return chat_id


def chats() -> list[dict]:
    """Диалоги для списка, свежие сверху: память, число обменов, первое сообщение."""
    with _db() as con:
        rows = con.execute("""
            SELECT c.id, c.scenario, c.created, c.memory, COUNT(t.n), MAX(t.created),
                   (SELECT question FROM turns WHERE chat_id = c.id AND n = 1)
            FROM chats c LEFT JOIN turns t ON t.chat_id = c.id GROUP BY c.id""").fetchall()
    out = [{"id": i, "scenario": s, "created": c, "memory": json.loads(m), "turns": n, "updated": u or c, "first": f}
           for i, s, c, m, n, u, f in rows]
    return sorted(out, key=lambda x: (x["updated"], x["id"]), reverse=True)


def load(chat_id: int) -> dict | None:
    with _db() as con:
        row = con.execute("SELECT scenario, created, memory FROM chats WHERE id = ?", (chat_id,)).fetchone()
        turns = con.execute("SELECT n, question, created, data FROM turns WHERE chat_id = ? ORDER BY n",
                            (chat_id,)).fetchall()
    if row is None:
        return None
    return {"id": chat_id, "scenario": row[0], "created": row[1], "memory": json.loads(row[2]),
            "turns": [{"n": n, "question": q, "created": c, **json.loads(d)} for n, q, c, d in turns]}


def delete(chat_id: int) -> None:
    with _lock, _db() as con:
        con.execute("DELETE FROM turns WHERE chat_id = ?", (chat_id,))
        con.execute("DELETE FROM chats WHERE id = ?", (chat_id,))
    _changed()


def set_memory(chat_id: int, memory: dict) -> dict:
    """Водитель поправил память — например, удалил лишнюю запись. Следующий ход начнётся с неё."""
    memory = _clean(memory)
    with _lock, _db() as con:
        con.execute("UPDATE chats SET memory = ? WHERE id = ?", (json.dumps(memory, ensure_ascii=False), chat_id))
    _changed()
    return memory


def _save(chat_id: int, turn: dict) -> None:
    data = {k: v for k, v in turn.items() if k not in ("n", "question", "created")}
    with _lock, _db() as con:
        con.execute("INSERT OR REPLACE INTO turns VALUES (?, ?, ?, ?, ?)",
                    (chat_id, turn["n"], turn["question"], json.dumps(data, ensure_ascii=False), turn["created"]))
        if "memory" in turn:
            con.execute("UPDATE chats SET memory = ? WHERE id = ?",
                        (json.dumps(turn["memory"], ensure_ascii=False), chat_id))
    _changed()


def _truncate(chat_id: int, n: int) -> None:
    """Убрать ходы с n-го; память — как после последнего оставшегося хода."""
    with _lock, _db() as con:
        con.execute("DELETE FROM turns WHERE chat_id = ? AND n >= ?", (chat_id, n))
        last = con.execute("SELECT data FROM turns WHERE chat_id = ? ORDER BY n DESC LIMIT 1", (chat_id,)).fetchone()
        memory = json.loads(last[0]).get("memory", EMPTY) if last else EMPTY
        con.execute("UPDATE chats SET memory = ? WHERE id = ?", (json.dumps(memory, ensure_ascii=False), chat_id))
    _changed()


def claim(chat_id: int) -> bool:
    """Занять диалог под ответ; освобождает его ask(), когда ответ кончился."""
    with _lock:
        if chat_id in _busy:
            return False
        _busy.add(chat_id)
        return True


def release(chat_id: int) -> None:
    with _lock:
        _busy.discard(chat_id)


def ask(chat_id: int, message: str) -> Iterator[dict]:
    """Сообщение в диалог (занятый заранее, claim): ответ событиями, в конце — ход целиком (done),
    уже сохранённый в истории. Оборвался ответ — ход не сохраняется: полуответов в истории нет.
    Возвращает сохранённый ход."""
    try:
        chat = load(chat_id)
        n = len(chat["turns"]) + 1
        turn = yield from reply(message, chat["memory"], chat["turns"][-config.WINDOW:])
        turn = {"n": n, "question": message, "created": _now(), **turn}
        _save(chat_id, turn)
        yield {"type": "done", "turn": _verified(turn)}
        return turn
    finally:
        release(chat_id)


# --- сценарии проверки -----------------------------------------------------------------

def scenarios() -> list[dict]:
    """Сценарии; у источников — подпись вида «п. 22.9 ПДД»."""
    items = json.loads(config.SCENARIOS_PATH.read_text(encoding="utf-8"))
    for sc in items:
        for step in sc["steps"]:
            for s in step.get("sources", []):
                s["label"] = pipeline.ref_label(s)
    return items


def _beyond(step: dict, n: int) -> list[int]:
    """От каких прежних шагов зависит ответ на n-м ходу, если их уже нет в окне модели."""
    return [k for k in step.get("depends", []) if n - k > config.WINDOW]


def _scenario_chat(sid: str) -> dict | None:
    with _db() as con:
        row = con.execute("SELECT MAX(id) FROM chats WHERE scenario = ?", (sid,)).fetchone()
    return load(row[0]) if row[0] else None


def _matched(sc: dict, chat: dict | None) -> list[dict]:
    """Ходы диалога, совпавшие с шагами сценария по тексту, — до первого расхождения."""
    out = []
    for step, t in zip(sc["steps"], chat["turns"] if chat else []):
        if t["question"] != step["say"]:
            break
        out.append(t)
    return out


def todo(sc: dict, chat: dict | None) -> int:
    """Сколько вызовов (ходов и повторов без памяти) осталось прогону сценария; 0 — прогнан целиком.
    Изменённый шаг и всё после него делаются заново."""
    done = _matched(sc, chat)
    left = 0
    for i, step in enumerate(sc["steps"], 1):
        t = done[i - 1] if i <= len(done) else None
        left += (t is None) + bool(_beyond(step, i) and (t is None or "ablation" not in t))
    return left


_run = {"running": False, "done": 0, "total": 0, "started": None, "finished": None, "error": None,
        "stopping": False, "stopped": False}
_stop = threading.Event()


class Stopped(Exception):
    """Прогон остановлен кнопкой."""


def state() -> dict:
    return {"revision": revision, **_run}


def running() -> bool:
    return _run["running"]


def stop() -> None:
    """Остановить прогон: идущие ответы обрываются; сделанные ходы остаются, следующий прогон продолжит."""
    if _run["running"]:
        _stop.set()
        _run["stopping"] = True
        _changed()


def _drain(events: Iterator[dict]):
    """Пройти события ответа до конца и вернуть то, что вернул генератор. Остановили прогон —
    оборвать: closing закрывает генератор, а с ним и поток ответа модели."""
    with closing(events):
        while True:
            if _stop.is_set():
                raise Stopped
            try:
                next(events)
            except StopIteration as done:
                return done.value


def _retry(make):
    """Сбой сети или перегрузка модели — до двух повторов, как у прогона контрольных вопросов."""
    for attempt in range(3):
        try:
            return _drain(make())
        except llm.LLMError as e:
            if not e.transient or attempt == 2:
                raise
            if _stop.wait(5 * (attempt + 1)):
                raise Stopped


def _ask_step(chat_id: int, message: str) -> Iterator[dict]:
    if not claim(chat_id):
        raise RuntimeError("в диалоге сценария идёт ответ — дождитесь его")
    return ask(chat_id, message)


def _play(sc: dict, fresh: bool) -> None:
    """Сценарий — обычный диалог, ход за ходом. На ходах, ответ на которые зависит от сказанного
    за окном, тот же ход повторяется без памяти — с тем же окном, что видел ход с памятью."""
    chat = None if fresh else _scenario_chat(sc["id"])
    chat_id = chat["id"] if chat else create(sc["id"])
    turns = _matched(sc, chat)
    if chat and len(turns) < len(chat["turns"]):
        _truncate(chat_id, len(turns) + 1)
    for i, step in enumerate(sc["steps"], 1):
        if i > len(turns):
            turns.append(_retry(lambda: _ask_step(chat_id, step["say"])))
            _progress()
        t = turns[i - 1]
        if _beyond(step, i) and "ablation" not in t:
            window = turns[max(0, i - 1 - config.WINDOW):i - 1]
            t["ablation"] = _retry(lambda: reply(step["say"], None, window))
            _save(chat_id, t)
            _progress()


def _progress() -> None:
    _run["done"] += 1
    _changed()


def run() -> None:
    """Прогон сценариев — параллельно, каждый ход за ходом. Недоделанный прогон продолжается с места
    остановки; если всё уже прогнано — сценарии прогоняются заново, прежние диалоги удаляются."""
    _stop.clear()
    _run.update(running=True, done=0, total=0, started=time.time(), finished=None, error=None,
                stopping=False, stopped=False)
    _changed()
    try:
        items = scenarios()
        for sc in items:                                  # ошибка в ожиданиях — стоп до трат
            rag._spans(sc["steps"])
            [re.compile(f["re"]) for step in sc["steps"] for k in ("facts", "remember", "forget") for f in step.get(k, [])]
        left = {sc["id"]: todo(sc, _scenario_chat(sc["id"])) for sc in items}
        fresh = not any(left.values())
        if fresh:
            for sc in items:
                while (old := _scenario_chat(sc["id"])) is not None:
                    delete(old["id"])
            left = {sc["id"]: todo(sc, None) for sc in items}
        _run["total"] = sum(left.values())
        plays = [sc for sc in items if left[sc["id"]]]
        with ThreadPoolExecutor(len(plays) or 1) as pool:
            jobs = [pool.submit(_play, sc, fresh) for sc in plays]
            errors = []
            for job in jobs:
                try:
                    job.result()
                except Stopped:
                    pass
                except Exception as e:                    # один сценарий упал — второй останавливаем
                    _stop.set()
                    errors.append(e)
        if errors:
            raise errors[0]
        _run["stopped"] = _stop.is_set()
    except Exception as e:
        traceback.print_exc()
        _run["error"] = str(e)
    finally:
        _run.update(running=False, stopping=False, finished=time.time())
        _changed()


# --- проверка ---------------------------------------------------------------------------

def _facts_text(m: dict) -> str:
    return " ; ".join(m["clarified"] + m["constraints"])


def check(sc: dict, chat: dict | None) -> list[dict]:
    """Ходы сценария против ожиданий. Память: цель на месте, сказанное водителем — в ней, отменённое —
    убрано. Ответ — как у контрольных вопросов (rag._judge): факты, источники, нужный пункт, цитаты;
    на ходах, ответ на которые зависит от сказанного за окном, — то же без памяти."""
    turns = _matched(sc, chat)
    spans, points = rag._spans(sc["steps"]), rag._spans(sc["steps"], whole=True)
    kept, gone, rows = {}, {}, []
    for i, step in enumerate(sc["steps"], 1):
        for f in step.get("forget", []):
            kept.pop(f["text"], None)
            gone[f["text"]] = f
        kept.update((f["text"], f) for f in step.get("remember", []))
        row = {"n": i, "say": step["say"], "why": step.get("why", ""), "out": bool(step.get("out")),
               "expect": step.get("expect", ""), "sources": [s["label"] for s in step.get("sources", [])],
               "facts": [{"text": f["text"], "absent": bool(f.get("absent"))} for f in step.get("facts", [])],
               "beyond": _beyond(step, i), "turn": None}
        t = turns[i - 1] if i <= len(turns) else None
        if t is not None:
            q = {"facts": step.get("facts", []), "sources": step.get("sources", [])}
            text = _facts_text(t["memory"])
            row["turn"] = rag._judge(q, {k: v for k, v in t.items() if k != "ablation"}, spans[i - 1], points[i - 1])
            row["memory"] = {
                "goal": bool(re.search(sc["goal"]["re"], t["memory"]["goal"], re.IGNORECASE)),
                "kept": [{"text": f["text"], "ok": bool(re.search(f["re"], text, re.IGNORECASE))} for f in kept.values()],
                "gone": [{"text": f["text"], "ok": not re.search(f["re"], text, re.IGNORECASE)} for f in gone.values()],
            }
            if "ablation" in t:
                row["ablation"] = rag._judge(q, t["ablation"], spans[i - 1], points[i - 1])
        rows.append(row)
    return rows


def _sourced(a: dict) -> bool:
    """Ответ с источником: хотя бы одна цитата слово в слово из чанка, ушедшего модели."""
    return any(x["found"] for x in a["quotes"])


def _tokens(t: dict) -> int:
    return t["understand"]["tokens"] + (t.get("usage") or {}).get("total_tokens", 0)


def summary(rows: list[dict]) -> dict:
    """Итог сценария: держится ли память, есть ли источники, верны ли ответы и что дала память
    на ходах, где сказанное водителем уже ушло из окна. Цена — оба вызова модели на ход."""
    done = [r for r in rows if r["turn"]]
    said_ = [r for r in done if not r["turn"]["unknown"]]
    judged = [r for r in done if r["turn"]["facts"]]
    beyond = [r for r in done if r["beyond"] and "ablation" in r]
    right = rag.correct
    turns = [r["turn"] for r in done]
    return {
        "steps": len(rows), "done": len(done),
        "goal": sum(r["memory"]["goal"] for r in done),
        "memory": sum(all(f["ok"] for f in r["memory"]["kept"] + r["memory"]["gone"]) for r in done),
        "said": len(said_), "sourced": sum(_sourced(r["turn"]) for r in said_),
        "out": sum(r["out"] for r in done), "out_ok": sum(r["out"] and r["turn"]["unknown"] for r in done),
        "judged": len(judged), "correct": sum(right(r["turn"]) for r in judged),
        "sources": sum(len(r["turn"]["found"]) for r in done), "found": sum(sum(r["turn"]["found"]) for r in done),
        "cited": sum(sum(r["turn"]["cited"]) for r in done),
        "beyond": len(beyond), "beyond_with": sum(right(r["turn"]) for r in beyond),
        "beyond_without": sum(right(r["ablation"]) for r in beyond),
        "paired": evaluate.paired([right(r["turn"]) for r in beyond], [right(r["ablation"]) for r in beyond]),
        "tokens": round(statistics.mean(map(_tokens, turns))) if turns else 0,
        "tokens_all": sum(map(_tokens, turns)) + sum(_tokens(r["ablation"]) for r in beyond),
        "prompt": [(t.get("usage") or {}).get("prompt_tokens", 0) for t in turns],
        "seconds": round(statistics.mean(t["understand"]["seconds"] + t["stage_seconds"] + t["seconds"]
                                         for t in turns), 1) if turns else 0,
    }


def overview() -> list[dict]:
    """Сценарии для списка диалогов: диалог последнего прогона, сколько осталось и итог."""
    out = []
    for sc in scenarios():
        chat = _scenario_chat(sc["id"])
        out.append({"id": sc["id"], "title": sc["title"], "chat_id": chat and chat["id"], "steps": len(sc["steps"]),
                    "todo": todo(sc, chat), "todo_fresh": todo(sc, None), "summary": summary(check(sc, chat))})
    return out


def view(chat_id: int) -> dict | None:
    """Диалог для экрана: ходы со сверенными цитатами; у прогона сценария — проверка каждого хода и итог."""
    chat = load(chat_id)
    if chat is None:
        return None
    sc = next((s for s in scenarios() if s["id"] == chat["scenario"]), None) if chat["scenario"] else None
    if sc is None:
        chat["turns"] = [_verified(t) for t in chat["turns"]]
        return chat
    rows = check(sc, chat)
    judged = [r["turn"] for r in rows if r["turn"]]
    chat["turns"] = judged + [_verified(t) for t in chat["turns"][len(judged):]]
    chat["title"] = sc["title"]
    chat["check"] = [{k: v for k, v in r.items() if k != "turn"} for r in rows]
    chat["summary"] = summary(rows)
    return chat
