"""Пайплайн индексации: документы → нарезка → эмбеддинги → индекс → проверка.

Запускается кнопкой в интерфейсе (или `python -m app.pipeline`) и идёт в фоне;
состояние шагов видно через state(). Итог сравнения стратегий пишется в
data/index/report.json и переживает перезапуск приложения.
"""
import json
import time
import traceback

from . import chunking, config, corpus, evaluate, index

STEPS = {"docs": "Документы", "chunks": "Нарезка", "embed": "Эмбеддинги", "index": "Индекс", "eval": "Проверка"}
REPORT = config.INDEX_DIR / "report.json"

revision = 0            # растёт на каждое изменение — интерфейс перерисовывается только тогда
_run = {"running": False, "started": None, "finished": None, "error": None,
        "steps": {key: {"status": "wait", "note": ""} for key in STEPS}}


def _n(n: int, one: str, few: str, many: str) -> str:
    """531 вектор, 532 вектора, 535 векторов."""
    tail = n % 100
    word = many if 10 < tail < 20 else one if tail % 10 == 1 else few if 2 <= tail % 10 <= 4 else many
    return f"{n} {word}"


def _step(key: str, status: str, note: str = "", progress: tuple[int, int] | None = None) -> None:
    global revision
    _run["steps"][key] = {"status": status, "note": note, "progress": progress}
    revision += 1


def state() -> dict:
    return {"revision": revision, **_run}


def running() -> bool:
    return _run["running"]


def run(download: bool = False) -> None:
    """Весь пайплайн. Падение шага останавливает следующие, причина — в карточке шага."""
    global revision
    _run.update(running=True, started=time.time(), finished=None, error=None)
    for key in STEPS:
        _step(key, "wait")
    current = "docs"
    try:
        _step(current, "run", "скачиваю с garant.ru и GitHub…" if download or not corpus.downloaded()
              else "читаю скачанное с диска")
        docs = corpus.prepare(force=download)
        _step(current, "done", f"{docs['pages']} стр. · "
              + _n(docs["questions"], "вопрос-эталон", "вопроса-эталона", "вопросов-эталонов"))

        current = "chunks"
        _step(current, "run")
        text = config.DOC_PATH.read_text(encoding="utf-8")
        units = corpus.structure(text)
        chunks = {s: chunking.split(s, text, units) for s in chunking.STRATEGIES}
        _step(current, "done", " · ".join(f"{chunking.STRATEGIES[s].lower()}: {len(c)}" for s, c in chunks.items()))

        current = "embed"
        if not index.loaded():
            _step(current, "run", "загружаю модель bge-m3…")
            index.model()
        vectors, tokens, cached = {}, {}, {}
        for s, cs in chunks.items():
            label = chunking.STRATEGIES[s].lower()
            def progress(done, total, hits, s=s, label=label):
                cached[s] = hits
                _step("embed", "run", f"{label}: {done} из {total}" + (f", из кэша {hits}" if hits else ""),
                      (done, total))
            vectors[s] = index.embed([c.text for c in cs], progress)
            tokens[s] = index.count_tokens([c.text for c in cs])
        total, hits = sum(len(v) for v in vectors.values()), sum(cached.values())
        dim = next(iter(vectors.values())).shape[1]
        _step(current, "done", f"{_n(total, 'вектор', 'вектора', 'векторов')} × {dim} · "
                               f"посчитано {total - hits}, из кэша {hits}")

        current = "index"
        _step(current, "run")
        for s in chunks:
            index.save(s, chunks[s], vectors[s], tokens[s])
        _step(current, "done", "FAISS × 2 + SQLite")

        current = "eval"
        _step(current, "run", "ищу ответы на вопросы билетов…")
        report = compare(text, units)
        REPORT.write_text(json.dumps(report, ensure_ascii=False), encoding="utf-8")
        _step(current, "done", _n(report["questions_total"], "вопрос", "вопроса", "вопросов") + " × 2 стратегии")
    except Exception as e:                                  # причина — словами, в карточке шага
        traceback.print_exc()
        _run["error"] = f"{STEPS[current]}: {e}"
        _step(current, "error", str(e))
    finally:
        _run.update(running=False, finished=time.time())
        revision += 1


def compare(text: str, units: list[corpus.Unit]) -> dict:
    """Форма нарезки и качество поиска обеих стратегий на эталонных вопросах."""
    questions = json.loads(config.QUESTIONS_PATH.read_text(encoding="utf-8"))
    golden = [[corpus.span(units, text, r) for r in q["refs"]] for q in questions]
    queries = index.embed([q["question"] for q in questions])
    builds = index.builds()
    report = {"questions_total": len(questions), "top": evaluate.TOP, "strategies": {}, "questions": []}
    judged = {}
    for s in chunking.STRATEGIES:
        hits = index.search(s, queries, evaluate.DEPTH)
        judged[s] = [evaluate.judge(h, g) for h, g in zip(hits, golden)]
        report["strategies"][s] = {
            "title": chunking.STRATEGIES[s],
            "build": builds.get(s, {}),
            "shape": evaluate.shape(index.chunk_rows(s), units),
            "search": evaluate.summary(judged[s]),
        }
    a, b = judged.values()          # порядок — как в chunking.STRATEGIES
    report["paired"] = {
        "hit1": evaluate.paired([j["rank"] == 1 for j in a], [j["rank"] == 1 for j in b]),
        f"hit{evaluate.TOP}": evaluate.paired([bool(j["rank"] and j["rank"] <= evaluate.TOP) for j in a],
                                              [bool(j["rank"] and j["rank"] <= evaluate.TOP) for j in b]),
    }
    for i, q in enumerate(questions):
        report["questions"].append({
            "id": q["id"], "question": q["question"], "ticket": q["ticket"], "n": q["n"],
            "refs": q["refs"],
            "rank": {s: judged[s][i]["rank"] for s in judged},
            "coverage": {s: round(judged[s][i]["coverage"], 2) for s in judged},
        })
    return report


def report() -> dict | None:
    return json.loads(REPORT.read_text(encoding="utf-8")) if REPORT.exists() else None


def ref_label(ref: dict) -> str:
    """{part: rules, number: 11.4} → «п. 11.4 ПДД»."""
    number = ref["number"]
    return {
        "rules": f"п. {number} ПДД" + (f", термин «{ref['term']}»" if ref.get("term") else ""),
        "signs": f"знак {number}",
        "marking": f"разметка {number}",
        "admission": f"п. {number} Основных положений",
        "faults": f"п. {number} Перечня неисправностей",
    }[ref["part"]]


def lookup(query: str, question_id: str | None = None) -> dict:
    """Поиск по обоим индексам. Для вопроса из эталона — отметка нужных чанков
    и подсветка того, где в чанке лежит ответ."""
    text = config.DOC_PATH.read_text(encoding="utf-8")
    question, golden = None, []
    if question_id:
        questions = json.loads(config.QUESTIONS_PATH.read_text(encoding="utf-8"))
        question = next((q for q in questions if q["id"] == question_id), None)
        if question:
            units = corpus.structure(text)
            golden = [corpus.span(units, text, r) for r in question["refs"]]
            question = {**question, "golden": [{"label": ref_label(r), "text": text[s:e]}
                                               for r, (s, e) in zip(question["refs"], golden)]}
            query = question["question"]
    started = time.perf_counter()
    vector = index.embed([query])
    results = {}
    for s in chunking.STRATEGIES:
        hits = index.search(s, vector, evaluate.TOP)[0]
        for rank, h in enumerate(hits, 1):
            h["rank"] = rank
            # Чанк начат или оборван посреди абзаца — в интерфейсе это видно многоточием.
            h["cut_start"] = h["start"] > 0 and text[h["start"] - 1] != "\n"
            h["cut_end"] = h["end"] < len(text) and text[h["end"]] != "\n"
            if golden:
                h["relevant"] = evaluate.relevant((h["start"], h["end"]), golden)
                h["marks"] = evaluate.marks((h["start"], h["end"]), golden)
        results[s] = hits
    return {"query": query, "question": question, "results": results,
            "ms": round(1000 * (time.perf_counter() - started))}


if __name__ == "__main__":
    run()
    print(json.dumps(state(), ensure_ascii=False, indent=1))
    r = report()
    if r:
        for s, v in r["strategies"].items():
            print(s, json.dumps({k: v for k, v in v["shape"].items() if k != "sizes"}, ensure_ascii=False))
            print(s, json.dumps(v["search"], ensure_ascii=False))
