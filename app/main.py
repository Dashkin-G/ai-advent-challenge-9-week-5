"""HTTP: интерфейс и API. Логика — в pipeline, rag и rerank, здесь только склейка."""
import asyncio
import json
import threading
from contextlib import asynccontextmanager
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, PlainTextResponse, StreamingResponse
from pydantic import BaseModel, Field

from . import chunking, config, corpus, index, pipeline, rag, rerank

STATIC = Path(__file__).resolve().parent.parent / "static"


def _warm() -> None:
    index.model()
    rerank.model()


@asynccontextmanager
async def lifespan(_app: FastAPI):
    # Индекс уже собран — модели поднимаем заранее, чтобы первый вопрос не ждал 20 секунд.
    if index.builds():
        threading.Thread(target=_warm, daemon=True).start()
    yield


app = FastAPI(title="RAG по ПДД", lifespan=lifespan)


class Build(BaseModel):
    download: bool = False


class Search(BaseModel):
    query: str = ""
    question_id: str | None = None


class Ask(BaseModel):
    question: str = ""


class Settings(BaseModel):
    candidates: int = Field(ge=1, le=config.CANDIDATES)
    threshold: float = Field(ge=0, le=1)
    top: int = Field(ge=1, le=10)


@app.get("/")
def page():
    return FileResponse(STATIC / "index.html")


@app.get("/api/state")
def state():
    """Лёгкое состояние для частого опроса: шаги пайплайна, прогоны проверок, ревизии."""
    return {**pipeline.state(), "model_loaded": index.loaded(), "control": rag.state(), "rerank": rerank.state()}


@app.get("/api/overview")
def overview():
    """Всё, что меняется только после прогона: документы, файлы индекса, сравнение."""
    report = pipeline.report()
    return {
        "docs": corpus.summary(),
        "downloaded": corpus.downloaded(),
        "files": index.files(),
        "config": {"model": config.EMBED_MODEL, "dtype": config.EMBED_DTYPE,
                   "size": config.CHUNK_SIZE, "overlap": config.FIXED_OVERLAP},
        "strategies": chunking.STRATEGIES,
        "report": {k: v for k, v in report.items() if k != "questions"} if report else None,
    }


@app.get("/api/questions")
def questions():
    report = pipeline.report()
    return report["questions"] if report else []


@app.post("/api/build")
async def build(body: Build):
    if pipeline.running():
        raise HTTPException(409, "Пайплайн уже идёт")
    # Поток, а не задача цикла: эмбеддинги на процессоре — долгая работа без await.
    threading.Thread(target=pipeline.run, args=(body.download,), daemon=True).start()
    await asyncio.sleep(0.05)
    return pipeline.state()


@app.post("/api/search")
async def search(body: Search):
    if len(index.builds()) < len(chunking.STRATEGIES):
        raise HTTPException(409, "Индекса ещё нет — сначала постройте его")
    if not body.query.strip() and not body.question_id:
        raise HTTPException(400, "Пустой запрос")
    return await asyncio.to_thread(pipeline.lookup, body.query.strip(), body.question_id)


def _ready_for_answers() -> None:
    if rag.STRATEGY not in index.builds():
        raise HTTPException(409, "Индекса ещё нет — сначала постройте его")
    if not config.DASHSCOPE_API_KEY:
        raise HTTPException(409, "Нет ключа модели: впишите DASHSCOPE_API_KEY в .env и перезапустите приложение")


async def _both_modes(question: str):
    """RAG и RAG + фильтр параллельно, каждый в своём потоке (поиск, реранкер и HTTP-клиент
    синхронные); события уходят строками JSON по мере прихода — ответы печатаются на глазах."""
    loop, queue = asyncio.get_running_loop(), asyncio.Queue()

    def work(mode: str) -> None:
        try:
            for event in rag.answer(question, mode):
                loop.call_soon_threadsafe(queue.put_nowait, {"mode": mode, **event})
        except Exception as e:                              # причина — в колонке режима
            loop.call_soon_threadsafe(queue.put_nowait, {"mode": mode, "type": "error", "text": str(e)})
        finally:
            loop.call_soon_threadsafe(queue.put_nowait, None)

    for mode in rag.LIVE:
        threading.Thread(target=work, args=(mode,), daemon=True).start()
    left = len(rag.LIVE)
    while left:
        event = await queue.get()
        if event is None:
            left -= 1
        else:
            yield json.dumps(event, ensure_ascii=False) + "\n"


@app.post("/api/ask")
async def ask(body: Ask):
    """Вопрос агенту в двух режимах RAG — поток событий (NDJSON)."""
    if not body.question.strip():
        raise HTTPException(400, "Пустой вопрос")
    _ready_for_answers()
    return StreamingResponse(_both_modes(body.question.strip()), media_type="application/x-ndjson")


@app.get("/api/control")
def control():
    """Контрольные вопросы, последний прогон с итогом и что задаст следующий прогон."""
    questions = rag.control()
    return {"questions": questions, "run": rag.report(), "model": config.LLM_MODEL,
            "key": bool(config.DASHSCOPE_API_KEY), "pending": [mode for _, mode in rag.pending(questions)]}


@app.post("/api/control/run")
async def control_run():
    if rag.running():
        raise HTTPException(409, "Прогон уже идёт")
    _ready_for_answers()
    threading.Thread(target=rag.run, daemon=True).start()
    await asyncio.sleep(0.05)
    return rag.state()


@app.post("/api/control/stop")
def control_stop():
    rag.stop()
    return rag.state()


@app.get("/api/rerank")
def rerank_view():
    """Вкладка «Фильтр»: настройки агента и все оценки проверки — метрики считает интерфейс."""
    return {"settings": rerank.settings, "defaults": rerank.DEFAULTS, "report": rerank.report(),
            "model": config.RERANK_MODEL, "key": bool(config.DASHSCOPE_API_KEY)}


@app.post("/api/rerank/settings")
def rerank_settings(body: Settings):
    rerank.settings.update(body.model_dump())
    return rerank.settings


@app.post("/api/rerank/run")
async def rerank_run():
    if rerank.running():
        raise HTTPException(409, "Проверка уже идёт")
    _ready_for_answers()                    # вопросы переписывает модель
    threading.Thread(target=rerank.run, daemon=True).start()
    await asyncio.sleep(0.05)
    return rerank.state()


@app.get("/api/doc", response_class=PlainTextResponse)
def doc():
    """Очищенный текст правил — ровно то, что режется на чанки."""
    if not config.DOC_PATH.exists():
        raise HTTPException(404, "Документы ещё не скачаны")
    return config.DOC_PATH.read_text(encoding="utf-8")


if __name__ == "__main__":
    uvicorn.run(app, host=config.HOST, port=config.PORT)
