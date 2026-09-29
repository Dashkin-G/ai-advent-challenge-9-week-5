"""HTTP: интерфейс и API. Логика — в pipeline, здесь только склейка."""
import asyncio
import threading
from contextlib import asynccontextmanager
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, PlainTextResponse
from pydantic import BaseModel

from . import chunking, config, corpus, index, pipeline

STATIC = Path(__file__).resolve().parent.parent / "static"


@asynccontextmanager
async def lifespan(_app: FastAPI):
    # Индекс уже собран — модель поднимаем заранее, чтобы первый поиск не ждал 15 секунд.
    if index.builds():
        threading.Thread(target=index.model, daemon=True).start()
    yield


app = FastAPI(title="Индекс документов", lifespan=lifespan)


class Build(BaseModel):
    download: bool = False


class Search(BaseModel):
    query: str = ""
    question_id: str | None = None


@app.get("/")
def page():
    return FileResponse(STATIC / "index.html")


@app.get("/api/state")
def state():
    """Лёгкое состояние для частого опроса: шаги пайплайна и номер ревизии."""
    return {**pipeline.state(), "model_loaded": index.loaded()}


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


@app.get("/api/doc", response_class=PlainTextResponse)
def doc():
    """Очищенный текст правил — ровно то, что режется на чанки."""
    if not config.DOC_PATH.exists():
        raise HTTPException(404, "Документы ещё не скачаны")
    return config.DOC_PATH.read_text(encoding="utf-8")


if __name__ == "__main__":
    uvicorn.run(app, host=config.HOST, port=config.PORT)
