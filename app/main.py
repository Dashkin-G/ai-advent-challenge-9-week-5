"""HTTP: интерфейс и API. Логика — в pipeline, rag, rerank и chat, здесь только склейка."""
import asyncio
import json
import threading
from collections.abc import Iterator
from contextlib import asynccontextmanager
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, PlainTextResponse, StreamingResponse
from pydantic import BaseModel, Field

from . import chat, chunking, config, corpus, index, pipeline, rag, rerank

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


class Say(BaseModel):
    text: str = ""


class Memory(BaseModel):
    goal: str = ""
    clarified: list[str] = []
    constraints: list[str] = []


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
    return {**pipeline.state(), "model_loaded": index.loaded(), "control": rag.state(), "rerank": rerank.state(),
            "chat": chat.state()}


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


def _pump(events: Iterator[dict]):
    """Ответ идёт в своём потоке — поиск, реранкер и HTTP-клиент синхронные — и доходит до конца,
    даже если страницу закрыли: ход сохранится в истории. События уходят строками JSON по мере прихода."""
    loop, queue = asyncio.get_running_loop(), asyncio.Queue()

    def work() -> None:
        try:
            for event in events:
                loop.call_soon_threadsafe(queue.put_nowait, event)
        except Exception as e:                              # причина — в ответе на экране
            loop.call_soon_threadsafe(queue.put_nowait, {"type": "error", "text": str(e)})
        finally:
            loop.call_soon_threadsafe(queue.put_nowait, None)

    threading.Thread(target=work, daemon=True).start()

    async def lines():
        while (event := await queue.get()) is not None:
            yield json.dumps(event, ensure_ascii=False) + "\n"
    return lines()


@app.get("/api/chats")
def chats():
    """Список диалогов и сценарии проверки с итогом последнего прогона."""
    return {"chats": chat.chats(), "scenarios": chat.overview(), "window": config.WINDOW,
            "key": bool(config.DASHSCOPE_API_KEY)}


@app.post("/api/chats")
def chat_create():
    return {"id": chat.create()}


@app.get("/api/chats/{chat_id}")
def chat_view(chat_id: int):
    got = chat.view(chat_id)
    if got is None:
        raise HTTPException(404, "Такого диалога нет")
    return got


@app.delete("/api/chats/{chat_id}")
def chat_delete(chat_id: int):
    if not chat.claim(chat_id):
        raise HTTPException(409, "В диалоге идёт ответ — дождитесь его")
    chat.delete(chat_id)
    chat.release(chat_id)
    return {"ok": True}


@app.patch("/api/chats/{chat_id}")
def chat_memory(chat_id: int, body: Memory):
    """Водитель поправил память задачи."""
    if chat.load(chat_id) is None:
        raise HTTPException(404, "Такого диалога нет")
    return chat.set_memory(chat_id, body.model_dump())


@app.post("/api/chats/{chat_id}/ask")
async def chat_ask(chat_id: int, body: Say):
    """Сообщение в диалог — поток событий (NDJSON): память, поиск, реранкер, промпт, модель, ход целиком."""
    text = body.text.strip()
    if not text:
        raise HTTPException(400, "Пустое сообщение")
    _ready_for_answers()
    if chat.load(chat_id) is None:
        raise HTTPException(404, "Такого диалога нет")
    if not chat.claim(chat_id):
        raise HTTPException(409, "В этом диалоге уже идёт ответ")
    return StreamingResponse(_pump(chat.ask(chat_id, text)), media_type="application/x-ndjson")


@app.post("/api/scenarios/run")
async def scenarios_run():
    if chat.running():
        raise HTTPException(409, "Прогон уже идёт")
    _ready_for_answers()
    threading.Thread(target=chat.run, daemon=True).start()
    await asyncio.sleep(0.05)
    return chat.state()


@app.post("/api/scenarios/stop")
def scenarios_stop():
    chat.stop()
    return chat.state()


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
