import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from agent.react_agernt import ReActAgent


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.agent = ReActAgent()
    yield


app = FastAPI(title="AI Agent API", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    query: str = Field(..., min_length=1, description="用户输入的问题")


@app.post("/chat")
async def chat(payload: ChatRequest):
    agent = app.state.agent

    async def generate():
        loop = asyncio.get_event_loop()
        queue = asyncio.Queue()

        def run_agent():
            last_content = ""
            for message in agent.execute_stream(payload.query):
                if message.content and message.content.strip():
                    # 每条内容都发出去，用特殊前缀区分
                    loop.call_soon_threadsafe(queue.put_nowait, ("chunk", message.content))
                    last_content = message.content
            # 全部跑完，发出最终结果
            loop.call_soon_threadsafe(queue.put_nowait, ("final", last_content))
            loop.call_soon_threadsafe(queue.put_nowait, None)

        loop.run_in_executor(None, run_agent)

        while True:
            item = await queue.get()
            if item is None:
                break
            kind, content = item
            # 用 \x00 分隔类型和内容，前端按这个解析
            yield f"{kind}\x00{content}\n\x01\n"

    return StreamingResponse(generate(), media_type="text/plain; charset=utf-8")


@app.get("/")
async def health_check():
    return {"status": "ok"}