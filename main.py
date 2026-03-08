from contextlib import asynccontextmanager

from fastapi import FastAPI

from agent.react_agernt import ReActAgent
from api.routes import agent, rag
from rag.vector_store import VectorStoreService


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.agent = ReActAgent()
    app.state.vector_store = VectorStoreService()
    yield


app = FastAPI(title="AI Agent API", version="1.0.0", lifespan=lifespan)

app.include_router(agent.router)
app.include_router(rag.router)


@app.get("/", tags=["health"])
async def health_check():
    return {"status": "ok", "service": "ai-agent"}