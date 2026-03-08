from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from agent.react_agernt import ReActAgent
from rag.vector_store import VectorStoreService

app = FastAPI(title="AI Agent API", version="1.0.0")

agent_service = ReActAgent()
vector_store_service = VectorStoreService()


class ChatRequest(BaseModel):
    query: str = Field(..., min_length=1, description="用户输入的问题")


class ChatResponse(BaseModel):
    answer: str


@app.get("/")
async def health_check():
    return {"status": "ok", "service": "ai-agent"}


@app.post("/agent/chat", response_model=ChatResponse)
async def agent_chat(payload: ChatRequest):
    try:
        chunks: list[str] = []
        for message in agent_service.execute_stream(payload.query):
            content = message.content
            if isinstance(content, str):
                chunks.append(content)
            elif isinstance(content, list):
                chunks.extend(str(item) for item in content)
        return ChatResponse(answer="".join(chunks).strip())
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Agent调用失败: {str(e)}") from e


@app.post("/rag/reload")
async def reload_knowledge_base():
    try:
        vector_store_service.load_document()
        return {"message": "知识库重建成功"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"知识库重建失败: {str(e)}") from e
