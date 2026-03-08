import asyncio

from fastapi import APIRouter, HTTPException, Request

from api.schemas.agent import ChatRequest, ChatResponse

router = APIRouter(prefix="/agent", tags=["agent"])


@router.post("/chat", response_model=ChatResponse)
async def agent_chat(payload: ChatRequest, request: Request):
    agent = request.app.state.agent
    try:
        chunks: list[str] = []

        def run_agent():
            for message in agent.execute_stream(payload.query):
                content = message.content
                if isinstance(content, str):
                    chunks.append(content)
                elif isinstance(content, list):
                    chunks.extend(str(item) for item in content)

        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, run_agent)

        return ChatResponse(answer="".join(chunks).strip())
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Agent调用失败: {str(e)}") from e
