import asyncio

from fastapi import APIRouter, HTTPException, Request

from api.schemas.rag import ReloadResponse

router = APIRouter(prefix="/rag", tags=["rag"])


@router.post("/reload", response_model=ReloadResponse)
async def reload_knowledge_base(request: Request):
    vector_store = request.app.state.vector_store
    try:
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, vector_store.load_document)
        return ReloadResponse(message="知识库重建成功")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"知识库重建失败: {str(e)}") from e
