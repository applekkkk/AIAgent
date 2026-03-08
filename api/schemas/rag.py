from pydantic import BaseModel


class ReloadResponse(BaseModel):
    message: str
