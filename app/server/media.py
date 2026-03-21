from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse

from app.server.middleware import get_media_store_dir, verify_media_token

router = APIRouter()


@router.get("/media/{filename}", tags=["Media"])
async def get_media(filename: str, token: str | None = Query(default=None)):
    if not verify_media_token(filename, token):
        raise HTTPException(status_code=403, detail="Invalid token")

    media_store = get_media_store_dir()
    file_path = media_store / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Media not found")
    return FileResponse(file_path)
