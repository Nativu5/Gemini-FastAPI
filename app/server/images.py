from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse

from ..server.middleware import get_image_store_dir, verify_image_token

router = APIRouter()


@router.get("/images/{filename}", tags=["Images"])
async def get_image(filename: str, token: str | None = Query(default=None)):
    if not verify_image_token(filename, token):
        raise HTTPException(status_code=403, detail="Invalid token")

    image_store = get_image_store_dir()
    file_path = image_store / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Image not found")
    return FileResponse(file_path)
