from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from ..server.middleware import get_image_store_dir

router = APIRouter()


@router.get("/images/{filename}", tags=["Images"])
async def get_image(filename: str):
    image_store = get_image_store_dir()
    file_path = image_store / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Image not found")
    return FileResponse(file_path)
