# controller/controller.py

import os
import logging

from fastapi import APIRouter, Body, HTTPException
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse
from pydantic import BaseModel

from app import create_scene  # ensure this points to your root app.py

router = APIRouter(prefix="/api/v1", tags=["scenes"])
logger = logging.getLogger("controller")
logging.basicConfig(level=logging.INFO)


class SceneRequest(BaseModel):
    scene_description: str


class SceneResponse(BaseModel):
    message: str
    video_filename: str
    code: str


@router.post(
    "/createScene",
    response_model=SceneResponse,
    summary="Create and render a Manim scene",
)
async def create_scene_endpoint(
    payload: SceneRequest = Body(
        ...,
        example={"scene_description": "A yellow circle zooming in"},
    )
):
    """
    Accepts a scene description, generates & renders the Manim scene,
    and returns the video filename for later retrieval.
    """
    try:
        logger.info("Rendering scene: %s", payload.scene_description)

        # Offload blocking work and unpack both return values
        message, video_path, generated_code = await run_in_threadpool(
            create_scene, payload.scene_description
        )
        logger.info("Render completed, video at: %s", video_path)

        filename = os.path.basename(video_path)
        return SceneResponse(
            message= message ,
            video_filename=filename,
            code=generated_code,
        )

    except Exception as e:
        logger.exception("Error in create_scene_endpoint")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to create scene: {e}"
        )


@router.get(
    "/video/{video_filename}",
    summary="Retrieve a previously generated video"
)
async def get_video(video_filename: str):
    """
    Serve the rendered video file from disk.
    """
    video_dir = os.path.join("temp", "videos")
    file_path = os.path.join(video_dir, video_filename)

    if not os.path.isfile(file_path):
        logger.warning("Requested video does not exist: %s", file_path)
        raise HTTPException(status_code=404, detail="Video not found")

    # FileResponse will handle range requests, streaming, etc.
    return FileResponse(
        path=file_path,
        media_type="video/mp4",
        filename=video_filename
    )
