import os

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from src.controller.controller import router  # adjust the import to your folder structure

app = FastAPI(title="ANI_AI Scene Generator")

# Mount the controller's API routes
app.include_router(router)

# Serve all files in ./static (including generated videos) under /static
app.mount(
    "/src/static",
    StaticFiles(directory="src/static", html=False),
    name="static",
)


# Serve index.html at the root
@app.get("/", response_class=FileResponse, summary="Main page")
async def serve_index():
    index_path = os.path.join("src","static", "index.html")
    return index_path
