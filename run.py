import uvicorn

if __name__ == "__main__":
    # Run the FastAPI app from main.py at 0.0.0.0:8000 with auto-reload enabled
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)