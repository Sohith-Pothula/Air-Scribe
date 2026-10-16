"""
AirScribe web server.

    python -m backend.main          (or: uvicorn backend.main:app --reload)

Then open http://localhost:8000
"""
import base64
import os

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, Field

from .recognizer import ROOT, Recognizer, RecognitionError

FRONTEND_DIR = os.path.join(ROOT, "frontend")
HAND_MODEL_PATH = os.path.join(ROOT, "hand_landmarker.task")
MAX_IMAGE_BYTES = 5 * 1024 * 1024

app = FastAPI(title="AirScribe", version="2.0")
recognizer = Recognizer()


class PredictRequest(BaseModel):
    image: str = Field(..., description="PNG data URL: white strokes on black")
    mode: str = Field("word", pattern="^(word|digit)$")


@app.get("/api/health")
def health():
    return {
        "ok": True,
        "word": recognizer.gemini_available,
        "digit": recognizer.digit_available,
    }


@app.post("/api/predict")
async def predict(req: PredictRequest):
    payload = req.image.split(",", 1)[-1]  # strip "data:image/png;base64,"
    try:
        image_bytes = base64.b64decode(payload, validate=True)
    except Exception:
        raise HTTPException(400, "Image is not valid base64.")
    if len(image_bytes) > MAX_IMAGE_BYTES:
        raise HTTPException(413, "Image too large.")

    try:
        # Gemini / TensorFlow calls block, so keep them off the event loop
        return await run_in_threadpool(recognizer.predict, image_bytes, req.mode)
    except RecognitionError as e:
        raise HTTPException(422, str(e))


@app.get("/models/hand_landmarker.task")
def hand_model():
    """Serve the local MediaPipe hand model so the browser doesn't need to download it elsewhere."""
    if not os.path.exists(HAND_MODEL_PATH):
        raise HTTPException(404, "hand_landmarker.task not found")
    return FileResponse(HAND_MODEL_PATH, media_type="application/octet-stream",
                        headers={"Cache-Control": "public, max-age=86400"})


# The frontend is plain static files, mounted last so /api routes win.
app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 8000))
    print(f"AirScribe running at http://localhost:{port}")
    uvicorn.run("backend.main:app", host="0.0.0.0", port=port)
