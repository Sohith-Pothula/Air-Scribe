"""
Recognition engine for AirScribe.

Takes a stroke image (white strokes on a black background, exactly what the browser
draws) and returns what was written:
  * "word"  -> Google Gemini reads the handwriting and gives a short meaning
  * "digit" -> the offline MNIST CNN (digit_cnn.keras) predicts a single digit
"""
import json
import os
import threading

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DIGIT_MODEL_PATH = os.path.join(ROOT, "digit_cnn.keras")

GEMINI_PROMPT = (
    "This image shows handwriting drawn in the air with a fingertip, so strokes are shaky "
    "and letters may be rough or uneven. Read the word (or short phrase / number) that was "
    "written. If it is a real word, give a short, simple dictionary meaning (max 25 words). "
    "If it is a number, say what number it is. If you cannot read it, set text to '?'. "
    'Reply ONLY as JSON: {"text": "...", "meaning": "..."}'
)


class RecognitionError(Exception):
    """Raised for problems the user should see (empty canvas, API failure, ...)."""


# ------------------------------------------------------------------ config
def load_api_key():
    """Read the key from the environment or the .env file in the project root."""
    for name in ("GEMINI_API_KEY", "GOOGLE_API_KEY"):
        if os.environ.get(name):
            return os.environ[name].strip()
    env_path = os.path.join(ROOT, ".env")
    if os.path.exists(env_path):
        with open(env_path, encoding="utf-8") as f:
            for line in f:
                key, _, value = line.partition("=")
                if key.strip() in ("GEMINI_API_KEY", "GOOGLE_API_KEY") and value.strip():
                    value = value.strip().strip('"').strip("'")
                    if value and "PASTE_YOUR_KEY" not in value:
                        return value
    return None


API_KEY = load_api_key()
# Tried in order until one succeeds. Override the first with the GEMINI_MODEL env var.
GEMINI_MODELS = list(dict.fromkeys([
    os.environ.get("GEMINI_MODEL", "gemini-3.5-flash"),
    "gemini-3.1-flash-lite",
    "gemini-flash-latest",
]))


# ------------------------------------------------------------------ image helpers
def decode_image(data: bytes):
    """PNG/JPEG bytes -> grayscale uint8 image (strokes > 0)."""
    arr = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise RecognitionError("Could not read the drawing.")
    # Drop anti-aliasing noise so only real strokes count
    img[img < 40] = 0
    return img


def crop_for_gemini(gray):
    """Crop the strokes and render them black-on-white, which is easiest to read."""
    ys, xs = np.nonzero(gray)
    if len(xs) == 0:
        return None
    pad = 30
    y1, y2 = max(ys.min() - pad, 0), min(ys.max() + pad, gray.shape[0])
    x1, x2 = max(xs.min() - pad, 0), min(xs.max() + pad, gray.shape[1])
    crop = cv2.dilate(gray[y1:y2, x1:x2], np.ones((3, 3), np.uint8))
    ok, buf = cv2.imencode(".png", 255 - crop)
    return buf.tobytes() if ok else None


def preprocess_mnist(gray):
    """Convert the drawn strokes into a 28x28 image shaped like MNIST."""
    # Strokes are drawn thin for display; thicken them so they look like MNIST pen strokes
    gray = cv2.dilate(gray, np.ones((9, 9), np.uint8))
    ys, xs = np.nonzero(gray)
    if len(xs) == 0:
        return None
    x1, x2, y1, y2 = xs.min(), xs.max(), ys.min(), ys.max()
    digit = gray[y1:y2 + 1, x1:x2 + 1]

    # Pad to a square so the digit isn't stretched
    h, w = digit.shape
    size = max(h, w)
    square = np.zeros((size, size), np.uint8)
    top, left = (size - h) // 2, (size - w) // 2
    square[top:top + h, left:left + w] = digit

    # Resize to 20x20 and center in a 28x28 frame (MNIST convention)
    digit = cv2.resize(square, (20, 20), interpolation=cv2.INTER_AREA)
    img = np.zeros((28, 28), np.float32)
    img[4:24, 4:24] = digit
    return img.reshape(1, 28, 28, 1) / 255.0


# ------------------------------------------------------------------ recognizers
class Recognizer:
    def __init__(self):
        self._gemini = None
        self._digit_model = None
        self._lock = threading.Lock()

    @property
    def gemini_available(self):
        return bool(API_KEY)

    @property
    def digit_available(self):
        return os.path.exists(DIGIT_MODEL_PATH)

    # TensorFlow is slow to import, so it is only loaded on the first digit request.
    def _load_digit_model(self):
        with self._lock:
            if self._digit_model is None:
                import tensorflow as tf
                self._digit_model = tf.keras.models.load_model(DIGIT_MODEL_PATH)
        return self._digit_model

    def predict_digit(self, gray):
        if not self.digit_available:
            raise RecognitionError("digit_cnn.keras not found. Run: python train.py")
        img = preprocess_mnist(gray)
        if img is None:
            raise RecognitionError("Draw something first!")
        probs = self._load_digit_model().predict(img, verbose=0)[0]
        digit = int(probs.argmax())
        return {
            "text": str(digit),
            "meaning": f"Recognised by the offline CNN with {probs.max() * 100:.1f}% confidence.",
            "confidence": round(float(probs.max()), 4),
            "engine": "cnn",
        }

    def predict_word(self, gray):
        if not self.gemini_available:
            raise RecognitionError("No GEMINI_API_KEY in .env, so only Digit mode is available.")
        png = crop_for_gemini(gray)
        if png is None:
            raise RecognitionError("Draw something first!")

        from google import genai
        from google.genai import types

        if self._gemini is None:
            self._gemini = genai.Client(api_key=API_KEY)

        last_error = None
        for model_name in GEMINI_MODELS:
            try:
                response = self._gemini.models.generate_content(
                    model=model_name,
                    contents=[types.Part.from_bytes(data=png, mime_type="image/png"), GEMINI_PROMPT],
                    config=types.GenerateContentConfig(response_mime_type="application/json",
                                                       temperature=0.2),
                )
                data = json.loads(response.text)
                text = str(data.get("text", "?")).strip() or "?"
                return {
                    "text": text,
                    "meaning": str(data.get("meaning", "")).strip(),
                    "confidence": None,
                    "engine": model_name,
                }
            except Exception as e:  # try the next model, then report
                last_error = e
        raise RecognitionError(f"Gemini error: {str(last_error)[:200]}")

    def predict(self, image_bytes: bytes, mode: str):
        gray = decode_image(image_bytes)
        if mode == "word":
            return self.predict_word(gray)
        return self.predict_digit(gray)
