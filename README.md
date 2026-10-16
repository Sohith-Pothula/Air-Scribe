# AirScribe

Write words in the air with your index finger, in your browser, and get the word and its meaning back.

## Quick start

Requires Python 3.10–3.12 and a webcam.

```bash
python -m venv venv
venv\Scripts\activate            # macOS/Linux: source venv/bin/activate
pip install -r requirements.txt
```

Add your Gemini key to `.env` in the project root (optional, needed for **Word** mode):

```
GEMINI_API_KEY=your_key_here
```

Run the server:

```bash
python -m backend.main
```

Then open **http://localhost:8000** and click **Start camera**.

> No key? The **Digit** mode still works offline with the CNN in `digit_cnn.keras`
> (create it with `python train.py` if it's missing).

## Using it

| Gesture / key            | Action                    |
|--------------------------|---------------------------|
| ☝️ Index finger up        | Draw                      |
| ✌️ Index + middle up      | Move without drawing      |
| 🖐️ Open palm for 1 sec    | Read the word             |
| `P`                      | Read the word             |
| `Z`                      | Undo last stroke          |
| `C`                      | Clear                     |

You can also draw with a mouse, pen, or touch screen.

## How it works

```
Browser                                   FastAPI backend
┌─────────────────────────────┐           ┌──────────────────────────┐
│ Webcam → MediaPipe (WASM)   │           │ POST /api/predict        │
│ 21 hand points per frame    │   PNG     │  ├ Word  → Gemini         │
│ Gesture → strokes → canvas  │ ───────▶  │  └ Digit → MNIST CNN      │
│ Shows word + meaning        │ ◀───────  │ returns {text, meaning}  │
└─────────────────────────────┘   JSON    └──────────────────────────┘
```

1. **Track** – the browser runs MediaPipe's hand model (`hand_landmarker.task`, served by the backend) on every camera frame. Video never leaves your machine.
2. **Gesture** – a finger is "up" when its tip is farther from the wrist than its middle joint. Index only = draw, index + middle = move, all four = palm.
3. **Smooth** – the fingertip is exponentially smoothed, gestures are debounced over a few frames, and strokes are drawn as quadratic curves, so lines stay clean.
4. **Read** – the strokes are rendered white-on-black and posted to the backend. Gemini reads the handwriting and returns a short meaning; Digit mode crops/centres the strokes into a 28×28 MNIST-style image for the CNN.

The backend exists mainly to keep your Gemini API key secret and to run the TensorFlow model.

## Project layout

```
backend/
  main.py         FastAPI app: API routes + serves the frontend
  recognizer.py   Gemini word reading + offline CNN digit model
frontend/
  index.html      Page markup
  style.css       Design
  app.js          Camera, hand tracking, gestures, drawing, API calls
digit_cnn.keras   Trained digit model (from train.py)
hand_landmarker.task  MediaPipe hand model
train.py          Trains the digit CNN on MNIST
app.py            Original desktop (OpenCV window) version
```

## API

| Method | Path            | Body / response |
|--------|-----------------|-----------------|
| GET    | `/api/health`   | `{ ok, word, digit }` – which modes are available |
| POST   | `/api/predict`  | `{ image: "data:image/png;base64,...", mode: "word" \| "digit" }` → `{ text, meaning, confidence, engine }` |

Interactive docs: http://localhost:8000/docs
