"""
AirScribe: write WORDS in the air with your RIGHT INDEX FINGER, then get the word + its meaning.
Recognition uses Google Gemini (put GEMINI_API_KEY=... in a .env file next to this script).
Without a key it falls back to the offline single-digit model.

Gestures (right hand):
  Index finger up only          -> DRAW (fingertip leaves a red trail)
  Index + middle finger up      -> MOVE (move without drawing, "pen up")
  Open palm, hold ~1 second     -> PREDICT the word and show its meaning
Keys (click the window first):
  P  predict   C  clear   Q  quit
"""
import json
import os
import threading
import time

import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core.base_options import BaseOptions

MODEL_PATH = "digit_cnn.keras"
HAND_MODEL_PATH = "hand_landmarker.task"

STROKE_COLOR = (0, 0, 255)   # BGR: red. Use (255, 0, 0) for blue.
STROKE_THICKNESS = 4
SMOOTHING = 0.4              # 0 = no smoothing, closer to 1 = smoother but laggier
MAX_JUMP = 200               # px; bigger jumps between frames are not connected
PALM_HOLD_SECONDS = 1.0
LOST_GRACE_FRAMES = 8        # keep the stroke alive if the hand is lost for a few frames
DEBOUNCE_ENTER_DRAW = 2      # frames a gesture must be held before drawing starts
DEBOUNCE_LEAVE = 4           # frames before switching away from the current gesture

FONT = cv2.FONT_HERSHEY_SIMPLEX

# Landmark indices: (tip, pip) for index, middle, ring, pinky
FINGERS = {"index": (8, 6), "middle": (12, 10), "ring": (16, 14), "pinky": (20, 18)}

for path in (MODEL_PATH, HAND_MODEL_PATH):
    if not os.path.exists(path):
        raise FileNotFoundError(f"'{path}' not found. See README for how to get it.")

model = tf.keras.models.load_model(MODEL_PATH)


def preprocess(canvas):
    """Convert the drawn strokes into a 28x28 image shaped like MNIST."""
    gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    # Strokes are drawn thin for display; thicken them so they look like MNIST pen strokes
    gray = cv2.dilate(gray, np.ones((9, 9), np.uint8))
    ys, xs = np.nonzero(gray)
    if len(xs) == 0:
        return None

    # Crop tightly around the strokes
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


def predict_digit(canvas):
    img = preprocess(canvas)
    if img is None:
        return ["Draw something first!"]
    probs = model.predict(img, verbose=0)[0]
    return [f"Digit: {int(probs.argmax())} ({probs.max() * 100:.1f}%)",
            "(Add GEMINI_API_KEY to .env to read words + meanings)"]


# ---------------------------------------------------------------- Gemini word recognition
def load_api_key():
    """Read the key from the environment or a .env file next to this script."""
    for name in ("GEMINI_API_KEY", "GOOGLE_API_KEY"):
        if os.environ.get(name):
            return os.environ[name].strip()
    env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
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
# Tried in order until one succeeds (verified working). Override with GEMINI_MODEL env var.
GEMINI_MODELS = [os.environ.get("GEMINI_MODEL", "gemini-3.5-flash"),
                 "gemini-3.1-flash-lite", "gemini-flash-latest"]
_gemini_client = None

GEMINI_PROMPT = (
    "This image shows handwriting drawn in the air with a fingertip, so strokes are shaky "
    "and letters may be rough or uneven. Read the word (or short phrase / number) that was "
    "written. If it is a real word, give a short, simple dictionary meaning (max 25 words). "
    "If it is a number, say what number it is. If you cannot read it, set text to '?'. "
    'Reply ONLY as JSON: {"text": "...", "meaning": "..."}'
)


def canvas_to_png(canvas):
    """Crop the strokes and render them black-on-white, which is easiest to read."""
    gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    ys, xs = np.nonzero(gray)
    if len(xs) == 0:
        return None
    pad = 30
    y1, y2 = max(ys.min() - pad, 0), min(ys.max() + pad, gray.shape[0])
    x1, x2 = max(xs.min() - pad, 0), min(xs.max() + pad, gray.shape[1])
    crop = cv2.dilate(gray[y1:y2, x1:x2], np.ones((3, 3), np.uint8))
    ok, buf = cv2.imencode(".png", 255 - crop)
    return buf.tobytes() if ok else None


def predict_word(canvas):
    global _gemini_client
    from google import genai
    from google.genai import types

    png = canvas_to_png(canvas)
    if png is None:
        return ["Draw something first!"]
    if _gemini_client is None:
        _gemini_client = genai.Client(api_key=API_KEY)

    last_error = None
    for model_name in dict.fromkeys(GEMINI_MODELS):  # de-duplicated, order kept
        try:
            response = _gemini_client.models.generate_content(
                model=model_name,
                contents=[types.Part.from_bytes(data=png, mime_type="image/png"), GEMINI_PROMPT],
                config=types.GenerateContentConfig(response_mime_type="application/json",
                                                   temperature=0.2),
            )
            data = json.loads(response.text)
            text = str(data.get("text", "?")).strip() or "?"
            meaning = str(data.get("meaning", "")).strip()
            print(f"[AirScribe] {text}: {meaning}")
            return [f"Word: {text}", f"Meaning: {meaning}" if meaning else ""]
        except Exception as e:  # try the next model, then report
            last_error = e
    print(f"[AirScribe] Gemini error: {last_error}")
    return ["Gemini error:", str(last_error)[:150]]


def predict(canvas):
    return predict_word(canvas) if API_KEY else predict_digit(canvas)


class Predictor:
    """Runs predictions in a background thread so the camera feed never freezes."""

    def __init__(self):
        self.lines = []
        self.busy = False

    def start(self, canvas):
        if self.busy:
            return
        self.busy = True
        self.lines = ["Thinking..."]
        snapshot = canvas.copy()

        def run():
            try:
                self.lines = predict(snapshot)
            except Exception as e:
                self.lines = ["Error:", str(e)[:150]]
            finally:
                self.busy = False

        threading.Thread(target=run, daemon=True).start()

    def clear(self):
        if not self.busy:
            self.lines = []


def wrap_text(text, max_width, scale, thickness):
    words, lines, current = text.split(), [], ""
    for word in words:
        trial = f"{current} {word}".strip()
        if cv2.getTextSize(trial, FONT, scale, thickness)[0][0] <= max_width or not current:
            current = trial
        else:
            lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines


def draw_result_panel(img, lines):
    """Semi-transparent dark box with the word + meaning, wrapped to fit the window."""
    lines = [l for l in lines if l]
    if not lines:
        return
    h, w = img.shape[:2]
    scale, thick, line_h = 0.7, 2, 30
    wrapped = []
    for i, line in enumerate(lines):
        for part in wrap_text(line, w - 60, scale, thick):
            wrapped.append((part, i == 0))
    box_h = line_h * len(wrapped) + 20
    top = 75
    overlay = img.copy()
    cv2.rectangle(overlay, (15, top), (w - 15, top + box_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, img, 0.4, 0, img)
    for j, (part, is_title) in enumerate(wrapped):
        color = (0, 255, 255) if is_title else (255, 255, 255)
        cv2.putText(img, part, (30, top + 30 + j * line_h), FONT, scale, color, thick)


def create_hand_landmarker():
    options = vision.HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=HAND_MODEL_PATH),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.4,
        min_tracking_confidence=0.3,
    )
    return vision.HandLandmarker.create_from_options(options)


def pick_right_hand(result):
    """Return landmarks of the right hand (or the only visible hand), else None."""
    if not result.hand_landmarks:
        return None
    for landmarks, handedness in zip(result.hand_landmarks, result.handedness):
        # The frame is mirrored before detection, so "Right" is the user's right hand.
        if handedness and handedness[0].category_name == "Right":
            return landmarks
    if len(result.hand_landmarks) == 1:
        return result.hand_landmarks[0]
    return None


def fingers_up(landmarks):
    """A finger is 'up' (extended) when its tip is clearly farther from the wrist than its
    middle joint. Works at any hand angle, unlike comparing only y-coordinates."""
    wrist = landmarks[0]

    def dist(a, b):
        return ((a.x - b.x) ** 2 + (a.y - b.y) ** 2) ** 0.5

    return {name: dist(landmarks[tip], wrist) > dist(landmarks[pip], wrist) * 1.1
            for name, (tip, pip) in FINGERS.items()}


def classify(up):
    if all(up.values()):
        return "PALM"
    if up["index"] and up["middle"]:
        return "MOVE"
    if up["index"]:
        return "DRAW"
    return "IDLE"


def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam.")

    # Create the window explicitly and keep it on top so it isn't hidden behind the IDE
    cv2.namedWindow("AirScribe", cv2.WINDOW_NORMAL)
    cv2.setWindowProperty("AirScribe", cv2.WND_PROP_TOPMOST, 1)

    landmarker = create_hand_landmarker()
    canvas = None
    prev_point = None
    smooth_point = None
    palm_start = None
    predictor = Predictor()
    print("[AirScribe] Mode:", "Gemini word recognition" if API_KEY else
          "offline digits (no GEMINI_API_KEY found in .env)")
    last_ts = 0
    mode = "NO HAND"          # stable (debounced) mode
    candidate, candidate_count = None, 0
    lost_frames = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)  # mirror view feels natural
        h, w = frame.shape[:2]
        if canvas is None:
            canvas = np.zeros_like(frame)

        # --- Detect the hand ---
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        ts = max(int(time.monotonic() * 1000), last_ts + 1)  # must strictly increase
        last_ts = ts
        result = landmarker.detect_for_video(mp_image, ts)
        hand = pick_right_hand(result)

        raw_mode = "NO HAND"
        tip = None
        if hand is not None:
            lost_frames = 0
            raw = np.array([hand[8].x * w, hand[8].y * h])
            smooth_point = raw if smooth_point is None else SMOOTHING * smooth_point + (1 - SMOOTHING) * raw
            tip = (int(smooth_point[0]), int(smooth_point[1]))
            raw_mode = classify(fingers_up(hand))

            # Light hand skeleton dots for feedback
            for lm in hand:
                cv2.circle(frame, (int(lm.x * w), int(lm.y * h)), 3, (255, 255, 255), -1)
        else:
            lost_frames += 1

        # --- Debounce: a new gesture must be seen for a few frames before switching ---
        if raw_mode == "NO HAND":
            # Short tracking dropouts keep the current mode (stroke continues afterwards)
            if lost_frames > LOST_GRACE_FRAMES:
                mode, smooth_point, prev_point = "NO HAND", None, None
        else:
            if raw_mode == candidate:
                candidate_count += 1
            else:
                candidate, candidate_count = raw_mode, 1
            needed = DEBOUNCE_ENTER_DRAW if raw_mode == "DRAW" else DEBOUNCE_LEAVE
            if raw_mode != mode and candidate_count >= needed:
                mode = raw_mode

        # --- Draw on the canvas ---
        if mode == "DRAW" and tip is not None:
            if prev_point is not None and np.hypot(tip[0] - prev_point[0], tip[1] - prev_point[1]) < MAX_JUMP:
                cv2.line(canvas, prev_point, tip, (255, 255, 255), STROKE_THICKNESS, cv2.LINE_AA)
            prev_point = tip
        elif mode != "DRAW":
            prev_point = None
        # (mode == DRAW but hand briefly lost: keep prev_point so the stroke reconnects)

        # --- Open palm held => predict ---
        if mode == "PALM":
            if palm_start is None:
                palm_start = time.monotonic()
            elif time.monotonic() - palm_start >= PALM_HOLD_SECONDS:
                predictor.start(canvas)
                palm_start = float("inf")  # fire once per palm gesture
        else:
            palm_start = None

        # --- Overlay strokes on the live frame (red with dark outline) ---
        output = frame.copy()
        stroke_mask = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY) > 0
        outline_mask = cv2.dilate(stroke_mask.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
        output[outline_mask] = (0, 0, 0)
        output[stroke_mask] = STROKE_COLOR

        # Fingertip cursor
        if tip is not None:
            cursor_color = (0, 0, 255) if mode == "DRAW" else (0, 255, 0)
            cv2.circle(output, tip, 12, cursor_color, 3)

        # --- HUD ---
        mode_colors = {"DRAW": (0, 0, 255), "MOVE": (0, 255, 0), "PALM": (0, 255, 255),
                       "IDLE": (200, 200, 200), "NO HAND": (120, 120, 120)}
        cv2.putText(output, mode, (20, 40), FONT, 1, mode_colors[mode], 2)
        if mode == "PALM" and palm_start not in (None, float("inf")):
            progress = min((time.monotonic() - palm_start) / PALM_HOLD_SECONDS, 1.0)
            cv2.rectangle(output, (20, 55), (20 + int(200 * progress), 65), (0, 255, 255), -1)
        draw_result_panel(output, predictor.lines)
        cv2.putText(output, "1 finger: draw | 2 fingers: move | palm: predict | C: clear  Q: quit",
                    (20, h - 20), FONT, 0.5, (255, 255, 255), 1)
        cv2.imshow("AirScribe", output)

        # --- Keyboard controls ---
        key = cv2.waitKey(1) & 0xFF
        if key == ord("c"):
            canvas[:] = 0
            predictor.clear()
        elif key == ord("p"):
            predictor.start(canvas)
        elif key == ord("q"):
            break

    landmarker.close()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
