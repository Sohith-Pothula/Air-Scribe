/**
 * AirScribe frontend
 * ------------------
 * 1. Webcam frames -> MediaPipe HandLandmarker (runs in the browser via WebAssembly)
 * 2. Finger positions -> gesture (DRAW / MOVE / PALM / IDLE), smoothed + debounced
 * 3. Fingertip path   -> strokes, rendered on a <canvas> as smooth curves
 * 4. Strokes          -> PNG -> POST /api/predict -> word + meaning
 */
import { FilesetResolver, HandLandmarker } from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/vision_bundle.mjs";

// ------------------------------------------------------------------ tuning
const CFG = {
  smoothing: 0.45,       // 0 = raw fingertip, closer to 1 = smoother but laggier
  maxJump: 0.25,         // fraction of width; bigger jumps start a new stroke
  palmHoldMs: 1000,      // hold open palm this long to read the word
  lostGraceFrames: 8,    // keep the stroke alive through short tracking dropouts
  debounceDraw: 2,       // frames before DRAW starts
  debounceLeave: 4,      // frames before switching to any other gesture
  minPointDist: 0.002,   // ignore sub-pixel jitter when adding points
};

// Landmark indices: (tip, pip) for index, middle, ring, pinky
const FINGERS = { index: [8, 6], middle: [12, 10], ring: [16, 14], pinky: [20, 18] };
const MODE_LABEL = { DRAW: "Drawing", MOVE: "Moving", PALM: "Reading…", IDLE: "Idle", "NO HAND": "No hand", MOUSE: "Mouse" };

// ------------------------------------------------------------------ DOM
const $ = (id) => document.getElementById(id);
const stage = $("stage"), video = $("video"), ink = $("ink"), ctx = ink.getContext("2d");
const overlay = $("overlay"), modeChip = $("mode-chip"), modeLabel = $("mode-label");
const palmBar = $("palm-bar"), palmFill = palmBar.querySelector("span");
const resultBox = $("result"), resultWord = $("result-word"), resultMeaning = $("result-meaning"), resultEngine = $("result-engine");
const btnPredict = $("btn-predict"), segmented = document.querySelector(".segmented");

// ------------------------------------------------------------------ state
const state = {
  strokes: [],           // [[{x,y}, ...], ...] in normalised (0..1) mirrored coords
  current: null,         // stroke being drawn
  mode: "MOUSE",         // stable gesture mode
  candidate: null, candidateCount: 0, lostFrames: 0,
  smooth: null, tip: null, hand: null,
  palmStart: null,
  recognition: "word",
  busy: false,
  live: false,
  lastVideoTime: -1,
  dirty: true,
};

let landmarker = null;
let landmarkerPromise = null;

// ------------------------------------------------------------------ backend
async function checkServer() {
  const el = $("server-status"), label = el.querySelector("span");
  try {
    const res = await fetch("/api/health");
    const h = await res.json();
    if (h.word) {
      el.className = "status ok"; label.textContent = "Gemini ready";
    } else {
      el.className = "status warn"; label.textContent = "Offline digits";
      $("mode-word").disabled = true;
      setRecognition("digit");
    }
    if (!h.digit) $("mode-digit").disabled = true;
  } catch {
    el.className = "status err"; label.textContent = "Server offline";
  }
}

async function predict() {
  if (state.busy) return;
  if (!state.strokes.some((s) => s.length > 1)) {
    toast("Draw something first");
    return;
  }
  state.busy = true;
  btnPredict.classList.add("loading");
  btnPredict.disabled = true;
  resultBox.classList.remove("error", "reveal");
  resultBox.classList.add("thinking");

  try {
    const res = await fetch("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ image: exportStrokes(), mode: state.recognition }),
    });
    const data = await res.json().catch(() => ({}));
    if (!res.ok) throw new Error(data.detail || `Request failed (${res.status})`);
    showResult(data);
    addRecent(data);
  } catch (err) {
    showError(err.message || "Something went wrong");
  } finally {
    state.busy = false;
    btnPredict.classList.remove("loading");
    btnPredict.disabled = false;
    resultBox.classList.remove("thinking");
  }
}

/** Render strokes white-on-black at camera resolution, the format the backend expects. */
function exportStrokes() {
  const w = video.videoWidth || 640, h = video.videoHeight || 480;
  const off = document.createElement("canvas");
  off.width = w; off.height = h;
  const c = off.getContext("2d");
  c.fillStyle = "#000"; c.fillRect(0, 0, w, h);
  c.strokeStyle = "#fff";
  c.lineWidth = Math.max(4, w / 150);
  c.lineCap = c.lineJoin = "round";
  for (const s of state.strokes) traceStroke(c, s, w, h);
  return off.toDataURL("image/png");
}

// ------------------------------------------------------------------ result UI
function showResult({ text, meaning, engine, confidence }) {
  resultWord.textContent = text;
  resultMeaning.textContent = meaning || "No meaning available.";
  resultEngine.textContent = engine === "cnn"
    ? `Offline CNN · ${(confidence * 100).toFixed(1)}%`
    : `via ${engine}`;
  replay(resultBox, "reveal");
}

function showError(msg) {
  resultWord.textContent = "Couldn't read that";
  resultMeaning.textContent = msg;
  resultEngine.textContent = "";
  resultBox.classList.add("error");
  replay(resultBox, "reveal");
}

function replay(el, cls) {
  el.classList.remove(cls);
  void el.offsetWidth; // restart the CSS animation
  el.classList.add(cls);
}

const recent = JSON.parse(localStorage.getItem("airscribe.recent") || "[]");
function addRecent(item) {
  if (!item.text || item.text === "?") return;
  recent.unshift({ text: item.text, meaning: item.meaning, engine: item.engine, confidence: item.confidence });
  recent.splice(8);
  localStorage.setItem("airscribe.recent", JSON.stringify(recent));
  renderRecent();
}
function renderRecent() {
  const list = $("recent-list");
  list.innerHTML = "";
  if (!recent.length) {
    list.innerHTML = '<li class="muted empty">Nothing yet</li>';
    return;
  }
  recent.forEach((r, i) => {
    const li = document.createElement("li");
    li.textContent = r.text;
    li.title = r.meaning || "";
    li.style.animationDelay = `${i * 30}ms`;
    li.onclick = () => showResult(r);
    list.appendChild(li);
  });
}

let toastTimer;
function toast(msg) {
  const t = $("toast");
  t.textContent = msg;
  t.classList.add("show");
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => t.classList.remove("show"), 2200);
}

// ------------------------------------------------------------------ canvas
function resizeCanvas() {
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  const { width, height } = stage.getBoundingClientRect();
  ink.width = Math.round(width * dpr);
  ink.height = Math.round(height * dpr);
  state.dirty = true;
}

/** Draw a stroke as quadratic curves through midpoints: smooth, no corners. */
function traceStroke(c, pts, w, h) {
  if (pts.length < 2) return;
  c.beginPath();
  c.moveTo(pts[0].x * w, pts[0].y * h);
  for (let i = 1; i < pts.length - 1; i++) {
    const mx = (pts[i].x + pts[i + 1].x) / 2, my = (pts[i].y + pts[i + 1].y) / 2;
    c.quadraticCurveTo(pts[i].x * w, pts[i].y * h, mx * w, my * h);
  }
  const last = pts[pts.length - 1];
  c.lineTo(last.x * w, last.y * h);
  c.stroke();
}

function render() {
  const w = ink.width, h = ink.height, dpr = w / stage.clientWidth || 1;
  ctx.clearRect(0, 0, w, h);
  ctx.lineCap = ctx.lineJoin = "round";

  // dark outline then the red ink, so strokes read on any background
  if (state.live) {
    ctx.strokeStyle = "rgba(0,0,0,0.55)";
    ctx.lineWidth = 9 * dpr;
    for (const s of state.strokes) traceStroke(ctx, s, w, h);
  }
  ctx.strokeStyle = "#ff5a45";
  ctx.lineWidth = 5 * dpr;
  for (const s of state.strokes) traceStroke(ctx, s, w, h);

  // hand landmarks
  if (state.hand) {
    ctx.fillStyle = "rgba(255,255,255,0.75)";
    for (const lm of state.hand) {
      ctx.beginPath();
      ctx.arc((1 - lm.x) * w, lm.y * h, 2.4 * dpr, 0, Math.PI * 2);
      ctx.fill();
    }
  }

  // fingertip cursor
  if (state.tip) {
    const color = state.mode === "DRAW" ? "#ff5a45" : state.mode === "PALM" ? "#facc15" : "#4ade80";
    ctx.strokeStyle = color;
    ctx.lineWidth = 2.5 * dpr;
    ctx.beginPath();
    ctx.arc(state.tip.x * w, state.tip.y * h, (state.mode === "DRAW" ? 8 : 12) * dpr, 0, Math.PI * 2);
    ctx.stroke();
  }
  state.dirty = false;
}

function setMode(mode) {
  if (modeChip.dataset.mode === mode) return;
  modeChip.dataset.mode = mode;
  modeLabel.textContent = MODE_LABEL[mode] || mode;
  stage.classList.toggle("drawing", mode === "DRAW");
}

// ------------------------------------------------------------------ strokes
function addPoint(p) {
  if (!state.current) {
    state.current = [p];
    state.strokes.push(state.current);
  } else {
    const last = state.current[state.current.length - 1];
    const d = Math.hypot(p.x - last.x, p.y - last.y);
    if (d > CFG.maxJump) {           // teleport -> start a fresh stroke
      state.current = [p];
      state.strokes.push(state.current);
    } else if (d > CFG.minPointDist) {
      state.current.push(p);
    }
  }
  state.dirty = true;
}
function endStroke() {
  state.current = null;
}
function clearAll() {
  state.strokes = [];
  state.current = null;
  state.dirty = true;
}
function undo() {
  state.strokes.pop();
  state.current = null;
  state.dirty = true;
}

// ------------------------------------------------------------------ gestures
const dist = (a, b) => Math.hypot(a.x - b.x, a.y - b.y);

/** A finger is "up" when its tip is clearly farther from the wrist than its middle joint. */
function fingersUp(lm) {
  const wrist = lm[0], up = {};
  for (const [name, [tip, pip]] of Object.entries(FINGERS)) {
    up[name] = dist(lm[tip], wrist) > dist(lm[pip], wrist) * 1.1;
  }
  return up;
}

function classify(up) {
  if (up.index && up.middle && up.ring && up.pinky) return "PALM";
  if (up.index && up.middle) return "MOVE";
  if (up.index) return "DRAW";
  return "IDLE";
}

/** Prefer the user's right hand (labelled "Left" because the video isn't flipped before detection),
 *  otherwise just use the first hand found. */
function pickHand(result) {
  const hands = result.landmarks || [];
  if (!hands.length) return null;
  for (let i = 0; i < hands.length; i++) {
    if (result.handednesses?.[i]?.[0]?.categoryName === "Left") return hands[i];
  }
  return hands[0];
}

function processHand(hand) {
  let raw = "NO HAND";
  if (hand) {
    state.lostFrames = 0;
    const p = { x: 1 - hand[8].x, y: hand[8].y };  // mirror to match the video
    state.smooth = state.smooth
      ? { x: CFG.smoothing * state.smooth.x + (1 - CFG.smoothing) * p.x,
          y: CFG.smoothing * state.smooth.y + (1 - CFG.smoothing) * p.y }
      : p;
    state.tip = state.smooth;
    raw = classify(fingersUp(hand));
  } else {
    state.lostFrames++;
    state.tip = null;
  }
  state.hand = hand;

  // debounce: a new gesture must hold for a few frames before we switch
  if (raw === "NO HAND") {
    if (state.lostFrames > CFG.lostGraceFrames) {
      state.mode = "NO HAND";
      state.smooth = null;
      endStroke();
    }
  } else {
    if (raw === state.candidate) state.candidateCount++;
    else { state.candidate = raw; state.candidateCount = 1; }
    const needed = raw === "DRAW" ? CFG.debounceDraw : CFG.debounceLeave;
    if (raw !== state.mode && state.candidateCount >= needed) state.mode = raw;
  }

  // drawing
  if (state.mode === "DRAW" && state.tip) addPoint({ ...state.tip });
  else if (state.mode !== "DRAW") endStroke();
  // (DRAW with hand briefly lost: keep the stroke open so it reconnects)

  // open palm held -> read
  if (state.mode === "PALM") {
    const now = performance.now();
    if (state.palmStart === null) state.palmStart = now;
    const progress = Math.min((now - state.palmStart) / CFG.palmHoldMs, 1);
    if (state.palmStart !== Infinity) {
      palmBar.classList.add("show");
      palmFill.style.transform = `scaleX(${progress})`;
      if (progress >= 1) {
        state.palmStart = Infinity;  // fire once per palm gesture
        predict();
      }
    } else {
      palmBar.classList.remove("show");
    }
  } else {
    state.palmStart = null;
    palmBar.classList.remove("show");
  }

  setMode(state.mode);
  state.dirty = true;
}

// ------------------------------------------------------------------ camera
function loadLandmarker() {
  if (landmarkerPromise) return landmarkerPromise;
  landmarkerPromise = (async () => {
    const fileset = await FilesetResolver.forVisionTasks(
      "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm");
    const opts = (delegate) => ({
      baseOptions: { modelAssetPath: "/models/hand_landmarker.task", delegate },
      runningMode: "VIDEO",
      numHands: 2,
      minHandDetectionConfidence: 0.5,
      minHandPresenceConfidence: 0.4,
      minTrackingConfidence: 0.3,
    });
    try {
      landmarker = await HandLandmarker.createFromOptions(fileset, opts("GPU"));
    } catch {
      landmarker = await HandLandmarker.createFromOptions(fileset, opts("CPU"));
    }
    return landmarker;
  })();
  landmarkerPromise.catch(() => { landmarkerPromise = null; });
  return landmarkerPromise;
}

const CAMERA_ERRORS = {
  NotAllowedError: "Camera permission was blocked. Click the camera icon in the address bar, allow it, then try again.",
  NotFoundError: "No camera found in this browser. Open http://localhost:8000 in Chrome or Edge (not an embedded/preview browser).",
  NotReadableError: "The camera is busy. Close other apps using it (Zoom, Teams, Camera app, app.py) and try again.",
  TimeoutError: "The camera didn't respond. Open http://localhost:8000 in Chrome or Edge, or check Windows camera privacy settings.",
  SecurityError: "Camera needs a secure page. Use http://localhost:8000 (not your IP address).",
};

function withTimeout(promise, ms) {
  return Promise.race([
    promise,
    new Promise((_, reject) => setTimeout(() => {
      const e = new Error("Camera timed out"); e.name = "TimeoutError"; reject(e);
    }, ms)),
  ]);
}

async function openCamera() {
  if (!navigator.mediaDevices?.getUserMedia) {
    const e = new Error("Camera API unavailable"); e.name = "SecurityError"; throw e;
  }
  // Fail fast when the browser can't see any camera at all (instead of hanging)
  const devices = await navigator.mediaDevices.enumerateDevices().catch(() => []);
  if (devices.length && !devices.some((d) => d.kind === "videoinput")) {
    const e = new Error("No camera"); e.name = "NotFoundError"; throw e;
  }
  try {
    return await withTimeout(navigator.mediaDevices.getUserMedia({
      video: { width: { ideal: 1280 }, height: { ideal: 720 }, facingMode: "user" }, audio: false,
    }), 20000);
  } catch (err) {
    // Some webcams reject specific resolutions: retry with defaults
    if (err.name === "OverconstrainedError" || err.name === "NotReadableError") {
      return withTimeout(navigator.mediaDevices.getUserMedia({ video: true, audio: false }), 20000);
    }
    throw err;
  }
}

async function startCamera() {
  const btn = $("start-camera"), note = $("overlay-note");
  btn.disabled = true;
  btn.textContent = "Waiting for camera…";
  note.textContent = "Allow camera access if your browser asks.";
  try {
    const stream = await openCamera();
    video.srcObject = stream;
    await video.play();
    if (!video.videoWidth) await new Promise((r) => video.addEventListener("loadedmetadata", r, { once: true }));
    stage.style.aspectRatio = `${video.videoWidth} / ${video.videoHeight}`;

    btn.textContent = "Loading hand model…";
    note.textContent = "First load can take a few seconds.";
    await loadLandmarker();

    state.live = true;
    state.mode = "NO HAND";
    stage.classList.add("live");
    overlay.classList.add("hidden");
    requestAnimationFrame(resizeCanvas);
    toast("Show your hand and raise your index finger to draw");
  } catch (err) {
    console.error("Camera start failed:", err);
    btn.disabled = false;
    btn.textContent = "Try again";
    note.textContent = CAMERA_ERRORS[err.name] || `Couldn't start: ${err.message || err}`;
  }
}

function useMouse() {
  overlay.classList.add("hidden");
  setMode("MOUSE");
  toast("Draw on the canvas, then press Read it");
}

// ------------------------------------------------------------------ main loop
function loop() {
  if (state.live && landmarker && video.readyState >= 2 && video.currentTime !== state.lastVideoTime) {
    state.lastVideoTime = video.currentTime;
    const result = landmarker.detectForVideo(video, performance.now());
    processHand(pickHand(result));
  }
  if (state.dirty) render();
  requestAnimationFrame(loop);
}

// ------------------------------------------------------------------ input
// Mouse / touch / pen drawing (works with or without the camera)
function pointerPos(e) {
  const r = ink.getBoundingClientRect();
  return { x: (e.clientX - r.left) / r.width, y: (e.clientY - r.top) / r.height };
}
ink.addEventListener("pointerdown", (e) => {
  if (!overlay.classList.contains("hidden")) return;
  ink.setPointerCapture(e.pointerId);
  endStroke();
  addPoint(pointerPos(e));
  ink.dataset.down = "1";
});
ink.addEventListener("pointermove", (e) => {
  if (ink.dataset.down) {
    for (const ev of e.getCoalescedEvents?.() || [e]) addPoint(pointerPos(ev));
  }
});
["pointerup", "pointercancel"].forEach((t) => ink.addEventListener(t, () => {
  delete ink.dataset.down;
  endStroke();
}));

function setRecognition(value) {
  state.recognition = value;
  segmented.dataset.value = value;
  segmented.querySelectorAll("button").forEach((b) => {
    const on = b.dataset.value === value;
    b.classList.toggle("active", on);
    b.setAttribute("aria-selected", on);
  });
}
segmented.querySelectorAll("button").forEach((b) =>
  b.addEventListener("click", () => !b.disabled && setRecognition(b.dataset.value)));

$("start-camera").addEventListener("click", startCamera);
$("use-mouse").addEventListener("click", useMouse);
btnPredict.addEventListener("click", predict);
$("btn-clear").addEventListener("click", clearAll);
$("btn-undo").addEventListener("click", undo);

window.addEventListener("keydown", (e) => {
  if (e.target.matches("input, textarea")) return;
  const k = e.key.toLowerCase();
  if (k === "p") predict();
  else if (k === "c" && !e.ctrlKey && !e.metaKey) clearAll();
  else if (k === "z") { e.preventDefault(); undo(); }
});

new ResizeObserver(resizeCanvas).observe(stage);

// reveal "How it works" blocks as they scroll into view
const io = new IntersectionObserver((entries) => entries.forEach((en) => {
  if (en.isIntersecting) { en.target.classList.add("in"); io.unobserve(en.target); }
}), { threshold: 0.15 });
document.querySelectorAll(".how > *, .steps li, .gestures > *").forEach((el, i) => {
  el.classList.add("reveal-up");
  el.style.transitionDelay = `${(i % 4) * 60}ms`;
  io.observe(el);
});

// ------------------------------------------------------------------ boot
setMode("MOUSE");
renderRecent();
checkServer();
loadLandmarker().catch((e) => console.warn("Hand model preload failed:", e)); // warm up in background
requestAnimationFrame(loop);
