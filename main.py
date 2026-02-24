"""
🐛 Pest Detection FastAPI Server
Classes: Aphid | Fruit Fly | Scale Insect

Endpoints (matching the HTML tester exactly):
  POST /predict-image   → CLIP filter + YOLO → JSON result
  POST /predict-video   → frame-by-frame CLIP + YOLO → JSON result
  GET  /result-image/{token}  → serve annotated image
  GET  /result-video/{token}  → serve annotated video
  GET  /health          → health check
"""

import os
import cv2
import uuid
import shutil
import tempfile
import re
import numpy as np
import torch
import clip
from PIL import Image
from pathlib import Path
from collections import Counter
from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile, File, HTTPException, Query, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from ultralytics import YOLO


# ─────────────────────────────────────────────
# CONFIG  (override with env variables)
# ─────────────────────────────────────────────
MODEL_PATH       = os.environ.get("YOLO_MODEL_PATH", "best.pt")
CLIP_MODEL_NAME  = os.environ.get("CLIP_MODEL",      "ViT-B/32")
CLASS_NAMES      = ["aphid", "fruit_fly", "scale_insect"]
CONF_THRESH      = float(os.environ.get("CONF_THRESH",      "0.25"))
IOU_THRESH       = float(os.environ.get("IOU_THRESH",       "0.5"))
PEST_PROB_THRESH = float(os.environ.get("PEST_PROB_THRESH", "0.5"))
CLIP_CHECK_EVERY = int(os.environ.get("CLIP_CHECK_EVERY",   "30"))
MAX_VIDEO_FRAMES = int(os.environ.get("MAX_VIDEO_FRAMES",   "1000"))

CLIP_PROMPTS = [
    "a photo of a pest insect or bug on a plant",
    "a photo with no pest insect",
]

TEMP_DIR = Path(tempfile.gettempdir()) / "pest_api"
TEMP_DIR.mkdir(exist_ok=True)


# ─────────────────────────────────────────────
# GLOBAL MODEL STATE
# ─────────────────────────────────────────────
class ModelState:
    yolo_model      = None
    clip_model      = None
    clip_preprocess = None
    device          = "cpu"
    loaded          = False
    error           = None

state = ModelState()


def load_models():
    try:
        state.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"⏳ Loading YOLO from: {MODEL_PATH}")
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"YOLO weights not found: {MODEL_PATH}")
        state.yolo_model = YOLO(MODEL_PATH)
        print("   ✅ YOLO ready")

        print(f"⏳ Loading CLIP ({CLIP_MODEL_NAME}) ...")
        state.clip_model, state.clip_preprocess = clip.load(
            CLIP_MODEL_NAME, device=state.device
        )
        print("   ✅ CLIP ready")

        state.loaded = True
        print("✅ All models loaded — server ready!")
    except Exception as exc:
        state.error = str(exc)
        print(f"❌ Model load error: {exc}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_models()
    yield
    shutil.rmtree(TEMP_DIR, ignore_errors=True)


# ─────────────────────────────────────────────
# APP
# ─────────────────────────────────────────────
app = FastAPI(
    title="🐛 Pest Detection API",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
def require_models():
    if not state.loaded:
        raise HTTPException(
            503, detail=f"Models not ready. Error: {state.error or 'still loading'}"
        )


def clip_info_dict(pil_img: Image.Image) -> dict:
    """
    Run CLIP and return dict matching HTML tester format:
      { "is_pest": bool, "best_match": str, "confidence": float }
    best_match and confidence updated after YOLO if possible.
    """
    tensor = state.clip_preprocess(pil_img).unsqueeze(0).to(state.device)
    tokens = clip.tokenize(CLIP_PROMPTS).to(state.device)
    with torch.no_grad():
        logits, _ = state.clip_model(tensor, tokens)
        probs     = logits.softmax(dim=-1)[0]

    pest_prob = float(probs[0].item())
    is_pest   = pest_prob >= PEST_PROB_THRESH
    return {
        "is_pest":    is_pest,
        "best_match": "unknown",      # updated after YOLO
        "confidence": round(pest_prob, 4),
    }


def clip_pest_prob(pil_img: Image.Image) -> float:
    """Lightweight version — returns pest probability only (for video frames)."""
    tensor = state.clip_preprocess(pil_img).unsqueeze(0).to(state.device)
    tokens = clip.tokenize(CLIP_PROMPTS).to(state.device)
    with torch.no_grad():
        logits, _ = state.clip_model(tensor, tokens)
        probs     = logits.softmax(dim=-1)
    return float(probs[0][0].item())


def yolo_on_bgr(bgr: np.ndarray):
    """Run YOLO on BGR numpy array. Returns (result_obj, annotated_bgr)."""
    results   = state.yolo_model(bgr, conf=CONF_THRESH, iou=IOU_THRESH, verbose=False)
    annotated = results[0].plot(line_width=1, font_size=4)
    return results[0], annotated


def parse_detections(result, frame_idx: int = None) -> list:
    """
    Convert YOLO result to list of dicts expected by HTML tester:
      { "class": str, "confidence": float, "box": {x1,y1,x2,y2}, "frame"?: int }
    """
    out = []
    if result.boxes is None:
        return out
    for box in result.boxes:
        cls_id = int(box.cls[0].item())
        conf   = float(box.conf[0].item())
        xyxy   = box.xyxy[0].tolist()
        det = {
            "class":      CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else "unknown",
            "confidence": round(conf, 4),
            "box": {
                "x1": round(xyxy[0], 1),
                "y1": round(xyxy[1], 1),
                "x2": round(xyxy[2], 1),
                "y2": round(xyxy[3], 1),
            },
        }
        if frame_idx is not None:
            det["frame"] = frame_idx
        out.append(det)
    return out


def save_temp_image(bgr: np.ndarray, token: str) -> str:
    """Save BGR image to temp dir, return serving URL."""
    path = TEMP_DIR / f"{token}.jpg"
    cv2.imwrite(str(path), bgr)
    return f"/result-image/{token}"


# ─────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────

@app.get("/health")
def health():
    return {
        "status":  "ok" if state.loaded else "loading",
        "device":  state.device,
        "loaded":  state.loaded,
        "error":   state.error,
        "classes": CLASS_NAMES,
    }


# ══════════════════════════════════════════════════════════════════════════════
# POST /predict-image
#
# HTML tester reads:
#   data.status              → "pest_detected" | "no_pest" | "unknown_pest"
#   data.class_wise_count    → { "fruit_fly": 13 }
#   data.detections          → [ { class, confidence, box:{x1,y1,x2,y2} } ]
#   data.clip_info           → { is_pest, best_match, confidence }
#   data.total_count         → int
#   data.message             → str  (shown in "no_pest" / "unknown_pest" card)
# ══════════════════════════════════════════════════════════════════════════════
@app.post("/predict-image")
async def predict_image(file: UploadFile = File(...)):
    require_models()

    raw = await file.read()
    if not raw:
        raise HTTPException(400, "Empty file")

    np_arr = np.frombuffer(raw, np.uint8)
    bgr    = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if bgr is None:
        raise HTTPException(400, "Cannot decode image — make sure it is jpg/png/webp")

    pil_img = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    token   = str(uuid.uuid4())

    # ── Step 1: CLIP ──────────────────────────────────────────────────────────
    cinfo = clip_info_dict(pil_img)

    if not cinfo["is_pest"]:
        # CLIP says not a pest image → return immediately
        save_temp_image(bgr, token)
        return {
            "status":           "no_pest",
            "message":          "CLIP filter: this does not appear to be a pest image.",
            "clip_info":        cinfo,
            "detections":       [],
            "class_wise_count": {},
            "total_count":      0,
            "annotated_image":  f"/result-image/{token}",
        }

    # ── Step 2: YOLO ──────────────────────────────────────────────────────────
    result, annotated_bgr = yolo_on_bgr(bgr)
    detections = parse_detections(result)
    ann_url    = save_temp_image(annotated_bgr, token)

    counts = dict(Counter(d["class"] for d in detections))
    total  = len(detections)

    # Update CLIP best_match to dominant YOLO class
    if counts:
        cinfo["best_match"] = max(counts, key=counts.get)

    # ── Step 3: Decide status (mirrors notebook logic) ─────────────────────
    # Notebook: if max_conf < 0.5 → "unknown pest"
    high_conf = [d for d in detections if d["confidence"] >= 0.5]

    if total == 0:
        status  = "no_pest"
        message = "CLIP says pest image, but YOLO found no objects."
    elif not high_conf:
        # Detections exist but all below 0.5 — matches notebook "unknown pest"
        status  = "unknown_pest"
        message = (
            "Pest detected, but confidence is too low to identify the species. "
            "It may not be in our database."
        )
    else:
        status  = "pest_detected"
        parts   = [f"{v} {k}" for k, v in counts.items()]
        message = f"Detected: {', '.join(parts)}"

    return {
        "status":           status,
        "message":          message,
        "clip_info":        cinfo,
        "detections":       detections,
        "class_wise_count": counts,
        "total_count":      total,
        "annotated_image":  ann_url,
    }


# ══════════════════════════════════════════════════════════════════════════════
# POST /predict-video
#
# HTML tester reads same keys as image, plus:
#   data.total_frames, data.processed_frames, data.fps, data.resolution
#   data.detections[i].frame   (frame number)
#   data.annotated_video       → "/result-video/<token>"
# ══════════════════════════════════════════════════════════════════════════════
@app.post("/predict-video")
async def predict_video(
    background_tasks: BackgroundTasks,
    file:        UploadFile = File(...),
    clip_every:  int  = Query(CLIP_CHECK_EVERY, description="Run CLIP every N frames"),
    max_frames:  int  = Query(MAX_VIDEO_FRAMES, description="0 = process all frames"),
    return_video: bool = Query(True, description="Return annotated video"),
):
    require_models()

    raw = await file.read()
    if not raw:
        raise HTTPException(400, "Empty file")

    token  = str(uuid.uuid4())
    suffix = Path(file.filename or "video.mp4").suffix or ".mp4"
    in_p   = TEMP_DIR / f"{token}_in{suffix}"
    out_p  = TEMP_DIR / f"{token}_out.mp4"
    in_p.write_bytes(raw)

    cap = cv2.VideoCapture(str(in_p))
    if not cap.isOpened():
        raise HTTPException(400, "Cannot open video file")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps    = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    limit  = max_frames if max_frames > 0 else total_frames

    writer = None
    if return_video:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(out_p), fourcc, fps, (width, height))

    all_detections = []
    total_counts   = Counter()
    frame_idx      = 0
    is_pest_frame  = True

    while cap.isOpened() and frame_idx < limit:
        ok, frame = cap.read()
        if not ok:
            break

        # CLIP every N frames
        if frame_idx % clip_every == 0:
            pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            is_pest_frame = clip_pest_prob(pil) >= PEST_PROB_THRESH

        if is_pest_frame:
            result, ann_frame = yolo_on_bgr(frame)
            dets = parse_detections(result, frame_idx=frame_idx)
            for d in dets:
                total_counts[d["class"]] += 1
            all_detections.extend(dets)
        else:
            ann_frame = frame.copy()
            cv2.putText(ann_frame, "Non-pest frame",
                        (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

        if writer:
            writer.write(ann_frame)

        frame_idx += 1

    cap.release()
    if writer:
        writer.release()

    background_tasks.add_task(lambda: in_p.unlink(missing_ok=True))

    total = sum(total_counts.values())

    if total == 0:
        status  = "no_pest"
        message = "No pests detected in the video."
    else:
        parts   = [f"{v} {k}" for k, v in total_counts.most_common()]
        status  = "pest_detected"
        message = f"Detected {total} pest instances across {frame_idx} frames: {', '.join(parts)}"

    return {
        "status":           status,
        "message":          message,
        "total_frames":     total_frames,
        "processed_frames": frame_idx,
        "fps":              round(fps, 2),
        "resolution":       f"{width}x{height}",
        "total_count":      total,
        "class_wise_count": dict(total_counts),
        "detections":       all_detections,      # includes "frame" key per detection
        "annotated_video":  f"/result-video/{token}" if return_video else None,
    }


# ─────────────────────────────────────────────
# SERVE ANNOTATED RESULTS
# ─────────────────────────────────────────────
@app.get("/result-image/{token}")
def serve_image(token: str):
    if not re.fullmatch(r"[0-9a-f\-]{36}", token):
        raise HTTPException(400, "Invalid token")
    p = TEMP_DIR / f"{token}.jpg"
    if not p.exists():
        raise HTTPException(404, "Image not found or expired")
    return FileResponse(str(p), media_type="image/jpeg")


@app.get("/result-video/{token}")
def serve_video(token: str):
    if not re.fullmatch(r"[0-9a-f\-]{36}", token):
        raise HTTPException(400, "Invalid token")
    p = TEMP_DIR / f"{token}_out.mp4"
    if not p.exists():
        raise HTTPException(404, "Video not found or expired")
    return FileResponse(str(p), media_type="video/mp4", filename="pest_result.mp4")


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)