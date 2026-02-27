"""
🐛 Pest Detection FastAPI Server
Classes: Aphid | Fruit Fly | Scale Insect

Detection Logic:
  IMAGE:
    Step 1 → YOLO runs first
    Step 2 → If YOLO finds pests → trust YOLO result (no CLIP needed)
    Step 3 → If YOLO finds 0    → CLIP fallback check
                                   CLIP >= 80% confident → "unidentified_pest"
                                   CLIP <  80%           → truly "no_pest"

  VIDEO:
    CLIP used only as a skip-gate (every N frames) to save processing time
    YOLO does the actual detection on frames CLIP approves
    Same YOLO-primary / CLIP-fallback logic applied per frame

  FILTER (both image + video):
    - conf  < 0.30        → rejected (too uncertain)
    - area  < 0.1% image  → rejected (noise / tiny speck)
    - area  > 40%  image  → rejected (whole-image false positive)
"""

import os
import cv2
import uuid
import shutil
import tempfile
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


# ───────────────── CONFIG ─────────────────
MODEL_PATH            = os.environ.get("YOLO_MODEL_PATH",       "best.pt")
CLIP_MODEL_NAME       = os.environ.get("CLIP_MODEL",            "ViT-B/32")
CLASS_NAMES           = ["aphid", "fruit_fly", "scale_insect"]
CONF_THRESH           = float(os.environ.get("CONF_THRESH",           "0.25"))
IOU_THRESH            = float(os.environ.get("IOU_THRESH",            "0.5"))
PEST_PROB_THRESH      = float(os.environ.get("PEST_PROB_THRESH",      "0.5"))   # video skip-gate
CLIP_FALLBACK_THRESH  = float(os.environ.get("CLIP_FALLBACK_THRESH",  "0.80"))  # strict fallback
CLIP_CHECK_EVERY      = int(os.environ.get("CLIP_CHECK_EVERY",        "30"))
MAX_VIDEO_FRAMES      = int(os.environ.get("MAX_VIDEO_FRAMES",        "1000"))

# Detection filters
MIN_CONF              = float(os.environ.get("MIN_CONF",       "0.30"))   # reject below this
MIN_AREA_RATIO        = float(os.environ.get("MIN_AREA_RATIO", "0.001"))  # reject < 0.1% of image
MAX_AREA_RATIO        = float(os.environ.get("MAX_AREA_RATIO", "0.40"))   # reject > 40% of image

CLIP_PROMPTS = [
    "a photo of a pest insect or bug on a plant",
    "a photo with no pest insect",
]

TEMP_DIR = Path(tempfile.gettempdir()) / "pest_api"
TEMP_DIR.mkdir(exist_ok=True)


# ───────────────── MODEL STATE ─────────────────
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
        state.yolo_model = YOLO(MODEL_PATH)
        state.clip_model, state.clip_preprocess = clip.load(
            CLIP_MODEL_NAME, device=state.device
        )
        state.loaded = True
        print("✅ Models loaded")
    except Exception as exc:
        state.error = str(exc)
        print(f"❌ Model load error: {exc}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_models()
    yield
    shutil.rmtree(TEMP_DIR, ignore_errors=True)


# ───────────────── APP ─────────────────
app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def require_models():
    if not state.loaded:
        raise HTTPException(503, detail="Models not ready")


# ───────────────── HELPERS ─────────────────

def clip_pest_prob(pil_img: Image.Image) -> float:
    """Returns pest probability 0.0–1.0."""
    tensor = state.clip_preprocess(pil_img).unsqueeze(0).to(state.device)
    tokens = clip.tokenize(CLIP_PROMPTS).to(state.device)
    with torch.no_grad():
        logits, _ = state.clip_model(tensor, tokens)
        probs = logits.softmax(dim=-1)
    return float(probs[0][0].item())


def yolo_on_bgr(bgr: np.ndarray):
    results   = state.yolo_model(bgr, conf=CONF_THRESH, iou=IOU_THRESH, verbose=False)
    annotated = results[0].plot(line_width=1, font_size=4)
    return results[0], annotated


def parse_detections(result, frame_idx: int = None) -> list:
    """
    Parse YOLO boxes with 3 filters:
      1. conf  < MIN_CONF       → skip  (too uncertain)
      2. area  < MIN_AREA_RATIO → skip  (noise / speck)
      3. area  > MAX_AREA_RATIO → skip  (whole-image false positive)
    """
    out = []
    if result.boxes is None:
        return out

    img_h, img_w = result.orig_shape[:2]
    img_area     = float(img_w * img_h)

    for box in result.boxes:
        cls_id = int(box.cls[0].item())
        conf   = float(box.conf[0].item())
        xyxy   = box.xyxy[0].tolist()

        # ── Filter 1: confidence too low ──────────────────────────────────
        if conf < MIN_CONF:
            continue

        # ── Filter 2 & 3: box area check ──────────────────────────────────
        box_w      = xyxy[2] - xyxy[0]
        box_h      = xyxy[3] - xyxy[1]
        area_ratio = (box_w * box_h) / img_area

        if area_ratio < MIN_AREA_RATIO:   # too small — noise
            continue
        if area_ratio > MAX_AREA_RATIO:   # too large — false positive
            continue

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
    path = TEMP_DIR / f"{token}.jpg"
    cv2.imwrite(str(path), bgr)
    return f"/result-image/{token}"


def clip_fallback_detection(pil_img: Image.Image, w: float, h: float,
                             frame_idx: int = None) -> tuple[list, float]:
    """
    CLIP fallback — only fires when YOLO found 0 detections.
    Returns unidentified_pest only if prob >= CLIP_FALLBACK_THRESH (80%).
    """
    prob = clip_pest_prob(pil_img)
    if prob >= CLIP_FALLBACK_THRESH:
        det = {
            "class":         "unidentified_pest",
            "confidence":    round(prob, 4),
            "box":           {"x1": 0.0, "y1": 0.0, "x2": w, "y2": h},
            "clip_fallback": True,
        }
        if frame_idx is not None:
            det["frame"] = frame_idx
        return [det], prob
    return [], prob


# ───────────────── IMAGE ENDPOINT ─────────────────
@app.post("/predict-image")
async def predict_image(file: UploadFile = File(...)):
    require_models()
    raw     = await file.read()
    np_arr  = np.frombuffer(raw, np.uint8)
    bgr     = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    h, w    = bgr.shape[:2]
    pil_img = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    token   = str(uuid.uuid4())

    # ── Step 1: YOLO ──────────────────────────────────────────────────────
    result, ann_bgr = yolo_on_bgr(bgr)
    detections      = parse_detections(result)        # filtered detections
    ann_url         = save_temp_image(ann_bgr, token)

    clip_fallback_triggered = False
    clip_prob               = None

    # ── Step 2: CLIP fallback ONLY if YOLO found nothing ─────────────────
    if len(detections) == 0:
        detections, clip_prob = clip_fallback_detection(pil_img, float(w), float(h))
        if detections:
            clip_fallback_triggered = True

    counts = dict(Counter(d["class"] for d in detections))
    total  = len(detections)

    response = {
        "status":                  "pest_detected" if total else "no_pest",
        "detections":              detections,
        "class_wise_count":        counts,
        "total_count":             total,
        "annotated_image":         ann_url,
        "clip_fallback_triggered": clip_fallback_triggered,
    }

    if clip_prob is not None:
        response["clip_info"] = {
            "is_pest":    clip_fallback_triggered,
            "confidence": round(clip_prob, 4),
            "best_match": "unidentified_pest" if clip_fallback_triggered else "none",
        }

    return response


# ───────────────── VIDEO ENDPOINT ─────────────────
@app.post("/predict-video")
async def predict_video(
    background_tasks: BackgroundTasks,
    file: UploadFile   = File(...),
    clip_every: int    = Query(CLIP_CHECK_EVERY),
    max_frames: int    = Query(MAX_VIDEO_FRAMES),
    return_video: bool = Query(True),
):
    require_models()

    raw   = await file.read()
    token = str(uuid.uuid4())
    in_p  = TEMP_DIR / f"{token}_in.mp4"
    out_p = TEMP_DIR / f"{token}_out.mp4"
    in_p.write_bytes(raw)

    cap          = cv2.VideoCapture(str(in_p))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps          = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    limit        = max_frames if max_frames > 0 else total_frames

    writer = None
    if return_video:
        writer = cv2.VideoWriter(
            str(out_p),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height),
        )

    frame_idx      = 0
    all_detections = []

    # Peak-frame tracking
    max_pests_in_frame = 0
    best_frame_counts  = Counter()
    best_frame_dets    = []

    last_clip_prob       = 0.0
    clip_fallback_frames = 0

    while cap.isOpened() and frame_idx < limit:
        ok, frame = cap.read()
        if not ok:
            break

        pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        # ── CLIP skip-gate every N frames ─────────────────────────────────
        if frame_idx % clip_every == 0:
            last_clip_prob = clip_pest_prob(pil)

        is_pest_frame = last_clip_prob >= PEST_PROB_THRESH

        if is_pest_frame:
            result, ann_frame = yolo_on_bgr(frame)
            dets = parse_detections(result, frame_idx)   # filtered

            # ── CLIP fallback per frame ───────────────────────────────────
            if len(dets) == 0:
                dets, _ = clip_fallback_detection(
                    pil, float(width), float(height), frame_idx
                )
                if dets:
                    clip_fallback_frames += 1
                    cv2.putText(
                        ann_frame,
                        f"Unidentified Pest (CLIP {dets[0]['confidence']:.0%})",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 165, 255),
                        2,
                        cv2.LINE_AA,
                    )

            current_count = len(dets)

            # Track peak frame
            if current_count > max_pests_in_frame:
                max_pests_in_frame = current_count
                best_frame_counts  = Counter(d["class"] for d in dets)
                best_frame_dets    = dets

            all_detections.extend(dets)

        else:
            ann_frame = frame

        if writer:
            writer.write(ann_frame)

        frame_idx += 1

    cap.release()
    if writer:
        writer.release()

    background_tasks.add_task(lambda: in_p.unlink(missing_ok=True))

    total = max_pests_in_frame

    return {
        "status":               "pest_detected" if total else "no_pest",
        "total_frames":         total_frames,
        "processed_frames":     frame_idx,
        "fps":                  round(fps, 2),
        "resolution":           f"{width}x{height}",
        "total_count":          total,
        "class_wise_count":     dict(best_frame_counts),
        "detections":           best_frame_dets,
        "clip_fallback_frames": clip_fallback_frames,
        "annotated_video":      f"/result-video/{token}" if return_video else None,
    }


# ───────────────── SERVE RESULTS ─────────────────
@app.get("/result-image/{token}")
def serve_image(token: str):
    p = TEMP_DIR / f"{token}.jpg"
    return FileResponse(str(p), media_type="image/jpeg")


@app.get("/result-video/{token}")
def serve_video(token: str):
    p = TEMP_DIR / f"{token}_out.mp4"
    return FileResponse(
        str(p),
        media_type="video/mp4",
        headers={"ngrok-skip-browser-warning": "true"},
    )


# ───────────────── ENTRY ─────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000)
