"""YOLO semantics backend (primary) + Grounding DINO + SAM2 (open-vocabulary fallback)."""
from __future__ import annotations

from pathlib import Path

import numpy as np

_REPO_ROOT    = Path(__file__).parent.parent.parent
_YOLO_WEIGHTS = _REPO_ROOT / "checkpoints" / "yolo_office" / "best.pt"

# COCO classes relevant to offices, workspaces, and classrooms.
# Appliances, food, outdoor items, and niche personal items are excluded.
_INDOOR_CLASSES = {
    "person",
    # seating & furniture
    "chair", "couch", "dining table", "potted plant",
    # screens & peripherals
    "monitor", "laptop", "mouse", "keyboard", "remote", "cell phone",
    # common desk / room items
    "bottle", "cup", "bowl", "book", "scissors", "clock", "vase", "backpack",
}


def load_yolo_model(weights_path: Path | None = None, device: str = "cuda"):
    """Load YOLOv8 detection model from local weights or an ultralytics model name."""
    from ultralytics import YOLO
    path = Path(weights_path) if weights_path else _YOLO_WEIGHTS
    # If it's a plain filename with no directory (e.g. "yolov8s.pt"), let
    # ultralytics resolve and auto-download it rather than treating it as a
    # missing local file.
    if not path.exists() and path.parent == Path("."):
        return YOLO(str(path)).to(device)
    if not path.exists():
        raise FileNotFoundError(
            f"YOLO weights not found: {path}\n"
            "Run python setup.py to download model weights."
        )
    return YOLO(str(path)).to(device)


def segment_frames_yolo(
    frames: list[np.ndarray],
    model,
    device: str = "cuda",
    conf_threshold: float = 0.25,
) -> list[list[dict]]:
    """Run YOLOv8 detection on every frame; return per-frame detection lists.

    Each detection dict has keys: label, mask (bool H×W), confidence,
    instance_id, bbox ([x1, y1, x2, y2] in pixels).
    """
    all_results: list[list[dict]] = []

    for frame_idx, frame in enumerate(frames):
        H, W = frame.shape[:2]
        results = model.predict(frame, conf=conf_threshold, device=device, verbose=False)
        frame_dets: list[dict] = []

        seg_masks = results[0].masks  # None for detection-only models
        if results and len(results[0].boxes):
            for i, box in enumerate(results[0].boxes):
                label = model.names[int(box.cls[0])].lower()
                if label == "tv":
                    label = "monitor"
                # Skip classes outside the indoor/office/school set when the
                # model has the full COCO vocabulary (80 classes).
                if len(model.names) >= 80 and label not in _INDOOR_CLASSES:
                    continue

                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                x1, y1 = max(0, int(x1)), max(0, int(y1))
                x2, y2 = min(W - 1, int(x2)), min(H - 1, int(y2))

                if seg_masks is not None and i < len(seg_masks.xy):
                    import cv2
                    poly = seg_masks.xy[i].astype(np.int32)
                    mask = np.zeros((H, W), dtype=bool)
                    if len(poly) >= 3:
                        cv2.fillPoly(mask.view(np.uint8), [poly], 1)
                else:
                    mask = np.zeros((H, W), dtype=bool)
                    mask[y1:y2 + 1, x1:x2 + 1] = True

                frame_dets.append({
                    "label":       label,
                    "mask":        mask,
                    "confidence":  float(box.conf[0]),
                    "instance_id": i,
                    "bbox":        [x1, y1, x2, y2],
                })

        all_results.append(frame_dets)

        if frame_idx == 0 or (frame_idx + 1) % 5 == 0:
            total = sum(len(r) for r in all_results)
            print(f"  Frame {frame_idx + 1}/{len(frames)}: "
                  f"{len(frame_dets)} dets  ({total} total so far)")

    return all_results


# ---------------------------------------------------------------------------
# Grounding DINO + SAM 2 — open-vocabulary fallback
# ---------------------------------------------------------------------------

_GDINO_WEIGHTS = _REPO_ROOT / "checkpoints" / "grounding_dino" / "groundingdino_swint_ogc.pth"
_GDINO_CONFIG  = _REPO_ROOT / "checkpoints" / "grounding_dino" / "GroundingDINO_SwinT_OGC.py"
_SAM2_WEIGHTS  = _REPO_ROOT / "checkpoints" / "sam2" / "sam2.1_hiera_large.pt"
_SAM2_CONFIG   = "configs/sam2.1/sam2.1_hiera_l.yaml"


def load_gsam2_models(device: str = "cuda") -> tuple:
    """Load Grounding DINO + SAM 2.1; return (grounding_model, sam_model)."""
    try:
        from groundingdino.util.inference import load_model as _load_gdino
    except ImportError:
        raise ImportError("groundingdino not installed. See Grounded-SAM-2 repo.")
    try:
        from sam2.build_sam import build_sam2
    except ImportError:
        raise ImportError("sam2 not installed. See Grounded-SAM-2 repo.")

    for path in [_GDINO_WEIGHTS, _GDINO_CONFIG, _SAM2_WEIGHTS]:
        if not path.exists():
            raise FileNotFoundError(
                f"GSAM2 weights not found: {path}\n"
                "Run python setup.py to download model weights."
            )

    grounding_model = _load_gdino(str(_GDINO_CONFIG), str(_GDINO_WEIGHTS), device=device)
    grounding_model = grounding_model.to(device).eval()
    sam_model = build_sam2(_SAM2_CONFIG, str(_SAM2_WEIGHTS), device=device)
    return grounding_model, sam_model


def segment_frames_gsam2(
    frames: list[np.ndarray],
    labels: list[str],
    grounding_model,
    sam_model,
    device: str = "cuda",
    box_threshold: float = 0.3,
    text_threshold: float = 0.25,
) -> list[list[dict]]:
    """Run Grounding DINO + SAM2 on every frame; return per-frame detection lists."""
    import torch
    from PIL import Image
    import groundingdino.datasets.transforms as T
    from groundingdino.util.inference import predict as gdino_predict
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    caption = " . ".join(labels) + " ."
    gdino_transform = T.Compose([
        T.RandomResize([800], max_size=1333),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    sam_predictor = SAM2ImagePredictor(sam_model)
    all_results: list[list[dict]] = []

    for frame_idx, frame in enumerate(frames):
        H, W = frame.shape[:2]
        pil = Image.fromarray(frame)
        img_t, _ = gdino_transform(pil, None)
        with torch.inference_mode():
            boxes_norm, logits, phrases = gdino_predict(
                model=grounding_model,
                image=img_t,
                caption=caption,
                box_threshold=box_threshold,
                text_threshold=text_threshold,
            )

        frame_dets: list[dict] = []
        if len(boxes_norm) > 0:
            bn = boxes_norm.cpu().numpy()
            cx, cy, bw, bh = bn[:, 0], bn[:, 1], bn[:, 2], bn[:, 3]
            boxes_xyxy = np.stack([
                np.clip((cx - bw / 2) * W, 0, W - 1),
                np.clip((cy - bh / 2) * H, 0, H - 1),
                np.clip((cx + bw / 2) * W, 0, W - 1),
                np.clip((cy + bh / 2) * H, 0, H - 1),
            ], axis=-1)
            confs = logits.cpu().numpy()
            matched_labels = [_match_label(ph, labels) for ph in phrases]

            sam_predictor.set_image(frame)
            for i, (box, conf, label) in enumerate(zip(boxes_xyxy, confs, matched_labels)):
                with torch.inference_mode():
                    masks, _, _ = sam_predictor.predict(
                        point_coords=None, point_labels=None,
                        box=box, multimask_output=False,
                    )
                frame_dets.append({
                    "label":       label,
                    "mask":        masks[0].astype(bool),
                    "confidence":  float(conf),
                    "instance_id": i,
                    "bbox":        [int(box[0]), int(box[1]), int(box[2]), int(box[3])],
                })

        all_results.append(frame_dets)

        if frame_idx == 0 or (frame_idx + 1) % 5 == 0:
            total = sum(len(r) for r in all_results)
            print(f"  Frame {frame_idx + 1}/{len(frames)}: "
                  f"{len(frame_dets)} dets  ({total} total so far)")

    return all_results


def _match_label(phrase: str, labels: list[str]) -> str:
    ph = phrase.lower().strip()
    for lbl in labels:
        if lbl in ph or ph in lbl:
            return lbl
    return ph
