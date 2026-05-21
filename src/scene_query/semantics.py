"""YOLO semantics backend: per-frame object detection."""
from __future__ import annotations

from pathlib import Path

import numpy as np

_REPO_ROOT    = Path(__file__).parent.parent.parent
_YOLO_WEIGHTS = _REPO_ROOT / "checkpoints" / "yolo_office" / "best.pt"

# COCO classes relevant to homes, schools, and offices.
# Vehicles, animals, outdoor furniture, and sports gear are excluded.
_INDOOR_CLASSES = {
    "person",
    # furniture & fixtures
    "chair", "couch", "bed", "dining table", "toilet", "potted plant",
    # screens & peripherals
    "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
    # kitchen / appliances
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl",
    "microwave", "oven", "toaster", "sink", "refrigerator",
    # office / school
    "book", "scissors", "clock", "vase",
    # bags & personal items
    "backpack", "handbag", "suitcase", "umbrella", "tie",
    # home
    "teddy bear", "hair drier", "toothbrush",
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
