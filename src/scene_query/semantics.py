"""Semantics backends: YOLO (default) or Grounded-SAM-2.

YOLO backend is the primary path — uses the user's trained YOLOv8 detection model
(10 office classes: bottle, chair, keyboard, monitor, mouse, mug, notebook, pen,
printer, stapler). No label input required; the model auto-detects all classes.

Grounded-SAM-2 is kept as a fallback for open-vocabulary labels.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image

_REPO_ROOT = Path(__file__).parent.parent.parent
_YOLO_WEIGHTS = _REPO_ROOT / "checkpoints" / "yolo_office" / "best.pt"

# --- YOLO backend ---

def load_yolo_model(weights_path: Path | None = None, device: str = "cuda"):
    """Load YOLOv8 detection model from local weights."""
    from ultralytics import YOLO
    path = Path(weights_path) if weights_path else _YOLO_WEIGHTS
    if not path.exists():
        raise FileNotFoundError(
            f"YOLO weights not found: {path}\n"
            "Copy best.pt from office-item-classifier into checkpoints/yolo_office/best.pt"
        )
    model = YOLO(str(path))
    model.to(device)
    return model


def segment_frames_yolo(
    frames: list[np.ndarray],
    model,
    device: str = "cuda",
    conf_threshold: float = 0.25,
) -> list[list[dict]]:
    """Run YOLOv8 detection on every frame; return per-frame detection lists.

    Each detection has a rectangular mask derived from its bounding box.
    Returns the same schema as the Grounded-SAM-2 backend so downstream code
    is unchanged.
    """
    all_results: list[list[dict]] = []

    for frame_idx, frame in enumerate(frames):
        H, W = frame.shape[:2]
        results = model.predict(frame, conf=conf_threshold, device=device, verbose=False)
        frame_dets: list[dict] = []

        if results and len(results[0].boxes):
            boxes = results[0].boxes
            for i, box in enumerate(boxes):
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                x1, y1 = max(0, int(x1)), max(0, int(y1))
                x2, y2 = min(W - 1, int(x2)), min(H - 1, int(y2))

                mask = np.zeros((H, W), dtype=bool)
                mask[y1:y2 + 1, x1:x2 + 1] = True

                label = model.names[int(box.cls[0])].lower()
                conf  = float(box.conf[0])

                frame_dets.append({
                    "label":       label,
                    "mask":        mask,
                    "confidence":  conf,
                    "instance_id": i,
                })

        all_results.append(frame_dets)

        if frame_idx == 0 or (frame_idx + 1) % 5 == 0:
            total = sum(len(r) for r in all_results)
            print(f"  Frame {frame_idx + 1}/{len(frames)}: "
                  f"{len(frame_dets)} dets  ({total} total so far)")

    return all_results


# --- Grounded-SAM-2 backend (fallback for open-vocabulary labels) ---

_GDINO_WEIGHTS = _REPO_ROOT / "checkpoints" / "grounding_dino" / "groundingdino_swint_ogc.pth"
_GDINO_CONFIG  = _REPO_ROOT / "checkpoints" / "grounding_dino" / "GroundingDINO_SwinT_OGC.py"
_SAM2_WEIGHTS  = _REPO_ROOT / "checkpoints" / "sam2" / "sam2.1_hiera_large.pt"
_SAM2_CONFIG   = "configs/sam2.1/sam2.1_hiera_l.yaml"


def load_models(
    device: str = "cuda",
    weights_gdino: Path | None = None,
    weights_sam2: Path | None = None,
) -> tuple:
    """Load Grounding DINO + SAM 2.1; return (grounding_model, sam_model)."""
    gdino_dir = Path(weights_gdino) if weights_gdino else _GDINO_WEIGHTS.parent
    sam2_dir  = Path(weights_sam2)  if weights_sam2  else _SAM2_WEIGHTS.parent

    gdino_weights = gdino_dir / "groundingdino_swint_ogc.pth"
    gdino_config  = gdino_dir / "GroundingDINO_SwinT_OGC.py"
    sam2_weights  = sam2_dir  / "sam2.1_hiera_large.pt"

    try:
        from groundingdino.util.inference import load_model as _load_gdino
    except ImportError:
        raise ImportError("groundingdino not installed. See Grounded-SAM-2 repo.")
    try:
        from sam2.build_sam import build_sam2
    except ImportError:
        raise ImportError("sam2 not installed. See Grounded-SAM-2 repo.")

    for path in [gdino_weights, gdino_config, sam2_weights]:
        if not path.exists():
            raise FileNotFoundError(f"Weights not found: {path}")

    grounding_model = _load_gdino(str(gdino_config), str(gdino_weights), device=device)
    grounding_model = grounding_model.to(device).eval()
    sam_model = build_sam2(_SAM2_CONFIG, str(sam2_weights), device=device)

    return grounding_model, sam_model


def segment_frames(
    frames: list[np.ndarray],
    labels: list[str],
    grounding_model,
    sam_model,
    device: str = "cuda",
    box_threshold: float = 0.3,
    text_threshold: float = 0.25,
) -> list[list[dict]]:
    """Run Grounded-SAM-2 on every frame and return per-frame detection lists."""
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
        if len(boxes_norm) == 0:
            all_results.append(frame_dets)
            continue

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
