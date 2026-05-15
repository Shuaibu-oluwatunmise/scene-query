"""Grounded-SAM-2 wrapper: frames + text labels -> per-frame instance masks."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image

_REPO_ROOT = Path(__file__).parent.parent.parent
_GDINO_WEIGHTS = _REPO_ROOT / "checkpoints" / "grounding_dino" / "groundingdino_swint_ogc.pth"
_GDINO_CONFIG  = _REPO_ROOT / "checkpoints" / "grounding_dino" / "GroundingDINO_SwinT_OGC.py"
_SAM2_WEIGHTS  = _REPO_ROOT / "checkpoints" / "sam2" / "sam2.1_hiera_large.pt"
_SAM2_CONFIG   = "configs/sam2.1/sam2.1_hiera_l.yaml"


def load_models(device: str = "cuda") -> tuple:
    """Load Grounding DINO + SAM 2.1; return (grounding_model, sam_model).

    Weights are read from checkpoints/grounding_dino/ and checkpoints/sam2/.
    Run scripts/download_grounding_dino.py and scripts/download_sam2.py first.
    """
    try:
        from groundingdino.util.inference import load_model as _load_gdino
    except ImportError:
        raise ImportError(
            "groundingdino not installed.\n"
            "Install Grounded-SAM-2 from: https://github.com/IDEA-Research/Grounded-SAM-2"
        )
    try:
        from sam2.build_sam import build_sam2
    except ImportError:
        raise ImportError(
            "sam2 not installed.\n"
            "Install Grounded-SAM-2 from: https://github.com/IDEA-Research/Grounded-SAM-2"
        )

    for path in [_GDINO_WEIGHTS, _GDINO_CONFIG, _SAM2_WEIGHTS]:
        if not path.exists():
            raise FileNotFoundError(
                f"Weights not found: {path}\n"
                "Run the download scripts:\n"
                "  python scripts/download_sam2.py\n"
                "  python scripts/download_grounding_dino.py"
            )

    grounding_model = _load_gdino(str(_GDINO_CONFIG), str(_GDINO_WEIGHTS), device=device)
    grounding_model = grounding_model.to(device).eval()

    sam_model = build_sam2(_SAM2_CONFIG, str(_SAM2_WEIGHTS), device=device)

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
    """Run Grounded-SAM-2 on every frame and return per-frame detection lists.

    Parameters
    ----------
    frames        : list of N (H, W, 3) uint8 RGB frames
    labels        : open-vocabulary label strings, e.g. ["chair", "table"]
    box_threshold : Grounding DINO detection confidence cutoff
    text_threshold: Grounding DINO text-matching confidence cutoff

    Returns
    -------
    List of N frame results. Each frame result is a list of detections:
        label       : str
        mask        : (H, W) bool
        confidence  : float
        instance_id : int
    """
    import groundingdino.datasets.transforms as T
    from groundingdino.util.inference import predict as gdino_predict
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    # Grounding DINO prompt: "chair . table . sofa ."
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

        # --- Grounding DINO: detect boxes ---
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

        # Convert normalised [cx, cy, w, h] -> absolute [x1, y1, x2, y2]
        bn = boxes_norm.cpu().numpy()
        cx, cy, bw, bh = bn[:, 0], bn[:, 1], bn[:, 2], bn[:, 3]
        boxes_xyxy = np.stack([
            np.clip((cx - bw / 2) * W, 0, W - 1),
            np.clip((cy - bh / 2) * H, 0, H - 1),
            np.clip((cx + bw / 2) * W, 0, W - 1),
            np.clip((cy + bh / 2) * H, 0, H - 1),
        ], axis=-1)
        confs = logits.cpu().numpy()

        # Match each returned phrase to the nearest label
        matched_labels = [_match_label(ph, labels) for ph in phrases]

        # --- SAM 2: segment each box ---
        sam_predictor.set_image(frame)

        for i, (box, conf, label) in enumerate(zip(boxes_xyxy, confs, matched_labels)):
            with torch.inference_mode():
                masks, _, _ = sam_predictor.predict(
                    point_coords=None,
                    point_labels=None,
                    box=box,
                    multimask_output=False,
                )
            frame_dets.append({
                "label":       label,
                "mask":        masks[0].astype(bool),   # (H, W)
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
    """Return the label from `labels` that best matches a Grounding DINO phrase."""
    ph = phrase.lower().strip()
    for lbl in labels:
        if lbl in ph or ph in lbl:
            return lbl
    return ph
