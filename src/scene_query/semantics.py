"""Grounded-SAM-2 wrapper: frames + text labels → per-frame instance masks."""
from __future__ import annotations

import numpy as np


def load_models(device: str = "cuda") -> tuple:
    """Load Grounding DINO + SAM 2.1; return (grounding_model, sam_model).

    Weights are read from checkpoints/grounding_dino/ and checkpoints/sam2/.
    """
    ...


def segment_frames(
    frames: list[np.ndarray],
    labels: list[str],
    grounding_model,
    sam_model,
    device: str = "cuda",
    box_threshold: float = 0.3,
    text_threshold: float = 0.25,
) -> list[list[dict]]:
    """Run Grounded-SAM-2 on every frame and track instances across frames.

    SAM 2's video propagation keeps instance IDs consistent across the clip,
    so the same chair accumulates points from all views that observe it.

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
        instance_id : int  — consistent across frames via SAM 2 tracking
    """
    ...
