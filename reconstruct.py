"""Entry point: phone video → labelled scene directory."""
from __future__ import annotations

import argparse
from pathlib import Path

from src.scene_query.geometry import extract_frames, load_model, run_vggt
from src.scene_query.lift import lift_masks, save_scene
from src.scene_query.semantics import load_models, segment_frames


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("video", type=Path, help="Input video file")
    p.add_argument("--out", type=Path, required=True, help="Output scene directory")
    p.add_argument("--fps", type=float, default=5.0, help="Frame sample rate")
    p.add_argument(
        "--labels",
        type=str,
        default="chair,table,sofa,door,window,bed,desk",
        help="Comma-separated object labels for Grounded-SAM-2",
    )
    p.add_argument("--device", type=str, default="cuda")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    labels = [l.strip() for l in args.labels.split(",")]

    print(f"Extracting frames at {args.fps} fps...")
    frames = extract_frames(args.video, fps=args.fps)
    print(f"  {len(frames)} frames extracted")

    print("Running VGGT...")
    vggt_model = load_model(args.device)
    geometry = run_vggt(frames, vggt_model, args.device)
    # geometry: {poses (N,4,4), depth (N,H,W), intrinsics (N,3,3), xyz (M,3), rgb (M,3)}

    print(f"Running Grounded-SAM-2 for: {labels}...")
    grounding_model, sam_model = load_models(args.device)
    masks_per_frame = segment_frames(
        frames, labels, grounding_model, sam_model, args.device
    )

    print("Lifting masks to 3D...")
    scene = lift_masks(
        frames=frames,
        depth_maps=geometry["depth"],
        intrinsics=geometry["intrinsics"],
        poses=geometry["poses"],
        masks_per_frame=masks_per_frame,
    )
    scene["poses"] = geometry["poses"]

    save_scene(scene, args.out)
    print(f"Scene saved → {args.out}")


if __name__ == "__main__":
    main()
