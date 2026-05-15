"""Entry point: images or video → labelled scene directory."""
from __future__ import annotations

import argparse
from pathlib import Path

from src.scene_query.geometry import extract_frames, load_frames_from_dir, load_model, run_vggt
from src.scene_query.lift import lift_masks, save_scene
from src.scene_query.semantics import load_models, segment_frames


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)

    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--video",  type=Path, help="Input video file (e.g. examples/room_test.mp4)")
    src.add_argument("--images", type=Path, help="Directory of input images (e.g. examples/room/)")

    p.add_argument("--out", type=Path, required=True, help="Output scene directory")
    p.add_argument("--fps", type=float, default=5.0, help="Frame sample rate (video input only)")
    p.add_argument(
        "--labels",
        type=str,
        default="chair,table,sofa,door,window,bed,desk",
        help="Comma-separated object labels for Grounded-SAM-2",
    )
    p.add_argument("--weights-vggt", type=Path, default=Path("checkpoints/vggt"),
                   help="VGGT weights directory")
    p.add_argument("--device", type=str, default="cuda")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    labels = [l.strip() for l in args.labels.split(",")]

    # --- Load frames ---
    if args.images:
        print(f"Loading images from {args.images}...")
        frames = load_frames_from_dir(args.images)
    else:
        print(f"Extracting frames at {args.fps} fps from {args.video}...")
        frames = extract_frames(args.video, fps=args.fps)
    print(f"  {len(frames)} frames ready")

    # --- Geometry (VGGT) ---
    print(f"\nRunning VGGT (weights: {args.weights_vggt})...")
    vggt_model = load_model(args.weights_vggt, args.device)
    geometry = run_vggt(frames, vggt_model, args.device)
    print(f"  Poses: {geometry['poses'].shape}  Depth: {geometry['depth'].shape}")

    # --- Semantics (Grounded-SAM-2) ---
    print(f"\nRunning Grounded-SAM-2 for: {labels}...")
    grounding_model, sam_model = load_models(args.device)
    masks_per_frame = segment_frames(
        frames, labels, grounding_model, sam_model, args.device
    )

    # --- Lift to 3D ---
    print("\nLifting masks to 3D...")
    scene = lift_masks(
        frames=frames,
        depth_maps=geometry["depth"],
        intrinsics=geometry["intrinsics"],
        poses=geometry["poses"],
        masks_per_frame=masks_per_frame,
    )
    scene["poses"] = geometry["poses"]

    save_scene(scene, args.out)
    print(f"\nScene saved → {args.out}")
    print("Query it with:")
    print(f'  python query.py {args.out} "find the chair"')


if __name__ == "__main__":
    main()
