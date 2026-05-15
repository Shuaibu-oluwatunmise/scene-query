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
    p.add_argument("--geometry-only", action="store_true",
                   help="Run VGGT only — skip semantics and lifting. Saves raw point cloud as .ply.")
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
    print(f"  Points: {len(geometry['xyz']):,}")

    if args.geometry_only:
        _save_geometry(geometry, args.out)
        return

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
    print(f"\nScene saved -> {args.out}")
    print("Query it with:")
    print(f'  python query.py {args.out} "find the chair"')


def _save_geometry(geometry: dict, output_dir: Path) -> None:
    """Save raw VGGT output: poses, depth, and a filtered .ply point cloud."""
    import struct
    import numpy as np
    from scipy.spatial import cKDTree

    output_dir.mkdir(parents=True, exist_ok=True)

    # Save poses and depth
    np.savez(output_dir / "poses.npz", poses=geometry["poses"])
    np.savez(output_dir / "depth.npz", depth=geometry["depth"],
             depth_conf=geometry["depth_conf"], intrinsics=geometry["intrinsics"])

    # Filter point cloud
    xyz, rgb, conf = geometry["xyz"], geometry["rgb"], geometry["xyz_conf"]
    mask = conf > np.percentile(conf, 25)
    xyz, rgb = xyz[mask], rgb[mask]

    tree = cKDTree(xyz)
    dists, _ = tree.query(xyz, k=17)
    mean_dists = dists[:, 1:].mean(axis=1)
    mask = mean_dists < mean_dists.mean() + 2.0 * mean_dists.std()
    xyz, rgb = xyz[mask], rgb[mask]

    # Write .ply
    ply_path = output_dir / "pointcloud_raw.ply"
    n = len(xyz)
    header = (
        "ply\nformat binary_little_endian 1.0\n"
        f"element vertex {n}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\n"
        "end_header\n"
    )
    rgb_u8 = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)
    dtype = np.dtype([("x","<f4"),("y","<f4"),("z","<f4"),
                      ("red","u1"),("green","u1"),("blue","u1")])
    data = np.empty(n, dtype=dtype)
    data["x"], data["y"], data["z"] = xyz[:,0], xyz[:,1], xyz[:,2]
    data["red"], data["green"], data["blue"] = rgb_u8[:,0], rgb_u8[:,1], rgb_u8[:,2]
    with open(ply_path, "wb") as f:
        f.write(header.encode())
        f.write(data.tobytes())

    print(f"\nGeometry saved -> {output_dir}")
    print(f"  poses.npz, depth.npz, pointcloud_raw.ply ({n:,} points)")


if __name__ == "__main__":
    main()
