"""Entry point: images or video -> labelled scene directory."""
from __future__ import annotations

import argparse
from pathlib import Path

from src.scene_query.geometry import extract_frames, load_frames_from_dir, load_model, run_vggt
from src.scene_query.lift import lift_masks, save_scene
from src.scene_query.semantics import load_models, segment_frames

# Colour palette shared with query.py
_PALETTE: dict[str, list[int]] = {
    "bulldozer": [255, 120,  30],
    "table":     [ 80, 200,  80],
    "wheel":     [200,  50,  50],
    "blade":     [ 50, 150, 255],
    "track":     [200, 200,  50],
    "chair":     [180,  60, 220],
    "sofa":      [220, 160,  60],
    "door":      [100, 200, 220],
    "window":    [220, 220, 100],
    "bed":       [160,  80, 160],
    "desk":      [100, 160,  60],
}
_DEFAULT_COLOUR = [160, 160, 160]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)

    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--video",  type=Path, help="Input video file")
    src.add_argument("--images", type=Path, help="Directory of input images")

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
    p.add_argument("--weights-gdino", type=Path, default=None,
                   help="Grounding DINO weights directory (default: checkpoints/grounding_dino/)")
    p.add_argument("--weights-sam2", type=Path, default=None,
                   help="SAM 2.1 weights directory (default: checkpoints/sam2/)")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--geometry-only", action="store_true",
                   help="Run VGGT only — skip semantics and lifting.")
    p.add_argument("--save-rrd", type=Path, default=None,
                   help="Also save a Rerun .rrd recording to this path")
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
    grounding_model, sam_model = load_models(
        args.device,
        weights_gdino=args.weights_gdino,
        weights_sam2=args.weights_sam2,
    )
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

    # --- Rerun recording ---
    if args.save_rrd:
        print(f"\nSaving Rerun recording -> {args.save_rrd}")
        _save_rrd(frames, geometry, masks_per_frame, scene, args.save_rrd)


def _save_rrd(
    frames: list,
    geometry: dict,
    masks_per_frame: list,
    scene: dict,
    rrd_path: Path,
) -> None:
    """Write a Rerun .rrd recording with:
    - Timeline: original frame, segmentation overlay, moving camera frustum
    - Static: full label-coloured 3-D point cloud + per-label sub-clouds
    """
    import cv2
    import numpy as np
    import rerun as rr

    rr.init("scene-query", spawn=False)
    rr.save(str(rrd_path))

    H_d, W_d = geometry["image_size"]   # VGGT depth resolution

    # --- Static: labelled 3-D point cloud ---
    label_colours = np.array(
        [_PALETTE.get(str(l), _DEFAULT_COLOUR) for l in scene["label"]],
        dtype=np.uint8,
    )
    rr.log("world/scene", rr.Points3D(scene["xyz"], colors=label_colours, radii=0.003),
           static=True)

    # Build label -> indices from the in-memory label array
    label_index: dict[str, list[int]] = {}
    for idx, lbl in enumerate(scene["label"]):
        label_index.setdefault(str(lbl), []).append(idx)

    for lbl, idxs in label_index.items():
        idx_arr = np.array(idxs, dtype=np.int64)
        colour  = _PALETTE.get(lbl, _DEFAULT_COLOUR)
        rr.log(
            f"world/labels/{lbl}",
            rr.Points3D(
                scene["xyz"][idx_arr],
                colors=np.tile(colour, (len(idx_arr), 1)).astype(np.uint8),
                radii=0.003,
            ),
            static=True,
        )

    # --- Per-frame timeline ---
    for i, frame in enumerate(frames):
        rr.set_time_sequence("frame", i)

        H_orig, W_orig = frame.shape[:2]

        # Original RGB frame
        rr.log("camera/rgb", rr.Image(frame))

        # Segmentation overlay: blend label colours onto the frame
        overlay = frame.astype(float)
        for det in masks_per_frame[i]:
            mask = det["mask"]          # (H_orig, W_orig) bool from SAM 2
            colour = np.array(_PALETTE.get(det["label"], _DEFAULT_COLOUR), dtype=float)
            overlay[mask] = overlay[mask] * 0.35 + colour * 0.65
        rr.log("camera/segmentation", rr.Image(overlay.astype(np.uint8)))

        # Camera frustum in 3-D world space
        pose = geometry["poses"][i]     # (4, 4) camera-to-world
        K    = geometry["intrinsics"][i]  # (3, 3) at VGGT resolution

        # Scale intrinsics from VGGT depth resolution to original frame size
        sx = W_orig / W_d
        sy = H_orig / H_d
        fx, fy = float(K[0, 0]) * sx, float(K[1, 1]) * sy
        cx, cy = float(K[0, 2]) * sx, float(K[1, 2]) * sy

        rr.log(
            "world/camera",
            rr.Transform3D(
                translation=pose[:3, 3].tolist(),
                mat3x3=pose[:3, :3].tolist(),
            ),
        )
        rr.log(
            "world/camera",
            rr.Pinhole(
                focal_length=[fx, fy],
                principal_point=[cx, cy],
                width=W_orig,
                height=H_orig,
            ),
        )
        # Image projected through the frustum in the 3-D view
        rr.log("world/camera/image", rr.Image(frame))

    print(f"Saved -> {rrd_path}")


def _save_geometry(geometry: dict, output_dir: Path) -> None:
    """Save raw VGGT output: poses, depth, and a filtered .ply point cloud."""
    import numpy as np
    from scipy.spatial import cKDTree

    output_dir.mkdir(parents=True, exist_ok=True)

    np.savez(output_dir / "poses.npz", poses=geometry["poses"])
    np.savez(output_dir / "depth.npz", depth=geometry["depth"],
             depth_conf=geometry["depth_conf"], intrinsics=geometry["intrinsics"])

    xyz, rgb, conf = geometry["xyz"], geometry["rgb"], geometry["xyz_conf"]
    mask = conf > np.percentile(conf, 25)
    xyz, rgb = xyz[mask], rgb[mask]

    tree = cKDTree(xyz)
    dists, _ = tree.query(xyz, k=17)
    mean_dists = dists[:, 1:].mean(axis=1)
    mask = mean_dists < mean_dists.mean() + 2.0 * mean_dists.std()
    xyz, rgb = xyz[mask], rgb[mask]

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
