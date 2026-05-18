"""Entry point: images or video -> labelled scene directory."""
from __future__ import annotations

import argparse
from pathlib import Path

from src.scene_query.geometry import extract_frames, load_frames_from_dir, load_model, run_vggt
from src.scene_query.lift import lift_masks, save_scene
from src.scene_query.semantics import (
    load_yolo_model, segment_frames_yolo,
    load_models, segment_frames,
)

def _label_colour(label: str) -> list[int]:
    """Deterministic, visually distinct colour for any label string."""
    import hashlib
    h = int(hashlib.md5(label.encode()).hexdigest(), 16)
    # Pick hue from hash, keep saturation and value high for visibility
    import colorsys
    hue = (h % 360) / 360.0
    r, g, b = colorsys.hsv_to_rgb(hue, 0.75, 0.95)
    return [int(r * 255), int(g * 255), int(b * 255)]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)

    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--video",  type=Path, help="Input video file")
    src.add_argument("--images", type=Path, help="Directory of input images")

    p.add_argument("--out", type=Path, required=True, help="Output scene directory")
    p.add_argument("--fps", type=float, default=2.0, help="Frame sample rate (video input only)")
    p.add_argument("--max-frames", type=int, default=50,
                   help="Cap frames fed to VGGT (video input only) — quality degrades beyond ~50")
    p.add_argument(
        "--backend", type=str, default="yolo", choices=["yolo", "gsam2"],
        help="Semantics backend: 'yolo' (default, uses trained office model) or 'gsam2'"
    )
    p.add_argument(
        "--labels",
        type=str,
        default="chair,table,sofa,door,window,bed,desk",
        help="Comma-separated object labels (only used with --backend gsam2)",
    )
    p.add_argument("--weights-vggt", type=Path, default=Path("checkpoints/vggt_omega"),
                   help="VGGT weights directory")
    p.add_argument("--weights-yolo", type=Path, default=None,
                   help="YOLOv8 detection weights .pt file (default: checkpoints/yolo_office/best.pt)")
    p.add_argument("--weights-gdino", type=Path, default=None,
                   help="Grounding DINO weights directory (only with --backend gsam2)")
    p.add_argument("--weights-sam2", type=Path, default=None,
                   help="SAM 2.1 weights directory (only with --backend gsam2)")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--geometry-only", action="store_true",
                   help="Run VGGT only — skip semantics and lifting.")
    p.add_argument("--save-rrd", type=Path, default=None,
                   help="Also save a Rerun .rrd recording to this path")
    return p.parse_args()


def _clean_scene(scene: dict, k: int = 16, std_ratio: float = 2.0) -> dict:
    """Remove statistical outliers from the labelled point cloud."""
    from scipy.spatial import cKDTree
    import numpy as np
    xyz = scene["xyz"]
    if len(xyz) < k + 1:
        return scene
    tree = cKDTree(xyz)
    dists, _ = tree.query(xyz, k=k + 1)
    mean_dists = dists[:, 1:].mean(axis=1)
    keep = mean_dists < (mean_dists.mean() + std_ratio * mean_dists.std())
    print(f"  Outlier removal: {len(xyz):,} -> {keep.sum():,} points")
    return {
        "xyz":        xyz[keep],
        "rgb":        scene["rgb"][keep],
        "label":      scene["label"][keep],
        "confidence": scene["confidence"][keep],
        "poses":      scene.get("poses"),
    }


def _voxel_downsample(xyz: "np.ndarray", rgb: "np.ndarray", voxel_size: float = 0.005):
    import numpy as np
    coords = np.floor(xyz / voxel_size).astype(np.int32)
    _, unique_idx = np.unique(coords, axis=0, return_index=True)
    return xyz[unique_idx], rgb[unique_idx]


def _save_frames(frames: list, frames_dir: Path) -> None:
    import cv2
    frames_dir.mkdir(parents=True, exist_ok=True)
    for i, frame in enumerate(frames):
        cv2.imwrite(str(frames_dir / f"{i:04d}.png"), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    print(f"  Frames saved -> {frames_dir}/")



def main() -> None:
    args = parse_args()

    # --- Load frames ---
    if args.images:
        print(f"Loading images from {args.images}...")
        frames = load_frames_from_dir(args.images)
    else:
        print(f"Extracting frames at {args.fps} fps from {args.video}...")
        frames = extract_frames(args.video, fps=args.fps, max_frames=args.max_frames)
        _save_frames(frames, args.out / "frames")
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

    # --- Semantics ---
    if args.backend == "yolo":
        print(f"\nRunning YOLO semantics...")
        yolo_model = load_yolo_model(args.weights_yolo, args.device)
        print(f"  Classes: {list(yolo_model.names.values())}")
        masks_per_frame = segment_frames_yolo(frames, yolo_model, args.device)
    else:
        labels = [l.strip() for l in args.labels.split(",")]
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

    # Remove outlier points from labelled cloud
    scene = _clean_scene(scene)

    save_scene(scene, args.out)

    import numpy as np

    # Save full VGGT cloud for background visualisation in query.py.
    # Keep top 50% by confidence, then voxel-downsample for uniform density.
    xyz_all  = geometry["xyz"]
    rgb_all  = geometry["rgb"]
    conf_all = geometry["xyz_conf"]
    keep     = conf_all > np.percentile(conf_all, 50)
    xyz_bg, rgb_bg = xyz_all[keep], rgb_all[keep]
    np.savez_compressed(args.out / "scene_cloud.npz", xyz=xyz_bg, rgb=rgb_bg)
    print(f"  Scene cloud: {len(xyz_bg):,} points (after conf filter)")

    # Save intrinsics alongside the scene so query.py can draw camera frustums
    np.savez_compressed(args.out / "intrinsics.npz", intrinsics=geometry["intrinsics"],
                        image_size=np.array(geometry["image_size"]))

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
    """Write a Rerun .rrd with 2 panels:
    - Left:  original camera feed playing as video (timeline)
    - Right: photo-coloured 3D scene with camera moving through it (timeline)
    """
    import numpy as np
    import rerun as rr
    import rerun.blueprint as rrb
    import rerun.blueprint.archetypes as ba

    bg = [20, 20, 20]

    blueprint = rrb.Blueprint(
        rrb.Horizontal(
            rrb.Spatial2DView(name="Camera", contents=["camera/rgb"]),
            rrb.Spatial3DView(
                name="3D Reconstruction",
                contents=["world/photo", "world/camera"],
                eye_controls=ba.EyeControls3D(tracking_entity="world/camera"),
                background=bg,
            ),
        ),
        collapse_panels=True,
        auto_views=False,
    )

    rr.init("scene-query", spawn=False)
    rr.save(str(rrd_path))
    rr.send_blueprint(blueprint, make_active=True, make_default=True)
    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Y_DOWN, static=True)

    H_d, W_d = geometry["image_size"]

    # Static: full photo-coloured point cloud
    xyz_all  = geometry["xyz"]
    rgb_all  = geometry["rgb"]
    conf_all = geometry["xyz_conf"]
    keep     = conf_all > np.percentile(conf_all, 50)
    rgb_u8   = (np.clip(rgb_all[keep], 0, 1) * 255).astype(np.uint8)
    rr.log("world/photo", rr.Points3D(
        xyz_all[keep], colors=rgb_u8, radii=rr.Radius.ui_points(2.0),
    ), static=True)

    # Per-frame timeline
    for i, frame in enumerate(frames):
        rr.set_time("frame", sequence=i)

        H_orig, W_orig = frame.shape[:2]
        rr.log("camera/rgb", rr.Image(frame))

        pose = geometry["poses"][i]
        K    = geometry["intrinsics"][i]
        sx, sy = W_orig / W_d, H_orig / H_d

        rr.log("world/camera", rr.Transform3D(
            translation=pose[:3, 3].tolist(),
            mat3x3=pose[:3, :3].tolist(),
        ))
        rr.log("world/camera", rr.Pinhole(
            focal_length=[float(K[0, 0]) * sx, float(K[1, 1]) * sy],
            principal_point=[float(K[0, 2]) * sx, float(K[1, 2]) * sy],
            width=W_orig, height=H_orig,
        ))
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
    mask = conf > np.percentile(conf, 50)
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
