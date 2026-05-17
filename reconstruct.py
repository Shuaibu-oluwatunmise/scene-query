"""Entry point: images or video -> labelled scene directory."""
from __future__ import annotations

import argparse
from pathlib import Path

from src.scene_query.geometry import extract_frames, load_frames_from_dir, load_model, run_vggt
from src.scene_query.lift import lift_masks, save_scene
from src.scene_query.semantics import load_models, segment_frames

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


def _save_frames(frames: list, frames_dir: Path) -> None:
    import cv2
    frames_dir.mkdir(parents=True, exist_ok=True)
    for i, frame in enumerate(frames):
        cv2.imwrite(str(frames_dir / f"{i:04d}.png"), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    print(f"  Frames saved -> {frames_dir}/")


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

    # Remove outlier points from labelled cloud
    scene = _clean_scene(scene)

    save_scene(scene, args.out)

    import numpy as np

    # Save full VGGT cloud for background visualisation in query.py.
    # Confidence-filtered so background panels show the whole scene,
    # regardless of what labels were specified at reconstruction time.
    xyz_all  = geometry["xyz"]
    rgb_all  = geometry["rgb"]
    conf_all = geometry["xyz_conf"]
    keep     = conf_all > np.percentile(conf_all, 25)
    np.savez_compressed(args.out / "scene_cloud.npz",
                        xyz=xyz_all[keep], rgb=rgb_all[keep])

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
    """Write a Rerun .rrd recording with:
    - Timeline: original frame, segmentation overlay, moving camera frustum
    - Static: full label-coloured 3-D point cloud + per-label sub-clouds
    """
    import cv2
    import numpy as np
    import rerun as rr
    import rerun.blueprint as rrb

    blueprint = rrb.Blueprint(
        rrb.Vertical(
            # Top row: three 3D views
            rrb.Horizontal(
                rrb.Spatial3DView(
                    name="Original",
                    contents=["world/photo_colours", "world/camera", "world/camera/**"],
                ),
                rrb.Spatial3DView(
                    name="Process — semantics",
                    contents=["world/label_colours", "world/labels/**",
                              "world/camera", "world/camera/**"],
                ),
                rrb.Spatial3DView(
                    name="Result — objects",
                    contents=["world/photo_colours", "world/bboxes/**",
                              "world/camera", "world/camera/**"],
                ),
            ),
            # Bottom row: 2D camera feed + segmentation overlay
            rrb.Horizontal(
                rrb.Spatial2DView(name="Camera feed",    contents=["camera/rgb"]),
                rrb.Spatial2DView(name="Segmentation",   contents=["camera/segmentation"]),
            ),
            row_shares=[3, 2],
        ),
        collapse_panels=True,
        auto_views=False,
    )

    rr.init("scene-query", spawn=False, default_blueprint=blueprint)
    rr.save(str(rrd_path))

    H_d, W_d = geometry["image_size"]   # VGGT depth resolution

    # --- Static: labelled 3-D point cloud ---
    label_colours = np.array(
        [_label_colour(str(l)) for l in scene["label"]],
        dtype=np.uint8,
    )
    # Build label -> indices from the in-memory label array
    label_index: dict[str, list[int]] = {}
    for idx, lbl in enumerate(scene["label"]):
        label_index.setdefault(str(lbl), []).append(idx)

    # Original photo colours — shows the scene as it actually looks
    rgb_u8 = (np.clip(scene["rgb"], 0, 1) * 255).astype(np.uint8)
    rr.log("world/photo_colours", rr.Points3D(scene["xyz"], colors=rgb_u8, radii=0.003),
           static=True)

    # Semantic label colours — one colour per object class
    rr.log("world/label_colours", rr.Points3D(scene["xyz"], colors=label_colours, radii=0.003),
           static=True)

    # Per-label sub-clouds (toggleable in the entity panel)
    for lbl, idxs in label_index.items():
        idx_arr = np.array(idxs, dtype=np.int64)
        colour  = _label_colour(lbl)
        pts     = scene["xyz"][idx_arr]
        rr.log(
            f"world/labels/{lbl}",
            rr.Points3D(
                pts,
                colors=np.tile(colour, (len(idx_arr), 1)).astype(np.uint8),
                radii=0.003,
            ),
            static=True,
        )

        # Tight bounding box per label (5th–95th percentile, no outlier blow-up)
        lo = np.percentile(pts, 5,  axis=0)
        hi = np.percentile(pts, 95, axis=0)
        inliers   = np.all((pts >= lo) & (pts <= hi), axis=1)
        pts_clean = pts[inliers] if inliers.any() else pts
        rr.log(
            f"world/bboxes/{lbl}",
            rr.Boxes3D(
                mins=[pts_clean.min(axis=0).tolist()],
                sizes=[(pts_clean.max(axis=0) - pts_clean.min(axis=0)).tolist()],
                colors=[colour],
                labels=[lbl],
            ),
            static=True,
        )

    # --- Per-frame timeline ---
    for i, frame in enumerate(frames):
        rr.set_time("frame", sequence=i)

        H_orig, W_orig = frame.shape[:2]

        # Original RGB frame
        rr.log("camera/rgb", rr.Image(frame))

        # Segmentation overlay: blend label colours onto the frame
        overlay = frame.astype(float)
        for det in masks_per_frame[i]:
            mask = det["mask"]          # (H_orig, W_orig) bool from SAM 2
            colour = np.array(_label_colour(det["label"]), dtype=float)
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
