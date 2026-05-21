"""Entry point: images or video -> geometry scene directory."""
from __future__ import annotations

import argparse
from pathlib import Path

from src.scene_query.geometry import extract_frames, load_frames_from_dir, load_model, run_vggt

_REPO_ROOT = Path(__file__).parent


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)

    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--video",  type=Path, help="Input video file")
    src.add_argument("--images", type=Path, help="Directory of input images")

    p.add_argument("--out", type=Path, required=True, help="Output scene directory")
    p.add_argument("--fps", type=float, default=2.0, help="Frame sample rate (video input only)")
    p.add_argument("--max-frames", type=int, default=50,
                   help="Cap frames fed to VGGT — quality degrades beyond ~50")
    p.add_argument("--weights-vggt", type=Path,
                   default=_REPO_ROOT / "checkpoints" / "vggt_omega",
                   help="VGGT-Omega weights directory")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--save-rrd", type=Path, default=None,
                   help="Also save a Rerun .rrd recording to this path")
    return p.parse_args()


def _save_frames(frames: list, frames_dir: Path) -> None:
    import cv2
    frames_dir.mkdir(parents=True, exist_ok=True)
    for i, frame in enumerate(frames):
        cv2.imwrite(str(frames_dir / f"{i:04d}.png"), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    print(f"  Frames saved -> {frames_dir}/")


def main() -> None:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

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

    import numpy as np

    # Camera poses
    np.savez_compressed(args.out / "poses.npz", poses=geometry["poses"])

    # Full VGGT cloud — keep top 50% by confidence
    xyz_all  = geometry["xyz"]
    rgb_all  = geometry["rgb"]
    conf_all = geometry["xyz_conf"]
    keep     = conf_all > np.percentile(conf_all, 50)
    xyz_bg, rgb_bg = xyz_all[keep], rgb_all[keep]
    np.savez_compressed(args.out / "scene_cloud.npz", xyz=xyz_bg, rgb=rgb_bg)
    print(f"  Scene cloud: {len(xyz_bg):,} points (after conf filter)")

    # Intrinsics for camera frustum drawing and back-projection
    np.savez_compressed(args.out / "intrinsics.npz", intrinsics=geometry["intrinsics"],
                        image_size=np.array(geometry["image_size"]))

    # Depth maps for GSAM2 mask back-projection at query time
    np.savez_compressed(args.out / "depth.npz",
                        depth=geometry["depth"].astype(np.float16))

    print(f"\nScene saved -> {args.out}")
    print("Query it with:")
    print(f'  python query.py {args.out} bulldozer')
    print(f'  python query.py {args.out} --preset office')

    if args.save_rrd:
        print(f"\nSaving Rerun recording -> {args.save_rrd}")
        _save_rrd(frames, geometry, args.save_rrd)


def _save_rrd(frames: list, geometry: dict, rrd_path: Path) -> None:
    """Write a Rerun .rrd: raw camera feed | 3D reconstruction."""
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
        rrb.TimePanel(playback_speed=0.1, loop_mode=rrb.components.LoopMode.All),
        collapse_panels=True,
        auto_views=False,
    )

    rr.init("scene-query", spawn=False)
    rr.save(str(rrd_path))
    rr.send_blueprint(blueprint, make_active=True, make_default=True)
    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Y_DOWN, static=True)

    H_d, W_d = geometry["image_size"]

    xyz_all  = geometry["xyz"]
    rgb_all  = geometry["rgb"]
    conf_all = geometry["xyz_conf"]
    keep     = conf_all > np.percentile(conf_all, 50)
    rgb_u8   = (np.clip(rgb_all[keep], 0, 1) * 255).astype(np.uint8)
    rr.log("world/photo", rr.Points3D(
        xyz_all[keep], colors=rgb_u8, radii=rr.Radius.ui_points(2.0),
    ), static=True)

    for i, frame in enumerate(frames):
        rr.set_time("frame", sequence=i)

        H_orig, W_orig = frame.shape[:2]
        rr.log("camera/rgb", rr.Image(frame))

        pose   = geometry["poses"][i]
        K      = geometry["intrinsics"][i]
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


if __name__ == "__main__":
    main()
