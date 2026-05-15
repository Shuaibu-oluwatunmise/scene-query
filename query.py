"""Entry point: query a labelled scene directory."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from src.scene_query.query_engine import (
    load_scene,
    query_free_space,
    query_object,
    query_reachable,
)


def _label_colour(label: str) -> list[int]:
    """Deterministic, visually distinct colour for any label string."""
    import hashlib
    import colorsys
    h = int(hashlib.md5(label.encode()).hexdigest(), 16)
    r, g, b = colorsys.hsv_to_rgb((h % 360) / 360.0, 0.75, 0.95)
    return [int(r * 255), int(g * 255), int(b * 255)]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("scene_dir", type=Path, help="Scene directory from reconstruct.py")
    p.add_argument("query", nargs="?", help='Object to find, e.g. "find the chair"')
    p.add_argument("--free-space", action="store_true", help="Query traversable floor space")
    p.add_argument("--floor-z", type=float, default=0.0, help="Floor height in metres")
    p.add_argument(
        "--reachable-from",
        nargs=3,
        type=float,
        metavar=("X", "Y", "Z"),
        help="Robot base position; returns objects within reach",
    )
    p.add_argument("--reach", type=float, default=0.7, help="Arm reach radius in metres")
    p.add_argument("--no-vis", action="store_true", help="Skip all visualisation")
    p.add_argument(
        "--save-rrd", type=Path, default=None,
        help="Save Rerun recording to this .rrd file instead of spawning a live viewer",
    )
    p.add_argument(
        "--images", type=Path, default=None,
        help="Original images directory — enables frame timeline and segmentation overlay",
    )
    return p.parse_args()


def _label_colours(scene: dict) -> np.ndarray:
    return np.array([_label_colour(str(l)) for l in scene["label"]], dtype=np.uint8)


# ---------------------------------------------------------------------------
# Standard (non-object-query) visualisation
# ---------------------------------------------------------------------------

def visualise(
    scene: dict,
    result=None,
    query_type: str | None = None,
    save_rrd: Path | None = None,
    images_dir: Path | None = None,
    scene_dir: Path | None = None,
) -> None:
    """Show the scene in Rerun.

    When save_rrd is set and query_type is "object", generates the focused
    6-panel recording. Otherwise logs a plain scene + optional overlay.
    """
    import rerun as rr

    if save_rrd is not None and query_type == "object" and result is not None:
        _visualise_6panel(scene, result, save_rrd, images_dir, scene_dir)
        return

    rr.init("scene-query", spawn=False)
    if save_rrd:
        rr.save(str(save_rrd))
        print(f"Recording to {save_rrd} ...")
    else:
        rr.spawn()

    rr.log("world/scene", rr.Points3D(scene["xyz"], colors=_label_colours(scene), radii=0.003))

    for lbl, idxs in scene["label_index"].items():
        idx_arr = np.array(idxs, dtype=np.int64)
        colour  = _label_colour(lbl)
        rr.log(
            f"world/labels/{lbl}",
            rr.Points3D(
                scene["xyz"][idx_arr],
                colors=np.tile(colour, (len(idx_arr), 1)).astype(np.uint8),
                radii=0.003,
            ),
        )

    if scene.get("poses") is not None:
        for i, pose in enumerate(scene["poses"]):
            rr.log(
                f"world/cameras/{i:02d}",
                rr.Transform3D(translation=pose[:3, 3].tolist(), mat3x3=pose[:3, :3].tolist()),
            )

    if result is not None:
        if query_type == "object":
            rr.log(
                "world/query/bbox",
                rr.Boxes3D(
                    mins=[result["bbox_min"].tolist()],
                    sizes=[(result["bbox_max"] - result["bbox_min"]).tolist()],
                    colors=[[255, 255, 0]],
                    labels=[result["label"]],
                ),
            )
            rr.log(
                "world/query/centroid",
                rr.Points3D([result["centroid"].tolist()], colors=[[255, 255, 0]], radii=0.05),
            )
        elif query_type == "free_space":
            rr.log("world/query/free_space", rr.Points3D(result, colors=[[0, 220, 100]], radii=0.025))
        elif query_type == "reachable":
            for r in result:
                rr.log(
                    f"world/query/reachable/{r['label']}",
                    rr.Boxes3D(
                        mins=[r["bbox_min"].tolist()],
                        sizes=[(r["bbox_max"] - r["bbox_min"]).tolist()],
                        colors=[[255, 200, 0]],
                        labels=[r["label"]],
                    ),
                )

    if save_rrd:
        print(f"Saved -> {save_rrd}")


# ---------------------------------------------------------------------------
# Focused 6-panel object-query recording
# ---------------------------------------------------------------------------

def _visualise_6panel(
    scene: dict,
    result: dict,
    save_rrd: Path,
    images_dir: Path | None,
    scene_dir: Path | None,
) -> None:
    """2×3 panel recording focused on the queried object.

    Top row (3D):
      Panel 2 – full photo-coloured reconstruction + animated camera frustum
      Panel 4 – grey scene + queried label in colour (no camera clutter)
      Panel 5 – grey scene + queried label + bbox, viewport locked to camera

    Bottom row (2D):
      Panel 1 – RGB camera feed
      Panel 3 – segmentation overlay (queried label highlighted, rest dimmed)
      Panel 6 – placeholder (coming soon)
    """
    import rerun as rr
    import rerun.blueprint as rrb
    import rerun.blueprint.archetypes as ba

    label  = result["label"]
    colour = _label_colour(label)

    has_frames     = images_dir is not None and images_dir.exists()
    has_intrinsics = (
        scene_dir is not None and
        (scene_dir / "intrinsics.npz").exists()
    )

    # --- Blueprint ---
    cam = ["world/camera"]

    panel2 = rrb.Spatial3DView(
        name="Reconstruction",
        contents=["world/photo"] + cam,
    )
    panel4 = rrb.Spatial3DView(
        name=f"Query: {label}",
        contents=["world/scene_bg", "world/query_pts"],
    )
    panel5 = rrb.Spatial3DView(
        name=f"{label} — bbox",
        contents=["world/scene_bg", "world/query_pts", "world/query_bbox", "world/camera"],
        eye_controls=ba.EyeControls3D(tracking_entity="world/camera"),
    )
    panel1 = rrb.Spatial2DView(name="RGB",          contents=["camera/rgb"])
    panel3 = rrb.Spatial2DView(name="Segmentation", contents=["camera/segmentation"])
    panel6 = rrb.TextDocumentView(name="—",         contents=["info/panel6"])

    blueprint = rrb.Blueprint(
        rrb.Vertical(
            rrb.Horizontal(panel2, panel4, panel5),
            rrb.Horizontal(panel1, panel3, panel6),
            row_shares=[3, 2],
        ),
        collapse_panels=True,
        auto_views=False,
    )

    rr.init("scene-query", spawn=False, default_blueprint=blueprint)
    rr.save(str(save_rrd))

    # --- Static: photo-coloured full point cloud (panel 2 background) ---
    rgb_u8 = (np.clip(scene["rgb"], 0, 1) * 255).astype(np.uint8)
    rr.log("world/photo", rr.Points3D(scene["xyz"], colors=rgb_u8, radii=0.003), static=True)

    # --- Static: grey scene for context (panels 4 & 5) ---
    bg = np.full((len(scene["xyz"]), 3), 65, dtype=np.uint8)
    rr.log("world/scene_bg", rr.Points3D(scene["xyz"], colors=bg, radii=0.002), static=True)

    # --- Static: queried label points, bright ---
    idxs = np.array(scene["label_index"][label], dtype=np.int64)
    pts  = scene["xyz"][idxs]
    rr.log(
        "world/query_pts",
        rr.Points3D(pts, colors=np.tile(colour, (len(pts), 1)).astype(np.uint8), radii=0.005),
        static=True,
    )

    # --- Static: tight bbox ---
    rr.log(
        "world/query_bbox",
        rr.Boxes3D(
            mins=[result["bbox_min"].tolist()],
            sizes=[(result["bbox_max"] - result["bbox_min"]).tolist()],
            colors=[colour],
            labels=[label],
        ),
        static=True,
    )

    # --- Static: panel 6 placeholder ---
    rr.log("info/panel6", rr.TextDocument("Panel 6 — coming soon."), static=True)

    # --- Per-frame timeline ---
    if not has_frames:
        print(f"Saved -> {save_rrd}")
        return

    from src.scene_query.geometry import load_frames_from_dir
    frames = load_frames_from_dir(images_dir)

    intrinsics = None
    image_size = None
    if has_intrinsics:
        intr       = np.load(scene_dir / "intrinsics.npz")
        intrinsics = intr["intrinsics"]   # (N, 3, 3) at VGGT depth resolution
        image_size = intr["image_size"]   # [H_d, W_d]

    poses = scene.get("poses")

    for i, frame in enumerate(frames):
        rr.set_time("frame", sequence=i)

        # Panel 1: raw RGB
        rr.log("camera/rgb", rr.Image(frame))

        if intrinsics is None or poses is None or i >= len(poses):
            rr.log("camera/segmentation", rr.Image(frame))
            continue

        pose = poses[i]
        K    = intrinsics[i] if i < len(intrinsics) else intrinsics[-1]
        H_d, W_d     = int(image_size[0]), int(image_size[1])
        H_orig, W_orig = frame.shape[:2]
        sx, sy = W_orig / W_d, H_orig / H_d
        fx = float(K[0, 0]) * sx
        fy = float(K[1, 1]) * sy
        cx = float(K[0, 2]) * sx
        cy = float(K[1, 2]) * sy

        # Animated camera frustum (panels 2 & 5 see this)
        rr.log(
            "world/camera",
            rr.Transform3D(translation=pose[:3, 3].tolist(), mat3x3=pose[:3, :3].tolist()),
        )
        rr.log(
            "world/camera",
            rr.Pinhole(focal_length=[fx, fy], principal_point=[cx, cy], width=W_orig, height=H_orig),
        )
        # Panel 3: segmentation — back-project 3D label points into this frame
        seg = _seg_overlay(frame, pts, pose, K, image_size, colour)
        rr.log("camera/segmentation", rr.Image(seg))

    print(f"Saved -> {save_rrd}")


def _seg_overlay(
    frame: np.ndarray,
    pts_world: np.ndarray,
    pose: np.ndarray,
    K_vggt: np.ndarray,
    image_size,
    colour: list[int],
) -> np.ndarray:
    """Back-project world-space 3D label points into the frame as a colour mask.

    Steps:
      1. Scale intrinsics from VGGT depth resolution → original frame resolution
      2. Transform points world → camera space
      3. Project to pixel coords; keep in-bounds, front-facing points
      4. Rasterize + dilate to fill projection gaps
      5. Blend: label region bright, rest dimmed
    """
    import cv2

    H, W   = frame.shape[:2]
    H_d, W_d = int(image_size[0]), int(image_size[1])

    K = K_vggt.astype(float).copy()
    K[0] *= W / W_d
    K[1] *= H / H_d

    # World → camera
    w2c    = np.linalg.inv(pose)
    pts_h  = np.hstack([pts_world, np.ones((len(pts_world), 1))])
    cam    = (w2c @ pts_h.T).T[:, :3]

    front  = cam[:, 2] > 0.01
    cam    = cam[front]
    if len(cam) == 0:
        return frame.copy()

    # Subsample so per-frame projection stays fast
    if len(cam) > 60_000:
        step = len(cam) // 60_000 + 1
        cam  = cam[::step]

    proj = (K @ cam.T).T
    px   = proj[:, 0] / proj[:, 2]
    py   = proj[:, 1] / proj[:, 2]

    valid = (px >= 0) & (px < W) & (py >= 0) & (py < H)
    px_i  = np.clip(np.round(px[valid]).astype(np.int32), 0, W - 1)
    py_i  = np.clip(np.round(py[valid]).astype(np.int32), 0, H - 1)

    mask = np.zeros((H, W), dtype=np.uint8)
    mask[py_i, px_i] = 255
    # Aggressive dilation to fill gaps from sparse 3D→2D projection
    mask = cv2.dilate(mask, np.ones((15, 15), np.uint8), iterations=4).astype(bool)

    # Overlay: label region gets colour tint, rest stays visible but slightly muted
    c       = np.array(colour, dtype=float)
    overlay = frame.astype(float)
    overlay[mask] = overlay[mask] * 0.35 + c * 0.65
    # Non-masked area stays close to original (just slight greyscale wash)
    grey              = overlay[~mask].mean(axis=1, keepdims=True)
    overlay[~mask]    = overlay[~mask] * 0.5 + grey * 0.5
    return np.clip(overlay, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args   = parse_args()
    scene  = load_scene(args.scene_dir)
    result = None
    query_type = None

    if args.free_space:
        result     = query_free_space(scene, floor_z=args.floor_z)
        query_type = "free_space"
        print(f"Traversable voxels: {len(result)}")

    elif args.reachable_from:
        base       = np.array(args.reachable_from, dtype=np.float32)
        result     = query_reachable(scene, base_xyz=base, reach=args.reach)
        query_type = "reachable"
        print(f"Reachable objects ({len(result)}):")
        for r in result:
            print(f"  {r['label']:20s}  centroid={r['centroid']}  n={r['n_points']}")

    elif args.query:
        label      = args.query.lower().removeprefix("find the ").removeprefix("find ").strip()
        result     = query_object(scene, label)
        query_type = "object"
        print(f"Label      : {result['label']}")
        print(f"Centroid   : {result['centroid']}")
        print(f"Bbox       : {result['bbox_min']} -> {result['bbox_max']}")
        print(f"Points     : {result['n_points']}")
        print(f"Confidence : {result['confidence']:.2f}")

    if not args.no_vis:
        visualise(
            scene,
            result,
            query_type,
            save_rrd=args.save_rrd,
            images_dir=args.images,
            scene_dir=args.scene_dir,
        )


if __name__ == "__main__":
    main()
