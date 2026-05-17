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

    if (save_rrd is not None and query_type == "object" and result is not None
            and images_dir is not None and scene_dir is not None):
        _visualise_focused(scene, result, save_rrd, images_dir, scene_dir)
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
# Focused 5-panel object-query recording
# ---------------------------------------------------------------------------

def _visualise_focused(
    scene: dict,
    result: dict,
    save_rrd: Path,
    images_dir: Path,
    scene_dir: Path,
) -> None:
    """5-panel recording.

    Top row  (timeline):  Camera feed | 3D reconstruction (cam-locked) | Segmentation
    Bottom row (static):  Grey 3D + queried label coloured | Plan view with bbox
    """
    import rerun as rr
    import rerun.blueprint as rrb
    import rerun.blueprint.archetypes as ba

    label  = result["label"]
    colour = _label_colour(label)

    poses_arr = scene["poses"]   # (N, 4, 4)

    # Starting eye for panels 4 & 5: first camera pose
    # Column 3 = position, column 2 = forward, column 1 = up (in world space)
    cam0      = poses_arr[0]
    cam0_pos  = cam0[:3, 3].tolist()
    cam0_fwd  = (cam0[:3, 3] + cam0[:3, 2]).tolist()   # one unit forward
    cam0_up   = (-cam0[:3, 1]).tolist()                 # negate: OpenCV Y points down

    bg = [20, 20, 20]

    first_cam_eye = ba.EyeControls3D(
        position=cam0_pos,
        look_target=cam0_fwd,
        eye_up=cam0_up,
    )

    blueprint = rrb.Blueprint(
        rrb.Vertical(
            # Top row: timeline-driven panels
            rrb.Horizontal(
                rrb.Spatial2DView(name="Camera", contents=["camera/rgb"]),
                rrb.Spatial3DView(
                    name="3D Reconstruction",
                    contents=["world/photo", "world/camera"],
                    eye_controls=ba.EyeControls3D(tracking_entity="world/camera"),
                    background=bg,
                ),
                rrb.Spatial2DView(name=f"Segmentation — {label}", contents=["camera/segmentation"]),
            ),
            # Bottom row: static overview panels, both start from first camera POV
            rrb.Horizontal(
                rrb.Spatial3DView(
                    name=f"Objects — {label}",
                    contents=["world/scene_bg", "world/query_pts"],
                    eye_controls=first_cam_eye,
                    background=bg,
                ),
                rrb.Spatial3DView(
                    name=f"{label} — bbox",
                    contents=["world/photo", "world/query_bbox"],
                    eye_controls=first_cam_eye,
                    background=bg,
                ),
            ),
            row_shares=[3, 2],
        ),
        collapse_panels=True,
        auto_views=False,
    )

    rr.init("scene-query", spawn=False)
    rr.save(str(save_rrd))
    rr.send_blueprint(blueprint, make_active=True, make_default=True)

    # --- Static entities ---
    # Use full VGGT scene cloud if available, else fall back to labelled points
    bg_xyz = scene["scene_xyz"] if scene.get("scene_xyz") is not None else scene["xyz"]
    bg_rgb = scene["scene_rgb"] if scene.get("scene_rgb") is not None else scene["rgb"]
    bg_rgb_u8 = (np.clip(bg_rgb, 0, 1) * 255).astype(np.uint8)

    # Photo-coloured cloud (panels 2 & 5) — full scene regardless of labels
    rr.log("world/photo", rr.Points3D(
        bg_xyz, colors=bg_rgb_u8, radii=rr.Radius.ui_points(1.5)
    ), static=True)

    # Grey cloud for panel 4 context — full scene
    grey = np.full((len(bg_xyz), 3), 65, dtype=np.uint8)
    rr.log("world/scene_bg", rr.Points3D(
        bg_xyz, colors=grey, radii=rr.Radius.ui_points(1.0)
    ), static=True)

    # All points belonging to the queried label
    idxs = np.array(scene["label_index"][label], dtype=np.int64)
    pts  = scene["xyz"][idxs]

    # Panel 4: only show points inside the (percentile-trimmed) bbox — removes outliers
    bbox_min = result["bbox_min"]
    bbox_max = result["bbox_max"]
    in_bbox  = np.all((pts >= bbox_min) & (pts <= bbox_max), axis=1)
    pts_clean = pts[in_bbox]
    rr.log("world/query_pts", rr.Points3D(
        pts_clean,
        colors=np.tile(colour, (len(pts_clean), 1)).astype(np.uint8),
        radii=rr.Radius.ui_points(2.0),
    ), static=True)

    # Panel 5: tight bbox over photo-coloured scene
    rr.log("world/query_bbox", rr.Boxes3D(
        mins=[bbox_min.tolist()],
        sizes=[(bbox_max - bbox_min).tolist()],
        colors=[colour],
        labels=[label],
    ), static=True)

    # --- Per-frame timeline (top row) ---
    from src.scene_query.geometry import load_frames_from_dir
    frames = load_frames_from_dir(images_dir)

    intr       = np.load(scene_dir / "intrinsics.npz")
    intrinsics = intr["intrinsics"]
    image_size = intr["image_size"]

    for i, frame in enumerate(frames):
        rr.set_time("frame", sequence=i)

        rr.log("camera/rgb", rr.Image(frame))

        pose = poses_arr[i]
        K    = intrinsics[i] if i < len(intrinsics) else intrinsics[-1]
        H_d, W_d       = int(image_size[0]), int(image_size[1])
        H_orig, W_orig = frame.shape[:2]
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
    """Back-project 3D label points into frame and return a colour overlay.

    Uses a density-map approach: accumulate all projected point hits per pixel,
    Gaussian-blur the density, then threshold. Much more filled than sparse
    point + dilation because every one of the 500K+ label points contributes.
    """
    import cv2

    H, W     = frame.shape[:2]
    H_d, W_d = int(image_size[0]), int(image_size[1])

    K = K_vggt.astype(float).copy()
    K[0] *= W / W_d
    K[1] *= H / H_d

    # World → camera space
    w2c   = np.linalg.inv(pose)
    pts_h = np.hstack([pts_world, np.ones((len(pts_world), 1))])
    cam   = (w2c @ pts_h.T).T[:, :3]

    # Keep only front-facing points
    cam = cam[cam[:, 2] > 0.01]
    if len(cam) == 0:
        return frame.copy()

    # Project to pixel coords (all points — density accuracy requires no subsampling)
    proj = (K @ cam.T).T
    px   = proj[:, 0] / proj[:, 2]
    py   = proj[:, 1] / proj[:, 2]

    valid = (px >= 0) & (px < W) & (py >= 0) & (py < H)
    px_i  = np.clip(np.round(px[valid]).astype(np.int32), 0, W - 1)
    py_i  = np.clip(np.round(py[valid]).astype(np.int32), 0, H - 1)

    # Accumulate hit density per pixel, blur to fill small gaps, threshold
    density = np.zeros((H, W), dtype=np.float32)
    np.add.at(density, (py_i, px_i), 1.0)
    density = cv2.GaussianBlur(density, (0, 0), sigmaX=3)

    # Threshold at 1 hit (after blur ~0.5 keeps only real projections)
    mask = (density > 0.5).astype(np.uint8)
    # Small closing to seal holes inside the object silhouette
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((11, 11), np.uint8))
    mask = mask.astype(bool)

    # Colour blend: label region bright, background lightly desaturated
    c       = np.array(colour, dtype=float)
    overlay = frame.astype(float)
    overlay[mask]  = overlay[mask] * 0.3 + c * 0.7
    overlay[~mask] = overlay[~mask] * 0.6 + 20          # visible but clearly dimmed
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
        images_dir = args.images
        if images_dir is None:
            candidate = args.scene_dir / "frames"
            if candidate.is_dir():
                images_dir = candidate
        visualise(
            scene,
            result,
            query_type,
            save_rrd=args.save_rrd,
            images_dir=images_dir,
            scene_dir=args.scene_dir,
        )


if __name__ == "__main__":
    main()
