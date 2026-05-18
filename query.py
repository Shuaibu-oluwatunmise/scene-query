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
    p.add_argument("query", nargs="*", help='Objects to find, e.g. "bulldozer table mug"')
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
        if isinstance(result, list):
            _visualise_multi(scene, result, save_rrd, images_dir, scene_dir)
        else:
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
                    centers=[result["obb_center"].tolist()],
                    half_sizes=[result["obb_half"].tolist()],
                    quaternions=[rr.Quaternion(xyzw=result["obb_quat"].tolist())],
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
            ),
            # Bottom row: static overview panels
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
        rrb.TimePanel(playback_speed=0.1, loop_mode=rrb.components.LoopMode.All),
        collapse_panels=True,
        auto_views=False,
    )

    rr.init("scene-query", spawn=False)
    rr.save(str(save_rrd))
    rr.send_blueprint(blueprint, make_active=True, make_default=True)

    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Y_DOWN, static=True)

    # --- Static entities ---
    # Use full VGGT scene cloud if available, else fall back to labelled points
    bg_xyz = scene["scene_xyz"] if scene.get("scene_xyz") is not None else scene["xyz"]
    bg_rgb = scene["scene_rgb"] if scene.get("scene_rgb") is not None else scene["rgb"]
    bg_rgb_u8 = (np.clip(bg_rgb, 0, 1) * 255).astype(np.uint8)

    # Photo-coloured cloud (panels 2 & 5) — full scene regardless of labels
    rr.log("world/photo", rr.Points3D(
        bg_xyz, colors=bg_rgb_u8, radii=rr.Radius.ui_points(2.0)
    ), static=True)

    # Grey cloud for panel 4 context — full scene
    grey = np.full((len(bg_xyz), 3), 65, dtype=np.uint8)
    rr.log("world/scene_bg", rr.Points3D(
        bg_xyz, colors=grey, radii=rr.Radius.ui_points(1.0)
    ), static=True)

    # All points belonging to the queried label
    # pts needed for segmentation overlay in the timeline loop below
    idxs = np.array(scene["label_index"][label], dtype=np.int64)
    pts  = scene["xyz"][idxs]

    # Panel 4: highlight ALL scene points inside the bbox, grey everything else
    bbox_min = result["bbox_min"]
    bbox_max = result["bbox_max"]
    in_bbox  = np.all((bg_xyz >= bbox_min) & (bg_xyz <= bbox_max), axis=1)

    rr.log("world/query_pts", rr.Points3D(
        bg_xyz[in_bbox],
        colors=np.tile(colour, (in_bbox.sum(), 1)).astype(np.uint8),
        radii=rr.Radius.ui_points(2.5),
    ), static=True)

    # Panel 5: OBB over photo-coloured scene
    rr.log("world/query_bbox", rr.Boxes3D(
        centers=[result["obb_center"].tolist()],
        half_sizes=[result["obb_half"].tolist()],
        quaternions=[rr.Quaternion(xyzw=result["obb_quat"].tolist())],
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

    print(f"Saved -> {save_rrd}")


# ---------------------------------------------------------------------------
# Multi-object 4-panel recording
# ---------------------------------------------------------------------------

def _visualise_multi(
    scene: dict,
    results: list[dict],
    save_rrd: Path,
    images_dir: Path,
    scene_dir: Path,
) -> None:
    """4-panel recording for multiple queried objects.

    Top row  (timeline): Camera feed | 3D reconstruction (cam-locked)
    Bottom row (static): Grey scene + all labels coloured | Photo cloud + all OBBs
    """
    import rerun as rr
    import rerun.blueprint as rrb
    import rerun.blueprint.archetypes as ba

    labels  = [r["label"] for r in results]
    colours = [_label_colour(lbl) for lbl in labels]
    name    = " + ".join(labels)

    poses_arr = scene["poses"]
    cam0      = poses_arr[0]
    cam0_pos  = cam0[:3, 3].tolist()
    cam0_fwd  = (cam0[:3, 3] + cam0[:3, 2]).tolist()
    cam0_up   = (-cam0[:3, 1]).tolist()
    bg        = [20, 20, 20]

    first_cam_eye = ba.EyeControls3D(
        position=cam0_pos,
        look_target=cam0_fwd,
        eye_up=cam0_up,
    )

    blueprint = rrb.Blueprint(
        rrb.Vertical(
            rrb.Horizontal(
                rrb.Spatial2DView(name="Camera", contents=["camera/rgb"]),
                rrb.Spatial3DView(
                    name="3D Reconstruction",
                    contents=["world/photo", "world/camera"],
                    eye_controls=ba.EyeControls3D(tracking_entity="world/camera"),
                    background=bg,
                ),
            ),
            rrb.Horizontal(
                rrb.Spatial3DView(
                    name=f"Objects — {name}",
                    contents=["world/scene_bg", "world/query_pts"],
                    eye_controls=first_cam_eye,
                    background=bg,
                ),
                rrb.Spatial3DView(
                    name="Bounding Boxes",
                    contents=["world/photo", "world/query_bbox/**"],
                    eye_controls=first_cam_eye,
                    background=bg,
                ),
            ),
            row_shares=[3, 2],
        ),
        rrb.TimePanel(playback_speed=0.1, loop_mode=rrb.components.LoopMode.All),
        collapse_panels=True,
        auto_views=False,
    )

    rr.init("scene-query", spawn=False)
    rr.save(str(save_rrd))
    rr.send_blueprint(blueprint, make_active=True, make_default=True)
    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Y_DOWN, static=True)

    bg_xyz    = scene["scene_xyz"] if scene.get("scene_xyz") is not None else scene["xyz"]
    bg_rgb    = scene["scene_rgb"] if scene.get("scene_rgb") is not None else scene["rgb"]
    bg_rgb_u8 = (np.clip(bg_rgb, 0, 1) * 255).astype(np.uint8)

    rr.log("world/photo", rr.Points3D(
        bg_xyz, colors=bg_rgb_u8, radii=rr.Radius.ui_points(2.0),
    ), static=True)

    grey = np.full((len(bg_xyz), 3), 65, dtype=np.uint8)
    rr.log("world/scene_bg", rr.Points3D(
        bg_xyz, colors=grey, radii=rr.Radius.ui_points(1.0),
    ), static=True)

    # Assign intersecting points to the label with the smallest bbox (most specific).
    # Process largest bbox first so smallest overwrites.
    volumes = [float(np.prod(r["bbox_max"] - r["bbox_min"])) for r in results]
    order   = np.argsort(volumes)[::-1]   # largest → smallest

    colour_map = np.zeros((len(bg_xyz), 3), dtype=np.uint8)
    assigned   = np.zeros(len(bg_xyz), dtype=bool)

    for i in order:
        result  = results[i]
        colour  = colours[i]
        in_bbox = np.all((bg_xyz >= result["bbox_min"]) & (bg_xyz <= result["bbox_max"]), axis=1)
        colour_map[in_bbox] = colour
        assigned |= in_bbox

    if assigned.any():
        rr.log("world/query_pts", rr.Points3D(
            bg_xyz[assigned],
            colors=colour_map[assigned],
            radii=rr.Radius.ui_points(2.5),
        ), static=True)

    for result, colour in zip(results, colours):
        label    = result["label"]
        rr.log(f"world/query_bbox/{label}", rr.Boxes3D(
            centers=[result["obb_center"].tolist()],
            half_sizes=[result["obb_half"].tolist()],
            quaternions=[rr.Quaternion(xyzw=result["obb_quat"].tolist())],
            colors=[colour],
            labels=[label],
        ), static=True)

    # Per-frame timeline: raw camera feed + moving camera in 3D (no overlay)
    from src.scene_query.geometry import load_frames_from_dir
    frames = load_frames_from_dir(images_dir)

    intr       = np.load(scene_dir / "intrinsics.npz")
    intrinsics = intr["intrinsics"]
    image_size = intr["image_size"]

    for i, frame in enumerate(frames):
        rr.set_time("frame", sequence=i)
        rr.log("camera/rgb", rr.Image(frame))

        pose           = poses_arr[i]
        K              = intrinsics[i] if i < len(intrinsics) else intrinsics[-1]
        H_d, W_d       = int(image_size[0]), int(image_size[1])
        H_orig, W_orig = frame.shape[:2]
        sx, sy         = W_orig / W_d, H_orig / H_d

        rr.log("world/camera", rr.Transform3D(
            translation=pose[:3, 3].tolist(),
            mat3x3=pose[:3, :3].tolist(),
        ))
        rr.log("world/camera", rr.Pinhole(
            focal_length=[float(K[0, 0]) * sx, float(K[1, 1]) * sy],
            principal_point=[float(K[0, 2]) * sx, float(K[1, 2]) * sy],
            width=W_orig, height=H_orig,
        ))

    print(f"Saved -> {save_rrd}")


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
        query_type = "object"
        if len(args.query) == 1:
            label  = args.query[0].lower().removeprefix("find the ").removeprefix("find ").strip()
            result = query_object(scene, label)
            print(f"Label      : {result['label']}")
            print(f"Centroid   : {result['centroid']}")
            print(f"Bbox       : {result['bbox_min']} -> {result['bbox_max']}")
            print(f"Points     : {result['n_points']}")
            print(f"Confidence : {result['confidence']:.2f}")
        else:
            results = []
            for q in args.query:
                label = q.lower().removeprefix("find the ").removeprefix("find ").strip()
                try:
                    r = query_object(scene, label)
                    results.append(r)
                    print(f"Label      : {r['label']}")
                    print(f"Centroid   : {r['centroid']}")
                    print(f"Bbox       : {r['bbox_min']} -> {r['bbox_max']}")
                    print(f"Points     : {r['n_points']}")
                    print(f"Confidence : {r['confidence']:.2f}")
                    print()
                except KeyError as e:
                    print(f"Warning: {e}")
            result = results

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
