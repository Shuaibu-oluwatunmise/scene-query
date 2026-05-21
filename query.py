"""Entry point: query a labelled scene directory."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

_DEVICE = "cuda"
try:
    import torch
    if not torch.cuda.is_available():
        _DEVICE = "cpu"
except ImportError:
    _DEVICE = "cpu"

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

def _load_detections(scene_dir: Path) -> list[list[dict]]:
    """Load per-frame bbox detections saved by reconstruct.py."""
    import json
    dets_path = scene_dir / "detections.json"
    if not dets_path.exists():
        return []
    with open(dets_path) as f:
        return json.load(f)


def _visualise_focused(
    scene: dict,
    result: dict,
    save_rrd: Path,
    images_dir: Path,
    scene_dir: Path,
) -> None:
    """4-panel recording for a single queried object.

    Panel 1 (timeline): Raw camera feed
    Panel 2 (timeline): 3D reconstruction (cam-locked)
    Panel 3 (timeline): 2D bboxes filtered to the queried label
    Panel 4 (static):   Photo cloud + OBB of queried object
    """
    import rerun as rr
    import rerun.blueprint as rrb
    import rerun.blueprint.archetypes as ba

    label  = result["label"]
    colour = _label_colour(label)

    poses_arr = scene["poses"]
    cam0      = poses_arr[0]
    cam0_pos  = cam0[:3, 3].tolist()
    cam0_fwd  = (cam0[:3, 3] + cam0[:3, 2]).tolist()
    cam0_up   = (-cam0[:3, 1]).tolist()
    bg        = [20, 20, 20]

    first_cam_eye = ba.EyeControls3D(
        position=cam0_pos, look_target=cam0_fwd, eye_up=cam0_up,
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
                rrb.Spatial2DView(
                    name=f"Detections — {label}",
                    contents=["camera/rgb", "camera/query_dets"],
                ),
                rrb.Spatial3DView(
                    name=f"{label} — bbox",
                    contents=["world/photo", "world/query_bbox"],
                    eye_controls=first_cam_eye,
                    background=bg,
                ),
            ),
        ),
        rrb.TimePanel(playback_speed=0.1, loop_mode=rrb.components.LoopMode.All),
        collapse_panels=True,
        auto_views=False,
    )

    rr.init("scene-query", spawn=False)
    rr.save(str(save_rrd))
    rr.send_blueprint(blueprint, make_active=True, make_default=True)
    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Y_DOWN, static=True)

    # Static entities
    bg_xyz    = scene["scene_xyz"] if scene.get("scene_xyz") is not None else scene["xyz"]
    bg_rgb    = scene["scene_rgb"] if scene.get("scene_rgb") is not None else scene["rgb"]
    bg_rgb_u8 = (np.clip(bg_rgb, 0, 1) * 255).astype(np.uint8)

    rr.log("world/photo", rr.Points3D(
        bg_xyz, colors=bg_rgb_u8, radii=rr.Radius.ui_points(2.0),
    ), static=True)

    rr.log("world/query_bbox", rr.Boxes3D(
        centers=[result["obb_center"].tolist()],
        half_sizes=[result["obb_half"].tolist()],
        quaternions=[rr.Quaternion(xyzw=result["obb_quat"].tolist())],
        colors=[colour],
        labels=[label],
    ), static=True)

    # Per-frame timeline
    from src.scene_query.geometry import load_frames_from_dir
    frames     = load_frames_from_dir(images_dir)
    detections = _load_detections(scene_dir)

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
            translation=pose[:3, 3].tolist(), mat3x3=pose[:3, :3].tolist(),
        ))
        rr.log("world/camera", rr.Pinhole(
            focal_length=[float(K[0, 0]) * sx, float(K[1, 1]) * sy],
            principal_point=[float(K[0, 2]) * sx, float(K[1, 2]) * sy],
            width=W_orig, height=H_orig,
        ))

        # Panel 3: bboxes for this label only
        frame_dets = detections[i] if i < len(detections) else []
        hits = [d for d in frame_dets if d["label"] == label]
        if hits:
            rr.log("camera/query_dets", rr.Boxes2D(
                mins=[[d["bbox"][0], d["bbox"][1]] for d in hits],
                sizes=[[d["bbox"][2] - d["bbox"][0], d["bbox"][3] - d["bbox"][1]] for d in hits],
                colors=[colour] * len(hits),
                labels=[f"{label} {d['confidence']:.2f}" for d in hits],
            ))
        else:
            rr.log("camera/query_dets", rr.Clear(recursive=False))

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

    Panel 1 (timeline): Raw camera feed
    Panel 2 (timeline): 3D reconstruction (cam-locked)
    Panel 3 (timeline): 2D bboxes filtered to queried labels
    Panel 4 (static):   Photo cloud + OBBs for all queried objects
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
        position=cam0_pos, look_target=cam0_fwd, eye_up=cam0_up,
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
                rrb.Spatial2DView(
                    name=f"Detections — {name}",
                    contents=["camera/rgb", "camera/query_dets"],
                ),
                rrb.Spatial3DView(
                    name="Bounding Boxes",
                    contents=["world/photo", "world/query_bbox/**"],
                    eye_controls=first_cam_eye,
                    background=bg,
                ),
            ),
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

    for result, colour in zip(results, colours):
        lbl = result["label"]
        rr.log(f"world/query_bbox/{lbl}", rr.Boxes3D(
            centers=[result["obb_center"].tolist()],
            half_sizes=[result["obb_half"].tolist()],
            quaternions=[rr.Quaternion(xyzw=result["obb_quat"].tolist())],
            colors=[colour],
            labels=[lbl],
        ), static=True)

    label_set = set(labels)
    colour_by_label = {lbl: col for lbl, col in zip(labels, colours)}

    # Per-frame timeline
    from src.scene_query.geometry import load_frames_from_dir
    frames     = load_frames_from_dir(images_dir)
    detections = _load_detections(scene_dir)

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
            translation=pose[:3, 3].tolist(), mat3x3=pose[:3, :3].tolist(),
        ))
        rr.log("world/camera", rr.Pinhole(
            focal_length=[float(K[0, 0]) * sx, float(K[1, 1]) * sy],
            principal_point=[float(K[0, 2]) * sx, float(K[1, 2]) * sy],
            width=W_orig, height=H_orig,
        ))

        frame_dets = detections[i] if i < len(detections) else []
        hits = [d for d in frame_dets if d["label"] in label_set]
        if hits:
            rr.log("camera/query_dets", rr.Boxes2D(
                mins=[[d["bbox"][0], d["bbox"][1]] for d in hits],
                sizes=[[d["bbox"][2] - d["bbox"][0], d["bbox"][3] - d["bbox"][1]] for d in hits],
                colors=[colour_by_label[d["label"]] for d in hits],
                labels=[f"{d['label']} {d['confidence']:.2f}" for d in hits],
            ))
        else:
            rr.log("camera/query_dets", rr.Clear(recursive=False))

    print(f"Saved -> {save_rrd}")


# ---------------------------------------------------------------------------
# GSAM2 open-vocabulary fallback
# ---------------------------------------------------------------------------

def _gsam2_fallback(scene: dict, scene_dir: Path, label: str, images_dir: Path) -> dict:
    """Query an arbitrary label via Grounding DINO + SAM2 + point-cloud projection.

    Projects the saved VGGT background cloud into each frame, selects points
    that fall inside the GSAM2 mask, then fits an OBB — no VGGT re-run needed.
    """
    from src.scene_query.semantics import load_gsam2_models, segment_frames_gsam2
    from src.scene_query.geometry import load_frames_from_dir
    from src.scene_query.query_engine import _clean_and_fit_obb, _gravity_aligned_obb

    print(f"  '{label}' not in scene — running Grounding DINO + SAM2 fallback...")

    frames = load_frames_from_dir(images_dir)

    print("  Loading GSAM2 models...")
    gdino, sam = load_gsam2_models(device=_DEVICE)

    print(f"  Running GSAM2 on {len(frames)} frames...")
    masks_per_frame = segment_frames_gsam2(frames, [label], gdino, sam, device=_DEVICE)

    # Load background point cloud and camera data
    sc         = np.load(scene_dir / "scene_cloud.npz")
    xyz_bg     = sc["xyz"]
    intr       = np.load(scene_dir / "intrinsics.npz")
    intrinsics = intr["intrinsics"]
    image_size = intr["image_size"]
    poses      = scene["poses"]
    H_d, W_d   = int(image_size[0]), int(image_size[1])

    # Subsample to keep projection fast
    MAX_PTS = 300_000
    if len(xyz_bg) > MAX_PTS:
        sub    = np.random.choice(len(xyz_bg), MAX_PTS, replace=False)
        xyz_bg = xyz_bg[sub]

    collected = np.zeros(len(xyz_bg), dtype=bool)

    for i, (frame, frame_dets) in enumerate(zip(frames, masks_per_frame)):
        if not frame_dets:
            continue
        H_orig, W_orig = frame.shape[:2]
        combined_mask  = np.zeros((H_orig, W_orig), dtype=bool)
        for det in frame_dets:
            combined_mask |= det["mask"]
        if not combined_mask.any():
            continue

        pose = poses[i]
        K    = intrinsics[i] if i < len(intrinsics) else intrinsics[-1]
        sx, sy = W_orig / W_d, H_orig / H_d
        fx = K[0, 0] * sx;  fy = K[1, 1] * sy
        cx_k = K[0, 2] * sx; cy_k = K[1, 2] * sy

        R, t  = pose[:3, :3], pose[:3, 3]
        p_cam = (xyz_bg - t) @ R           # (N, 3) world → camera

        valid = p_cam[:, 2] > 0.01
        z     = np.where(valid, p_cam[:, 2], 1.0)
        xi    = (fx * p_cam[:, 0] / z + cx_k).astype(np.int32)
        yi    = (fy * p_cam[:, 1] / z + cy_k).astype(np.int32)
        valid &= (xi >= 0) & (xi < W_orig) & (yi >= 0) & (yi < H_orig)

        xc = np.clip(xi, 0, W_orig - 1)
        yc = np.clip(yi, 0, H_orig - 1)
        collected |= valid & combined_mask[yc, xc]

    pts = xyz_bg[collected]
    if len(pts) < 10:
        raise KeyError(f"GSAM2 found no 3D points for '{label}' in the scene.")

    print(f"  Collected {len(pts):,} points via GSAM2 projection")

    pts_clean = _clean_and_fit_obb(pts, poses)
    if len(pts_clean) < 3:
        pts_clean = pts

    center = pts_clean.mean(axis=0).astype(np.float32)
    obb    = _gravity_aligned_obb(pts_clean, poses)

    return {
        "label":      label,
        "centroid":   center,
        "bbox_min":   pts_clean.min(axis=0).astype(np.float32),
        "bbox_max":   pts_clean.max(axis=0).astype(np.float32),
        "obb_center": obb["center"],
        "obb_half":   obb["half"],
        "obb_quat":   obb["quat"],
        "n_points":   len(pts),
        "confidence": 0.5,
        "indices":    np.where(collected)[0],
    }


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
        images_dir_for_fallback = args.images or (args.scene_dir / "frames")

        def _query_with_fallback(label: str) -> dict:
            try:
                return query_object(scene, label)
            except KeyError:
                return _gsam2_fallback(scene, args.scene_dir, label, images_dir_for_fallback)

        if len(args.query) == 1:
            label  = args.query[0].lower().removeprefix("find the ").removeprefix("find ").strip()
            result = _query_with_fallback(label)
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
                    r = _query_with_fallback(label)
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

    else:
        # No query argument — show all detected objects with OBBs
        query_type = "object"
        all_results = []
        for label in sorted(scene["label_index"].keys()):
            try:
                r = query_object(scene, label)
                all_results.append(r)
            except (KeyError, ValueError):
                pass
        if all_results:
            print(f"All objects in scene ({len(all_results)}):")
            for r in all_results:
                print(f"  {r['label']:20s}  centroid={r['centroid']}  n={r['n_points']}")
            result = all_results
        else:
            print("No labelled objects found in scene.")

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
