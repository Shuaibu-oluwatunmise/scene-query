"""Entry point: query a scene directory with GSAM2 open-vocabulary detection."""
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
    _clean_and_fit_obb,
    _gravity_aligned_obb,
)

# ---------------------------------------------------------------------------
# Environment presets
# ---------------------------------------------------------------------------

PRESETS: dict[str, list[str]] = {
    "office":    ["keyboard", "mouse", "chair", "laptop"],
    "home":      ["sofa", "chair", "table", "tv", "lamp", "plant", "mug",
                  "remote control", "book", "bed", "cushion"],
    "classroom": ["chair", "desk", "whiteboard", "projector", "laptop",
                  "backpack", "book", "pen", "monitor"],
    "kitchen":   ["microwave", "refrigerator", "oven", "sink", "cup", "plate",
                  "bowl", "bottle", "knife", "kettle", "toaster"],
    "warehouse": ["box", "pallet", "shelf", "forklift", "crate", "barrel",
                  "ladder", "trolley"],
}


def _label_colour(label: str) -> list[int]:
    import hashlib
    import colorsys
    h = int(hashlib.md5(label.encode()).hexdigest(), 16)
    r, g, b = colorsys.hsv_to_rgb((h % 360) / 360.0, 0.75, 0.95)
    return [int(r * 255), int(g * 255), int(b * 255)]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("scene_dir", type=Path, help="Scene directory from reconstruct.py")
    p.add_argument("query", nargs="*",
                   help='Objects to find, e.g. "bulldozer" or "monitor chair"')
    p.add_argument("--preset", choices=list(PRESETS.keys()), default=None,
                   help="Scan for all common objects in this environment type")
    p.add_argument("--images", type=Path, default=None,
                   help="Original images directory (default: <scene_dir>/frames)")
    p.add_argument("--no-vis", action="store_true", help="Skip all visualisation")
    p.add_argument("--save-rrd", type=Path, default=None,
                   help="Save Rerun recording to this .rrd file")
    p.add_argument("--box-threshold", type=float, default=0.3,
                   help="Grounding DINO box confidence threshold")
    p.add_argument("--text-threshold", type=float, default=0.25,
                   help="Grounding DINO text similarity threshold")
    return p.parse_args()


# ---------------------------------------------------------------------------
# GSAM2 query: single pass for all labels, then per-label lift + OBB
# ---------------------------------------------------------------------------

def _run_gsam2_query(
    scene: dict,
    scene_dir: Path,
    labels: list[str],
    images_dir: Path,
    box_threshold: float = 0.3,
    text_threshold: float = 0.25,
) -> list[dict]:
    """Segment all labels in one GSAM2 pass, back-project, fit OBBs."""
    from src.scene_query.semantics import load_gsam2_models, segment_frames_gsam2
    from src.scene_query.geometry import load_frames_from_dir
    from src.scene_query.lift import lift_masks

    frames = load_frames_from_dir(images_dir)

    print("  Loading Grounding DINO + SAM2 models...")
    gdino, sam = load_gsam2_models(device=_DEVICE)

    print(f"  Running GSAM2 on {len(frames)} frames for: {labels}")
    masks_per_frame = segment_frames_gsam2(
        frames, labels, gdino, sam,
        device=_DEVICE,
        box_threshold=box_threshold,
        text_threshold=text_threshold,
    )

    depth_path = scene_dir / "depth.npz"
    if not depth_path.exists():
        raise FileNotFoundError(
            f"depth.npz not found in {scene_dir} — re-run reconstruct.py to regenerate."
        )
    depth_maps = np.load(depth_path)["depth"].astype(np.float32)
    intr       = np.load(scene_dir / "intrinsics.npz")
    intrinsics = intr["intrinsics"]
    poses      = scene["poses"]

    results = []
    for label in labels:
        # Filter masks_per_frame to detections for this label only
        label_masks = [
            [d for d in frame_dets if d["label"] == label]
            for frame_dets in masks_per_frame
        ]

        total_dets = sum(len(fd) for fd in label_masks)
        if total_dets == 0:
            print(f"  No detections for '{label}' — skipping")
            continue

        lifted = lift_masks(
            frames=frames,
            depth_maps=depth_maps,
            intrinsics=intrinsics,
            poses=poses,
            masks_per_frame=label_masks,
        )
        pts = lifted["xyz"]

        if len(pts) < 10:
            print(f"  Too few 3D points for '{label}' ({len(pts)}) — skipping")
            continue

        pts_clean = _clean_and_fit_obb(pts, poses)
        if len(pts_clean) < 3:
            pts_clean = pts

        center = pts_clean.mean(axis=0).astype(np.float32)
        obb    = _gravity_aligned_obb(pts_clean, poses)

        results.append({
            "label":            label,
            "centroid":         center,
            "bbox_min":         pts_clean.min(axis=0).astype(np.float32),
            "bbox_max":         pts_clean.max(axis=0).astype(np.float32),
            "obb_center":       obb["center"],
            "obb_half":         obb["half"],
            "obb_quat":         obb["quat"],
            "n_points":         len(pts),
            "confidence":       float(np.mean([
                d["confidence"]
                for fd in label_masks for d in fd
            ])) if total_dets > 0 else 0.5,
            "gsam2_detections": masks_per_frame,  # full set; visualiser filters by label
        })
        print(f"  {label}: {len(pts):,} pts lifted, {len(pts_clean):,} after clean")

    return results


# ---------------------------------------------------------------------------
# Visualisation helpers
# ---------------------------------------------------------------------------

def _visualise_focused(
    scene: dict,
    result: dict,
    save_rrd: Path,
    images_dir: Path,
    scene_dir: Path,
) -> None:
    """4-panel recording for a single queried object."""
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

    bg_xyz    = scene["scene_xyz"]
    bg_rgb    = scene["scene_rgb"]
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

    from src.scene_query.geometry import load_frames_from_dir
    frames     = load_frames_from_dir(images_dir)
    detections = result.get("gsam2_detections") or []

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
        hits = [d for d in frame_dets if d.get("label") == label and "bbox" in d]
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


def _visualise_multi(
    scene: dict,
    results: list[dict],
    save_rrd: Path,
    images_dir: Path,
    scene_dir: Path,
) -> None:
    """4-panel recording for multiple queried objects."""
    import rerun as rr
    import rerun.blueprint as rrb
    import rerun.blueprint.archetypes as ba

    labels         = [r["label"] for r in results]
    colours        = [_label_colour(lbl) for lbl in labels]
    colour_by_lbl  = dict(zip(labels, colours))
    label_set      = set(labels)
    name           = " + ".join(labels)

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

    bg_xyz    = scene["scene_xyz"]
    bg_rgb    = scene["scene_rgb"]
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

    # All results share the same GSAM2 run — use the first result's detections
    detections = results[0].get("gsam2_detections") if results else []
    detections = detections or []

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
            translation=pose[:3, 3].tolist(), mat3x3=pose[:3, :3].tolist(),
        ))
        rr.log("world/camera", rr.Pinhole(
            focal_length=[float(K[0, 0]) * sx, float(K[1, 1]) * sy],
            principal_point=[float(K[0, 2]) * sx, float(K[1, 2]) * sy],
            width=W_orig, height=H_orig,
        ))

        frame_dets = detections[i] if i < len(detections) else []
        hits = [d for d in frame_dets if d.get("label") in label_set and "bbox" in d]
        if hits:
            rr.log("camera/query_dets", rr.Boxes2D(
                mins=[[d["bbox"][0], d["bbox"][1]] for d in hits],
                sizes=[[d["bbox"][2] - d["bbox"][0], d["bbox"][3] - d["bbox"][1]] for d in hits],
                colors=[colour_by_lbl.get(d["label"], [255, 255, 0]) for d in hits],
                labels=[f"{d['label']} {d['confidence']:.2f}" for d in hits],
            ))
        else:
            rr.log("camera/query_dets", rr.Clear(recursive=False))

    print(f"Saved -> {save_rrd}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    # Resolve images directory
    images_dir = args.images
    if images_dir is None:
        candidate = args.scene_dir / "frames"
        if candidate.is_dir():
            images_dir = candidate
    if images_dir is None or not images_dir.is_dir():
        raise SystemExit(
            f"No images directory found. Pass --images <dir> or ensure "
            f"{args.scene_dir / 'frames'} exists."
        )

    # Determine labels to query
    if args.preset:
        labels = PRESETS[args.preset]
        print(f"Preset '{args.preset}': querying {len(labels)} object classes")
    elif args.query:
        # Support both "find the bulldozer, cup, bottle" and separate args
        raw = []
        for q in args.query:
            raw.extend([x.strip() for x in q.split(",") if x.strip()])
        labels = [
            q.lower().removeprefix("find the ").removeprefix("find ").strip()
            for q in raw
        ]
        print(f"Querying: {labels}")
    else:
        print("No query specified. Available presets:")
        for name, objs in PRESETS.items():
            print(f"  --preset {name:10s}  {', '.join(objs[:5])}...")
        print("\nOr pass object names directly: python query.py <scene_dir> bulldozer")
        return

    scene = load_scene(args.scene_dir)

    print(f"\nRunning GSAM2 query...")
    results = _run_gsam2_query(
        scene,
        args.scene_dir,
        labels,
        images_dir,
        box_threshold=args.box_threshold,
        text_threshold=args.text_threshold,
    )

    if not results:
        print("No objects detected.")
        return

    print(f"\nResults ({len(results)} objects found):")
    for r in results:
        print(f"  {r['label']:20s}  centroid={r['centroid']}  "
              f"n={r['n_points']:,}  conf={r['confidence']:.2f}")

    if args.no_vis:
        return

    if args.save_rrd is None:
        print("\nPass --save-rrd <path.rrd> to save a Rerun recording.")
        return

    print(f"\nBuilding Rerun recording -> {args.save_rrd}")
    if len(results) == 1:
        _visualise_focused(scene, results[0], args.save_rrd, images_dir, args.scene_dir)
    else:
        _visualise_multi(scene, results, args.save_rrd, images_dir, args.scene_dir)


if __name__ == "__main__":
    main()
