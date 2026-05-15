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
        help="Original images directory (enables frame timeline in focused query rrd)",
    )
    return p.parse_args()


def _label_colours(scene: dict) -> np.ndarray:
    return np.array([_label_colour(str(l)) for l in scene["label"]], dtype=np.uint8)


def visualise(
    scene: dict,
    result=None,
    query_type: str | None = None,
    save_rrd: Path | None = None,
    images_dir: Path | None = None,
    scene_dir: Path | None = None,
) -> None:
    """Log the full scene + optional query result to Rerun.

    When save_rrd is given and query_type is "object", generates a focused
    recording showing only the queried label highlighted (rest dimmed).
    Otherwise logs the full multi-label scene.
    """
    import rerun as rr

    if save_rrd is not None and query_type == "object" and result is not None:
        _visualise_focused(scene, result, save_rrd, images_dir, scene_dir)
        return

    # --- Standard: full scene visualisation ---
    rr.init("scene-query", spawn=False)

    if save_rrd:
        rr.save(str(save_rrd))
        print(f"Recording to {save_rrd} ...")
    else:
        rr.spawn()

    rr.log(
        "world/scene",
        rr.Points3D(scene["xyz"], colors=_label_colours(scene), radii=0.003),
    )

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
                rr.Transform3D(
                    translation=pose[:3, 3].tolist(),
                    mat3x3=pose[:3, :3].tolist(),
                ),
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
                rr.Points3D(
                    [result["centroid"].tolist()],
                    colors=[[255, 255, 0]],
                    radii=0.05,
                ),
            )

        elif query_type == "free_space":
            rr.log(
                "world/query/free_space",
                rr.Points3D(result, colors=[[0, 220, 100]], radii=0.025),
            )

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


def _visualise_focused(
    scene: dict,
    result: dict,
    save_rrd: Path,
    images_dir: Path | None,
    scene_dir: Path | None,
) -> None:
    """Focused rrd: queried label bright, everything else dimmed.

    3-panel 3D layout:
      Scene context | Query: <label> isolated | Result with bbox

    If --images dir is provided (and intrinsics.npz exists in scene_dir):
      adds a bottom 2D camera feed panel with per-frame frustum in 3D.
    """
    import rerun as rr
    import rerun.blueprint as rrb

    label  = result["label"]
    colour = _label_colour(label)

    has_frames     = images_dir is not None and images_dir.exists()
    has_intrinsics = (
        scene_dir is not None and
        (scene_dir / "intrinsics.npz").exists()
    )

    cam_contents = ["world/camera", "world/camera/**", "world/cameras/**"]

    top_row = rrb.Horizontal(
        rrb.Spatial3DView(
            name="Scene context",
            contents=["world/scene_bg"] + cam_contents,
        ),
        rrb.Spatial3DView(
            name=f"Query: {label}",
            contents=["world/scene_bg", "world/query_pts"] + cam_contents,
        ),
        rrb.Spatial3DView(
            name="Result",
            contents=["world/scene_bg", "world/query_pts", "world/query_bbox"] + cam_contents,
        ),
    )

    if has_frames:
        layout = rrb.Vertical(
            top_row,
            rrb.Spatial2DView(name="Camera feed", contents=["camera/rgb"]),
            row_shares=[3, 2],
        )
    else:
        layout = top_row

    blueprint = rrb.Blueprint(layout, collapse_panels=True, auto_views=False)

    rr.init("scene-query", spawn=False, default_blueprint=blueprint)
    rr.save(str(save_rrd))

    # Static: full scene background (all points, dark gray for context)
    bg_colours = np.full((len(scene["xyz"]), 3), 70, dtype=np.uint8)
    rr.log("world/scene_bg", rr.Points3D(scene["xyz"], colors=bg_colours, radii=0.002), static=True)

    # Static: queried label points, full brightness
    idxs = np.array(scene["label_index"][label], dtype=np.int64)
    pts  = scene["xyz"][idxs]
    rr.log(
        "world/query_pts",
        rr.Points3D(
            pts,
            colors=np.tile(colour, (len(pts), 1)).astype(np.uint8),
            radii=0.005,
        ),
        static=True,
    )

    # Static: tight bbox for queried label
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

    # Static: camera trajectory markers (one axis per pose)
    if scene.get("poses") is not None:
        for i, pose in enumerate(scene["poses"]):
            rr.log(
                f"world/cameras/{i:02d}",
                rr.Transform3D(
                    translation=pose[:3, 3].tolist(),
                    mat3x3=pose[:3, :3].tolist(),
                ),
                static=True,
            )

    # Per-frame timeline: camera frustum + RGB feed
    if has_frames:
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
            rr.log("camera/rgb", rr.Image(frame))

            if intrinsics is not None and poses is not None and i < len(poses):
                pose  = poses[i]
                K     = intrinsics[i] if i < len(intrinsics) else intrinsics[-1]
                H_d, W_d   = int(image_size[0]), int(image_size[1])
                H_orig, W_orig = frame.shape[:2]
                sx, sy = W_orig / W_d, H_orig / H_d
                fx = float(K[0, 0]) * sx
                fy = float(K[1, 1]) * sy
                cx = float(K[0, 2]) * sx
                cy = float(K[1, 2]) * sy

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
                rr.log("world/camera/image", rr.Image(frame))

    print(f"Saved -> {save_rrd}")


def main() -> None:
    args = parse_args()
    scene = load_scene(args.scene_dir)

    result     = None
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
