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

# One colour per semantic label — matches the PLY export palette
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
    return p.parse_args()


def _label_colours(scene: dict) -> np.ndarray:
    return np.array(
        [_PALETTE.get(str(l), _DEFAULT_COLOUR) for l in scene["label"]],
        dtype=np.uint8,
    )


def visualise(
    scene: dict,
    result=None,
    query_type: str | None = None,
    save_rrd: Path | None = None,
) -> None:
    """Log the full scene + optional query result to Rerun.

    If save_rrd is given, writes a .rrd file (no viewer spawned).
    Otherwise spawns the Rerun desktop viewer.
    """
    import rerun as rr

    rr.init("scene-query", spawn=False)

    if save_rrd:
        rr.save(str(save_rrd))
        print(f"Recording to {save_rrd} ...")
    else:
        rr.spawn()

    # --- Full point cloud, coloured by semantic label ---
    rr.log(
        "world/scene",
        rr.Points3D(scene["xyz"], colors=_label_colours(scene), radii=0.003),
    )

    # Per-label sub-entities so the viewer blueprint can toggle each label
    for lbl, idxs in scene["label_index"].items():
        idx_arr = np.array(idxs, dtype=np.int64)
        colour  = _PALETTE.get(lbl, _DEFAULT_COLOUR)
        rr.log(
            f"world/labels/{lbl}",
            rr.Points3D(
                scene["xyz"][idx_arr],
                colors=np.tile(colour, (len(idx_arr), 1)).astype(np.uint8),
                radii=0.003,
            ),
        )

    # --- Camera trajectory (one 3-D axis per pose) ---
    if scene.get("poses") is not None:
        for i, pose in enumerate(scene["poses"]):
            rr.log(
                f"world/cameras/{i:02d}",
                rr.Transform3D(
                    translation=pose[:3, 3].tolist(),
                    mat3x3=pose[:3, :3].tolist(),
                ),
            )

    # --- Query result overlay ---
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
        visualise(scene, result, query_type, save_rrd=args.save_rrd)


if __name__ == "__main__":
    main()
