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
    p.add_argument("--no-vis", action="store_true", help="Skip Rerun visualisation")
    return p.parse_args()


def visualise(scene: dict, result, query_type: str) -> None:
    """Log scene point cloud and query result to Rerun."""
    import rerun as rr

    rr.init("scene-query", spawn=True)
    rr.log("scene/points", rr.Points3D(scene["xyz"], colors=scene["rgb"]))

    if query_type == "object":
        rr.log("query/centroid", rr.Points3D([result["centroid"]], radii=0.05))
        rr.log(
            "query/bbox",
            rr.Boxes3D(mins=result["bbox_min"], maxs=result["bbox_max"]),
        )
    elif query_type == "free_space":
        rr.log("query/free_space", rr.Points3D(result, colors=[0, 200, 100]))
    elif query_type == "reachable":
        centroids = [r["centroid"] for r in result]
        if centroids:
            rr.log("query/reachable", rr.Points3D(centroids, radii=0.05))


def main() -> None:
    args = parse_args()
    scene = load_scene(args.scene_dir)

    if args.free_space:
        result = query_free_space(scene, floor_z=args.floor_z)
        query_type = "free_space"
        print(f"Traversable voxels: {len(result)}")

    elif args.reachable_from:
        base = np.array(args.reachable_from, dtype=np.float32)
        result = query_reachable(scene, base_xyz=base, reach=args.reach)
        query_type = "reachable"
        print(f"Reachable objects ({len(result)}):")
        for r in result:
            print(f"  {r['label']:20s}  centroid={r['centroid']}  n={r['n_points']}")

    elif args.query:
        label = args.query.lower().removeprefix("find the ").removeprefix("find ").strip()
        result = query_object(scene, label)
        query_type = "object"
        print(f"Label      : {result['label']}")
        print(f"Centroid   : {result['centroid']}")
        print(f"Bbox       : {result['bbox_min']} → {result['bbox_max']}")
        print(f"Points     : {result['n_points']}")
        print(f"Confidence : {result['confidence']:.2f}")

    else:
        print("Provide a query string, --free-space, or --reachable-from.")
        return

    if not args.no_vis:
        visualise(scene, result, query_type)


if __name__ == "__main__":
    main()
