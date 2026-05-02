from __future__ import annotations

import argparse
from pathlib import Path

from pxr import Usd, UsdGeom


KEYWORDS = ("grasp", "support", "slot", "top", "bottom", "center", "cap", "button")


def _root_prim(stage: Usd.Stage):
    default_prim = stage.GetDefaultPrim()
    if default_prim:
        return default_prim
    for child in stage.GetPseudoRoot().GetChildren():
        return child
    return None


def _print_bounds(stage: Usd.Stage, root_prim, applied_scale: float) -> None:
    bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_], useExtentsHint=True)
    world_bound = bbox_cache.ComputeWorldBound(root_prim)
    aligned = world_bound.ComputeAlignedBox()
    min_pt = aligned.GetMin()
    max_pt = aligned.GetMax()
    size = max_pt - min_pt
    meters_per_unit = UsdGeom.GetStageMetersPerUnit(stage)
    up_axis = UsdGeom.GetStageUpAxis(stage)
    effective_meters_per_unit = meters_per_unit * applied_scale
    print(f"  bounds_min=({min_pt[0]:.5f}, {min_pt[1]:.5f}, {min_pt[2]:.5f})")
    print(f"  bounds_max=({max_pt[0]:.5f}, {max_pt[1]:.5f}, {max_pt[2]:.5f})")
    print(f"  size_xyz=({size[0]:.5f}, {size[1]:.5f}, {size[2]:.5f})")
    print(f"  bottom_z={min_pt[2]:.5f}")
    print(f"  top_z={max_pt[2]:.5f}")
    print(f"  meters_per_unit={meters_per_unit}")
    print(f"  up_axis={up_axis}")
    print(f"  applied_scale={applied_scale}")
    print(f"  effective_meters_per_unit={effective_meters_per_unit}")
    print(
        "  metric_size_xyz=("
        f"{size[0] * effective_meters_per_unit:.5f}, "
        f"{size[1] * effective_meters_per_unit:.5f}, "
        f"{size[2] * effective_meters_per_unit:.5f})"
    )
    print(f"  metric_bottom_z={min_pt[2] * effective_meters_per_unit:.5f}")
    print(f"  metric_top_z={max_pt[2] * effective_meters_per_unit:.5f}")


def inspect_asset(path: Path, max_prims: int, applied_scale: float) -> None:
    print(f"FILE {path}")
    stage = Usd.Stage.Open(str(path))
    if stage is None:
        print("  failed_open")
        return

    root_prim = _root_prim(stage)
    if root_prim is None:
        print("  no_root_prim")
        return

    print(f"  root={root_prim.GetPath()}")
    _print_bounds(stage, root_prim, applied_scale)

    keyword_hits = []
    prim_count = 0
    for prim in stage.Traverse():
        prim_count += 1
        prim_name = prim.GetName().lower()
        if any(keyword in prim_name for keyword in KEYWORDS):
            keyword_hits.append((str(prim.GetPath()), prim.GetTypeName()))

    print(f"  prim_count={prim_count}")
    print("  keyword_hits:")
    if keyword_hits:
        for prim_path, prim_type in keyword_hits:
            print(f"    {prim_path} [{prim_type}]")
    else:
        print("    <none>")

    print("  prim_tree:")
    for idx, prim in enumerate(stage.Traverse()):
        if idx >= max_prims:
            print(f"    ... truncated after {max_prims} prims")
            break
        print(f"    {prim.GetPath()} [{prim.GetTypeName()}]")


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect USD assets for support, grasp, and bounds anchors.")
    parser.add_argument("paths", nargs="+", help="USD asset files to inspect")
    parser.add_argument("--max-prims", type=int, default=80)
    parser.add_argument("--applied-scale", type=float, default=1.0)
    args = parser.parse_args()

    for raw_path in args.paths:
        inspect_asset(Path(raw_path).expanduser().resolve(), args.max_prims, args.applied_scale)


if __name__ == "__main__":
    main()
