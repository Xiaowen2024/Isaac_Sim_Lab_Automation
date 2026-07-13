#!/usr/bin/env python3
"""Inspect canonical scene assets without launching Isaac Sim."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from pxr import Gf, Usd, UsdGeom


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ASSET_DIR = PROJECT_ROOT / "assets"

TABLE_SIZE_M = (1.20, 0.90, 0.04)
TABLE_CENTER_M = (0.35, 0.0, 0.76)
TABLE_TOP_Z_M = TABLE_CENTER_M[2] + 0.5 * TABLE_SIZE_M[2]


@dataclass(frozen=True)
class SceneAsset:
    name: str
    path: Path
    root_pos_m: tuple[float, float, float]
    scale: tuple[float, float, float] = (1.0, 1.0, 1.0)


SCENE_ASSETS = (
    SceneAsset(
        name="robot",
        path=ASSET_DIR / "xarm6_with_gripper_contact_refined.usd",
        root_pos_m=(0.0, 0.0, TABLE_TOP_Z_M + 0.0501),
    ),
    SceneAsset(
        name="holder",
        path=ASSET_DIR / "4_50ml_conical_holder_contact_refined.usd",
        root_pos_m=(0.43, -0.12, TABLE_TOP_Z_M + 0.004),
        scale=(0.00114, 0.00114, 0.001),
    ),
    SceneAsset(
        name="vortexer",
        path=ASSET_DIR / "vortexer_contact_refined.usd",
        root_pos_m=(0.43, 0.14, TABLE_TOP_Z_M + 0.010),
        scale=(0.001, 0.001, 0.001),
    ),
    SceneAsset(
        name="tube",
        path=ASSET_DIR / "autobio_50ml_tube_contact_refined.usda",
        root_pos_m=(0.43, -0.12, TABLE_TOP_Z_M + 0.14),
    ),
)


def _format_vec(vec) -> str:
    return f"({vec[0]: .4f}, {vec[1]: .4f}, {vec[2]: .4f})"


def _scaled_range(range_3d: Gf.Range3d, scale: tuple[float, float, float]) -> Gf.Range3d:
    min_v = range_3d.GetMin()
    max_v = range_3d.GetMax()
    corners = [
        Gf.Vec3d(x, y, z)
        for x in (min_v[0], max_v[0])
        for y in (min_v[1], max_v[1])
        for z in (min_v[2], max_v[2])
    ]
    scaled = [Gf.Vec3d(corner[0] * scale[0], corner[1] * scale[1], corner[2] * scale[2]) for corner in corners]
    result = Gf.Range3d()
    for corner in scaled:
        result.UnionWith(corner)
    return result


def _translated_range(range_3d: Gf.Range3d, translation: tuple[float, float, float]) -> Gf.Range3d:
    offset = Gf.Vec3d(*translation)
    return Gf.Range3d(range_3d.GetMin() + offset, range_3d.GetMax() + offset)


def _compute_default_prim_bounds(stage: Usd.Stage) -> Gf.Range3d | None:
    default_prim = stage.GetDefaultPrim()
    if not default_prim or not default_prim.IsValid():
        return None
    bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render", "proxy"])
    bound = bbox_cache.ComputeWorldBound(default_prim)
    box = bound.ComputeAlignedBox()
    if box.IsEmpty():
        return None
    return box


def inspect_asset(asset: SceneAsset) -> None:
    print(f"\n[{asset.name}]")
    print(f"  file: {asset.path}")
    print(f"  planned root pos: {_format_vec(asset.root_pos_m)}")
    print(f"  scale: {_format_vec(asset.scale)}")

    if not asset.path.exists():
        print("  status: missing")
        return

    stage = Usd.Stage.Open(str(asset.path))
    if stage is None:
        print("  status: failed to open")
        return

    default_prim = stage.GetDefaultPrim()
    default_path = default_prim.GetPath().pathString if default_prim and default_prim.IsValid() else "<none>"
    prim_count = sum(1 for _ in stage.Traverse())
    print(f"  default prim: {default_path}")
    print(f"  prim count: {prim_count}")

    local_bounds = _compute_default_prim_bounds(stage)
    if local_bounds is None:
        print("  local bounds: unavailable")
        return

    scaled_bounds = _scaled_range(local_bounds, asset.scale)
    scene_bounds = _translated_range(scaled_bounds, asset.root_pos_m)
    print(f"  local min: {_format_vec(local_bounds.GetMin())}")
    print(f"  local max: {_format_vec(local_bounds.GetMax())}")
    print(f"  scaled min: {_format_vec(scaled_bounds.GetMin())}")
    print(f"  scaled max: {_format_vec(scaled_bounds.GetMax())}")
    print(f"  scene min: {_format_vec(scene_bounds.GetMin())}")
    print(f"  scene max: {_format_vec(scene_bounds.GetMax())}")
    print(f"  scene bottom vs table top: {scene_bounds.GetMin()[2] - TABLE_TOP_Z_M: .4f} m")


def main() -> None:
    print("Contact-valid placement asset inspection")
    print(f"project: {PROJECT_ROOT}")
    print(f"table center: {_format_vec(TABLE_CENTER_M)}")
    print(f"table size: {_format_vec(TABLE_SIZE_M)}")
    print(f"table top z: {TABLE_TOP_Z_M:.4f}")

    for asset in SCENE_ASSETS:
        inspect_asset(asset)


if __name__ == "__main__":
    main()
