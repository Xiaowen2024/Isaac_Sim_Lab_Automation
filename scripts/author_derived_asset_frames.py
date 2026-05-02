from __future__ import annotations

import argparse
import struct
from pathlib import Path

import numpy as np
from pxr import Usd, UsdGeom, UsdPhysics


def load_binary_stl(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with open(path, "rb") as f:
        f.read(80)
        num_triangles = struct.unpack("<I", f.read(4))[0]
        data = np.fromfile(
            f,
            dtype=np.dtype(
                [
                    ("normal", "<f4", 3),
                    ("v0", "<f4", 3),
                    ("v1", "<f4", 3),
                    ("v2", "<f4", 3),
                    ("attr", "<u2"),
                ]
            ),
            count=num_triangles,
        )
    triangles = np.stack([data["v0"], data["v1"], data["v2"]], axis=1)
    return triangles, data["normal"]


def holder_slot_centers_raw(holder_stl: Path) -> list[tuple[float, float, float]]:
    _ = holder_stl
    # Derived from the STL slice geometry:
    # four regularly spaced hollows centered along one row, with the slot support
    # floor at local z = 0 rather than the top rim at z = 55.
    return [
        (16.005, -16.205, 0.0),
        (53.005, -16.205, 0.0),
        (90.005, -16.205, 0.0),
        (127.005, -16.205, 0.0),
    ]


def vortexer_support_center_raw(vortexer_stl: Path) -> tuple[float, float, float]:
    _ = vortexer_stl
    # The visible support is the broader annular cup floor, not the tiny raised
    # center boss. Use the annular floor height so the tube can sit inside the
    # hollow instead of riding on the boss.
    return (60.0, 35.0, 130.0)


def tube_frames_raw(tube_stl: Path) -> dict[str, tuple[float, float, float]]:
    triangles, _ = load_binary_stl(tube_stl)
    vertices = triangles.reshape(-1, 3)
    mins = vertices.min(axis=0)
    maxs = vertices.max(axis=0)
    center_xy = ((mins[0] + maxs[0]) * 0.5, (mins[1] + maxs[1]) * 0.5)
    center_z = (mins[2] + maxs[2]) * 0.5
    top_z = maxs[2]
    return {
        "bottom_center": (float(center_xy[0]), float(center_xy[1]), float(mins[2])),
        "center_frame": (float(center_xy[0]), float(center_xy[1]), float(center_z)),
        "top_center": (float(center_xy[0]), float(center_xy[1]), float(top_z)),
        "grasp_frame": (float(center_xy[0]), float(center_xy[1]), float(top_z)),
    }


def _root_prim(stage: Usd.Stage):
    default_prim = stage.GetDefaultPrim()
    if default_prim:
        return default_prim
    for child in stage.GetPseudoRoot().GetChildren():
        return child
    raise RuntimeError("USD has no default/root prim")


def _author_frame(parent_path: str, name: str, translate_xyz: tuple[float, float, float], stage: Usd.Stage) -> None:
    prim = UsdGeom.Xform.Define(stage, f"{parent_path}/{name}")
    xformable = UsdGeom.Xformable(prim)
    xformable.ClearXformOpOrder()
    xformable.AddTranslateOp().Set(translate_xyz)


def _find_mesh_prim(root_prim: Usd.Prim) -> Usd.Prim:
    for prim in Usd.PrimRange(root_prim):
        if prim.GetTypeName() == "Mesh":
            return prim
    raise RuntimeError(f"No mesh prim found under {root_prim.GetPath()}")


def _author_physics(root_prim: Usd.Prim, mesh_prim: Usd.Prim, *, dynamic: bool, collision_approx: str) -> None:
    rigid_api = UsdPhysics.RigidBodyAPI.Apply(root_prim)
    rigid_api.CreateRigidBodyEnabledAttr(True)
    collision_api = UsdPhysics.CollisionAPI.Apply(mesh_prim)
    collision_api.CreateCollisionEnabledAttr(True)
    mesh_collision_api = UsdPhysics.MeshCollisionAPI.Apply(mesh_prim)
    mesh_collision_api.CreateApproximationAttr(collision_approx)
    if not dynamic:
        rigid_api.CreateKinematicEnabledAttr(True)


def author_frames_and_physics(
    usd_path: Path,
    frame_positions: dict[str, tuple[float, float, float]],
    *,
    dynamic: bool,
    collision_approx: str,
) -> None:
    stage = Usd.Stage.Open(str(usd_path))
    root = _root_prim(stage)
    mesh_prim = _find_mesh_prim(root)
    task_frames = UsdGeom.Xform.Define(stage, f"{root.GetPath()}/task_frames")
    for name, position in frame_positions.items():
        _author_frame(str(task_frames.GetPath()), name, position, stage)
    _author_physics(root, mesh_prim, dynamic=dynamic, collision_approx=collision_approx)
    stage.GetRootLayer().Save()


def main() -> None:
    parser = argparse.ArgumentParser(description="Derive semantic frames from STL geometry and author them into USDs.")
    parser.add_argument("--assets-dir", required=True, help="Directory containing STL/USD asset pairs")
    args = parser.parse_args()

    assets_dir = Path(args.assets_dir).expanduser().resolve()

    holder_slots = holder_slot_centers_raw(assets_dir / "4_50ml_Conical_Holder.stl")
    vortexer_support = vortexer_support_center_raw(assets_dir / "Vortexer_rev.B.stl")
    tube_frames = tube_frames_raw(assets_dir / "50ml Conical EP Tube.STL")

    holder_frames = {f"slot_{idx}": center for idx, center in enumerate(holder_slots)}
    holder_frames["support_row_center"] = tuple(np.mean(np.array(holder_slots), axis=0))
    vortexer_frames = {"tube_support": vortexer_support}

    author_frames_and_physics(
        assets_dir / "4_50ml_Conical_Holder.usd",
        holder_frames,
        dynamic=False,
        collision_approx="none",
    )
    author_frames_and_physics(
        assets_dir / "Vortexer_rev.usd",
        vortexer_frames,
        dynamic=False,
        collision_approx="none",
    )
    author_frames_and_physics(
        assets_dir / "50ml Conical EP Tube.usd",
        tube_frames,
        dynamic=True,
        collision_approx="convexHull",
    )

    print("holder_slots", holder_slots)
    print("vortexer_support", vortexer_support)
    print("tube_frames", tube_frames)


if __name__ == "__main__":
    main()
