#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from wetlab_benchmark.task_config import (
    ASSETS,
    BENCHMARK_ROOT,
    GRIPPER_PAD_CONTACT_LOCAL_Y_M,
    GRIPPER_PAD_CONTACT_LOCAL_Z_M,
    IMPORTED_LAB_ASSETS,
)


CONTACT_ASSET_DIR = BENCHMARK_ROOT / "contact_assets_claude_v4"
REFINED_TUBE_USD = CONTACT_ASSET_DIR / "50ml_conical_ep_tube_contact_refined.usd"
REFINED_XARM_USD = CONTACT_ASSET_DIR / "xarm6_with_gripper_contact_refined.usd"
REFINED_HOLDER_USD = CONTACT_ASSET_DIR / "4_50ml_conical_holder_contact_refined.usd"
REFINED_VORTEXER_USD = CONTACT_ASSET_DIR / "vortexer_contact_refined.usd"
REFINED_PROXY_PAD_USD = CONTACT_ASSET_DIR / "proxy_gripper_pad_contact_refined.usd"
ASSET_UNITS_PER_M = 1000.0


def _require_pxr():
    try:
        from pxr import Gf, Usd, UsdGeom, UsdPhysics, UsdShade
    except ImportError as exc:  # pragma: no cover - requires Isaac Python
        raise RuntimeError("Run this script with Isaac Sim Python, e.g. /home/ubuntu/IsaacLab/_isaac_sim/python.sh") from exc
    return Gf, Usd, UsdGeom, UsdPhysics, UsdShade


def _set_physx_collision_offsets(stage, prim_path: str, *, rest_offset_m: float | None = None, contact_offset_m: float | None = None) -> None:
    """Override PhysX rest_offset / contact_offset on a specific collider prim.

    Negative rest_offset = colliders rest in stable overlap (used here to make the
    oversized grip_collar visually match the visible cap surface without moving
    the collider geometry itself, so closure dynamics and approach computation
    remain identical to the working baseline).

    Sets the underlying USD attributes (`physxCollision:restOffset`,
    `physxCollision:contactOffset`) directly. Equivalent to applying
    PhysxSchema.PhysxCollisionAPI but doesn't require that import — the user-site
    pxr install in this environment shadows the Isaac Sim one and doesn't ship
    PhysxSchema.
    """
    from pxr import Sdf
    prim = stage.GetPrimAtPath(prim_path)
    if not prim or not prim.IsValid():
        raise RuntimeError(f"Cannot set PhysX collision offsets on missing prim: {prim_path}")
    # Apply the schema by adding it to the prim's apiSchemas metadata so PhysX
    # actually reads these attributes at simulation time.
    schemas = prim.GetMetadata("apiSchemas") or Sdf.TokenListOp()
    existing = list(schemas.GetAddedOrExplicitItems())
    if "PhysxCollisionAPI" not in existing:
        existing.append("PhysxCollisionAPI")
        new_schemas = Sdf.TokenListOp()
        new_schemas.prependedItems = existing
        prim.SetMetadata("apiSchemas", new_schemas)
    if rest_offset_m is not None:
        attr = prim.CreateAttribute("physxCollision:restOffset", Sdf.ValueTypeNames.Float)
        attr.Set(float(rest_offset_m))
    if contact_offset_m is not None:
        attr = prim.CreateAttribute("physxCollision:contactOffset", Sdf.ValueTypeNames.Float)
        attr.Set(float(contact_offset_m))


def _physics_material(stage, UsdPhysics, UsdShade, path: str, *, static: float, dynamic: float, restitution: float):
    material = UsdShade.Material.Define(stage, path)
    api = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
    api.CreateStaticFrictionAttr(static)
    api.CreateDynamicFrictionAttr(dynamic)
    api.CreateRestitutionAttr(restitution)
    return material


def _bind_material(UsdShade, prim, material) -> None:
    UsdShade.MaterialBindingAPI.Apply(prim).Bind(material)


def _set_mass(UsdPhysics, prim, mass_kg: float) -> None:
    if prim and prim.IsValid():
        UsdPhysics.MassAPI.Apply(prim).CreateMassAttr(float(mass_kg))


def _disable_collision_subtree(stage, Usd, UsdPhysics, root_path: str) -> None:
    root = stage.GetPrimAtPath(root_path)
    if not root or not root.IsValid():
        return
    for prim in Usd.PrimRange(root):
        if prim.IsInstance():
            prim.SetInstanceable(False)
    for prim in Usd.PrimRange(root):
        if prim.HasAPI(UsdPhysics.CollisionAPI):
            UsdPhysics.CollisionAPI(prim).CreateCollisionEnabledAttr(False)


def _disable_all_collisions(stage, Usd, UsdPhysics) -> None:
    for prim in stage.Traverse():
        if prim.IsInstance():
            prim.SetInstanceable(False)
    for prim in stage.Traverse():
        if prim.HasAPI(UsdPhysics.CollisionAPI):
            UsdPhysics.CollisionAPI(prim).CreateCollisionEnabledAttr(False)


def _scaled_xyz(xyz_m: tuple[float, float, float], scale: tuple[float, float, float]) -> tuple[float, float, float]:
    return (xyz_m[0] / scale[0], xyz_m[1] / scale[1], xyz_m[2] / scale[2])


def _add_collision_cube(
    stage,
    Gf,
    UsdGeom,
    UsdPhysics,
    UsdShade,
    path: str,
    *,
    center,
    size,
    material,
    rotate_xyz_deg: tuple[float, float, float] | None = None,
) -> None:
    cube = UsdGeom.Cube.Define(stage, path)
    cube.CreateSizeAttr(1.0)
    # visibility="invisible" alone isn't enough for Kit's RTX renderer — the prim
    # still gets shaded with default material and shows up as a black blob in
    # truecolor renders. Mark purpose="guide" so all renderers exclude it from
    # the regular render pass. Collision is unaffected (PhysX reads CollisionAPI,
    # not purpose).
    cube.CreateVisibilityAttr(UsdGeom.Tokens.invisible)
    cube.CreatePurposeAttr(UsdGeom.Tokens.guide)
    xform = UsdGeom.Xformable(cube.GetPrim())
    xform.ClearXformOpOrder()
    xform.AddTranslateOp().Set(Gf.Vec3d(*center))
    if rotate_xyz_deg is not None:
        xform.AddRotateXYZOp().Set(Gf.Vec3f(*rotate_xyz_deg))
    xform.AddScaleOp().Set(Gf.Vec3f(*size))
    UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
    _bind_material(UsdShade, cube.GetPrim(), material)


def _add_visual_cube(stage, Gf, UsdGeom, path: str, *, center, size, color) -> None:
    cube = UsdGeom.Cube.Define(stage, path)
    cube.CreateSizeAttr(1.0)
    cube.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    xform = UsdGeom.Xformable(cube.GetPrim())
    xform.ClearXformOpOrder()
    xform.AddTranslateOp().Set(Gf.Vec3d(*center))
    xform.AddScaleOp().Set(Gf.Vec3f(*size))


def _add_collision_cylinder(stage, Gf, UsdGeom, UsdPhysics, UsdShade, path: str, *, center, radius: float, height: float, material) -> None:
    cylinder = UsdGeom.Cylinder.Define(stage, path)
    cylinder.CreateRadiusAttr(radius)
    cylinder.CreateHeightAttr(height)
    cylinder.CreateAxisAttr("Z")
    # See _add_collision_cube for why purpose="guide" is needed in addition to
    # visibility="invisible".
    cylinder.CreateVisibilityAttr(UsdGeom.Tokens.invisible)
    cylinder.CreatePurposeAttr(UsdGeom.Tokens.guide)
    xform = UsdGeom.Xformable(cylinder.GetPrim())
    xform.ClearXformOpOrder()
    xform.AddTranslateOp().Set(Gf.Vec3d(*center))
    UsdPhysics.CollisionAPI.Apply(cylinder.GetPrim())
    _bind_material(UsdShade, cylinder.GetPrim(), material)


def _add_open_well_boxes(
    stage,
    Gf,
    UsdGeom,
    UsdPhysics,
    UsdShade,
    *,
    prefix: str,
    center_xy_m: tuple[float, float],
    support_z_m: float,
    scale: tuple[float, float, float],
    inner_halfspan_xy_m: tuple[float, float],
    wall_thickness_m: float,
    wall_height_m: float,
    floor_thickness_m: float,
    material,
    visual_color: tuple[float, float, float] | None = None,
) -> None:
    cx, cy = center_xy_m
    hx, hy = inner_halfspan_xy_m
    t = wall_thickness_m
    walls = [
        (
            f"{prefix}_wall_left",
            (t, 2.0 * hy + 2.0 * t, wall_height_m),
            (cx - hx - 0.5 * t, cy, support_z_m + 0.5 * wall_height_m),
        ),
        (
            f"{prefix}_wall_right",
            (t, 2.0 * hy + 2.0 * t, wall_height_m),
            (cx + hx + 0.5 * t, cy, support_z_m + 0.5 * wall_height_m),
        ),
        (
            f"{prefix}_wall_front",
            (2.0 * hx, t, wall_height_m),
            (cx, cy - hy - 0.5 * t, support_z_m + 0.5 * wall_height_m),
        ),
        (
            f"{prefix}_wall_back",
            (2.0 * hx, t, wall_height_m),
            (cx, cy + hy + 0.5 * t, support_z_m + 0.5 * wall_height_m),
        ),
    ]
    if floor_thickness_m > 0.0:
        walls.append(
            (
                f"{prefix}_floor",
                (2.0 * hx, 2.0 * hy, floor_thickness_m),
                (cx, cy, support_z_m - 0.5 * floor_thickness_m),
            )
        )
    for name, size_m, center_m in walls:
        _add_collision_cube(
            stage,
            Gf,
            UsdGeom,
            UsdPhysics,
            UsdShade,
            f"/World/contact_colliders/{name}",
            center=_scaled_xyz(center_m, scale),
            size=_scaled_xyz(size_m, scale),
            material=material,
        )
        if visual_color is not None:
            _add_visual_cube(
                stage,
                Gf,
                UsdGeom,
                f"/World/visuals/{name}",
                center=_scaled_xyz(center_m, scale),
                size=_scaled_xyz(size_m, scale),
                color=visual_color,
            )


def _add_annular_floor_boxes(
    stage,
    Gf,
    UsdGeom,
    UsdPhysics,
    UsdShade,
    *,
    prefix: str,
    center_xy_m: tuple[float, float],
    floor_z_m: float,
    scale: tuple[float, float, float],
    outer_halfspan_xy_m: tuple[float, float],
    inner_void_halfspan_xy_m: tuple[float, float],
    thickness_m: float,
    material,
    visual_color: tuple[float, float, float] | None = None,
) -> None:
    cx, cy = center_xy_m
    hx, hy = outer_halfspan_xy_m
    vx, vy = inner_void_halfspan_xy_m
    if not (0.0 < vx < hx and 0.0 < vy < hy):
        raise ValueError("Annular floor inner void must be smaller than outer halfspan")
    floor_segments = [
        (
            f"{prefix}_floor_left",
            (hx - vx, 2.0 * hy, thickness_m),
            (cx - 0.5 * (hx + vx), cy, floor_z_m - 0.5 * thickness_m),
        ),
        (
            f"{prefix}_floor_right",
            (hx - vx, 2.0 * hy, thickness_m),
            (cx + 0.5 * (hx + vx), cy, floor_z_m - 0.5 * thickness_m),
        ),
        (
            f"{prefix}_floor_front",
            (2.0 * vx, hy - vy, thickness_m),
            (cx, cy - 0.5 * (hy + vy), floor_z_m - 0.5 * thickness_m),
        ),
        (
            f"{prefix}_floor_back",
            (2.0 * vx, hy - vy, thickness_m),
            (cx, cy + 0.5 * (hy + vy), floor_z_m - 0.5 * thickness_m),
        ),
    ]
    for name, size_m, center_m in floor_segments:
        _add_collision_cube(
            stage,
            Gf,
            UsdGeom,
            UsdPhysics,
            UsdShade,
            f"/World/contact_colliders/{name}",
            center=_scaled_xyz(center_m, scale),
            size=_scaled_xyz(size_m, scale),
            material=material,
        )
        if visual_color is not None:
            _add_visual_cube(
                stage,
                Gf,
                UsdGeom,
                f"/World/visuals/{name}",
                center=_scaled_xyz(center_m, scale),
                size=_scaled_xyz(size_m, scale),
                color=visual_color,
            )


def build_refined_tube(*, force: bool) -> Path:
    Gf, Usd, UsdGeom, UsdPhysics, UsdShade = _require_pxr()
    CONTACT_ASSET_DIR.mkdir(parents=True, exist_ok=True)
    if REFINED_TUBE_USD.exists() and not force:
        return REFINED_TUBE_USD
    shutil.copy2(ASSETS.tube_usd, REFINED_TUBE_USD)
    stage = Usd.Stage.Open(str(REFINED_TUBE_USD))
    if stage is None:
        raise RuntimeError(f"Unable to open {REFINED_TUBE_USD}")

    material = _physics_material(
        stage,
        UsdPhysics,
        UsdShade,
        "/World/Looks/TubeGlassPhysics",
        static=0.85,
        dynamic=0.65,
        restitution=0.02,
    )
    _set_mass(UsdPhysics, stage.GetPrimAtPath("/World"), 0.008)
    _disable_collision_subtree(stage, Usd, UsdPhysics, "/World")
    # Tube mesh is authored in millimeter-like local units and scaled by 0.001 at spawn.
    # Keep a full-height simple body collider for release/settle, then make the
    # upper grasp region more stepped: a slimmer sleeve and a taller, wider top
    # collar. That gives the finger shelf a real undercut to catch instead of a
    # nearly cylindrical surface that only produces squeeze-and-slip.
    center_x = IMPORTED_LAB_ASSETS.tube_center_local_xy_m[0] * ASSET_UNITS_PER_M
    center_y = IMPORTED_LAB_ASSETS.tube_center_local_xy_m[1] * ASSET_UNITS_PER_M
    body_bottom = IMPORTED_LAB_ASSETS.tube_bottom_from_root_m * ASSET_UNITS_PER_M
    body_top = IMPORTED_LAB_ASSETS.tube_top_from_root_m * ASSET_UNITS_PER_M
    body_center_z = 0.5 * (body_bottom + body_top)
    body_height = body_top - body_bottom
    grip_bottom = max(body_top - 42.0, body_bottom)
    grip_center_z = 0.5 * (grip_bottom + body_top)
    grip_height = body_top - grip_bottom
    collar_height = 9.0
    collar_center_z = body_top - 0.5 * collar_height
    _add_collision_cylinder(
        stage,
        Gf,
        UsdGeom,
        UsdPhysics,
        UsdShade,
        "/World/contact_colliders/body",
        center=(center_x, center_y, body_center_z),
        radius=16.0,
        height=body_height,
        material=material,
    )
    _add_collision_cylinder(
        stage,
        Gf,
        UsdGeom,
        UsdPhysics,
        UsdShade,
        "/World/contact_colliders/grip_sleeve",
        center=(center_x, center_y, grip_center_z),
        radius=18.5,
        height=grip_height,
        material=material,
    )
    _add_collision_cylinder(
        stage,
        Gf,
        UsdGeom,
        UsdPhysics,
        UsdShade,
        "/World/contact_colliders/grip_collar",
        center=(center_x, center_y, collar_center_z),
        radius=25.0,
        height=collar_height,
        material=material,
    )
    # v4 single change: pull the grip_collar contact equilibrium 7mm INTO the
    # collider so the pads visually rest at the visible cap surface (radius
    # 17.75mm) instead of at the 25mm collar boundary. Drive force is unchanged
    # (force = stiffness * (rest_offset - separation), shifting rest_offset by
    # delta shifts equilibrium separation by the same delta). Bump contact_offset
    # to comfortably exceed |rest_offset| so the broadphase keeps the contact
    # alive at deep equilibrium overlap.
    _set_physx_collision_offsets(
        stage,
        "/World/contact_colliders/grip_collar",
        rest_offset_m=-0.003,
        contact_offset_m=0.005,
    )
    stage.GetRootLayer().Save()
    return REFINED_TUBE_USD


def build_refined_robot(*, force: bool) -> Path:
    Gf, Usd, UsdGeom, UsdPhysics, UsdShade = _require_pxr()
    CONTACT_ASSET_DIR.mkdir(parents=True, exist_ok=True)
    if REFINED_XARM_USD.exists() and not force:
        return REFINED_XARM_USD
    shutil.copy2(ASSETS.xarm6_usd, REFINED_XARM_USD)
    stage = Usd.Stage.Open(str(REFINED_XARM_USD))
    if stage is None:
        raise RuntimeError(f"Unable to open {REFINED_XARM_USD}")

    material = _physics_material(
        stage,
        UsdPhysics,
        UsdShade,
        "/UF_ROBOT/Looks/FingerRubberPhysics",
        static=2.0,
        dynamic=1.5,
        restitution=0.0,
    )
    for link_name in (
        "link5",
        "link6",
        "xarm_gripper_base_link",
        "left_finger",
        "right_finger",
        "left_inner_knuckle",
        "right_inner_knuckle",
        "left_outer_knuckle",
        "right_outer_knuckle",
    ):
        _disable_collision_subtree(stage, Usd, UsdPhysics, f"/UF_ROBOT/{link_name}")
    # Keep the refined finger contact close to the real AG1002 / G1 fingertip
    # pad, but use a finite pad thickness and a tip-biased contact band rather
    # than the fully wall-aligned branch that never achieved pinch contact.
    # This preserves a more physical fingertip pad interpretation without going
    # back to the older deep invisible pocket.
    for side_name, side_sign in (("left_finger", -1.0), ("right_finger", 1.0)):
        base = f"/UF_ROBOT/{side_name}"
        main_y = side_sign * GRIPPER_PAD_CONTACT_LOCAL_Y_M
        bevel_y = side_sign * (GRIPPER_PAD_CONTACT_LOCAL_Y_M - 0.0010)
        shelf_y = side_sign * (GRIPPER_PAD_CONTACT_LOCAL_Y_M - 0.0005)
        shelf_z = min(GRIPPER_PAD_CONTACT_LOCAL_Z_M + 0.004, 0.054)
        _add_collision_cube(
            stage,
            Gf,
            UsdGeom,
            UsdPhysics,
            UsdShade,
            f"{base}/contact_pad",
            center=(0.0, main_y, GRIPPER_PAD_CONTACT_LOCAL_Z_M),
            size=(0.028, 0.005, 0.016),
            material=material,
        )
        _add_collision_cube(
            stage,
            Gf,
            UsdGeom,
            UsdPhysics,
            UsdShade,
            f"{base}/contact_shelf",
            center=(0.0, shelf_y, shelf_z),
            size=(0.024, 0.006, 0.008),
            material=material,
        )
        _add_collision_cube(
            stage,
            Gf,
            UsdGeom,
            UsdPhysics,
            UsdShade,
            f"{base}/contact_v_groove_neg",
            center=(0.0, bevel_y, GRIPPER_PAD_CONTACT_LOCAL_Z_M),
            size=(0.022, 0.0015, 0.014),
            rotate_xyz_deg=(0.0, 0.0, -4.0),
            material=material,
        )
        _add_collision_cube(
            stage,
            Gf,
            UsdGeom,
            UsdPhysics,
            UsdShade,
            f"{base}/contact_v_groove_pos",
            center=(0.0, bevel_y, GRIPPER_PAD_CONTACT_LOCAL_Z_M),
            size=(0.022, 0.0015, 0.014),
            rotate_xyz_deg=(0.0, 0.0, 4.0),
            material=material,
        )
        _add_collision_cube(
            stage,
            Gf,
            UsdGeom,
            UsdPhysics,
            UsdShade,
            f"{base}/contact_x_stop_neg",
            center=(-0.012, main_y, GRIPPER_PAD_CONTACT_LOCAL_Z_M),
            size=(0.003, 0.005, 0.016),
            material=material,
        )
        _add_collision_cube(
            stage,
            Gf,
            UsdGeom,
            UsdPhysics,
            UsdShade,
            f"{base}/contact_x_stop_pos",
            center=(0.012, main_y, GRIPPER_PAD_CONTACT_LOCAL_Z_M),
            size=(0.003, 0.005, 0.016),
            material=material,
        )
    stage.GetRootLayer().Save()
    return REFINED_XARM_USD


def build_refined_holder(*, force: bool) -> Path:
    Gf, Usd, UsdGeom, UsdPhysics, UsdShade = _require_pxr()
    CONTACT_ASSET_DIR.mkdir(parents=True, exist_ok=True)
    if REFINED_HOLDER_USD.exists() and not force:
        return REFINED_HOLDER_USD
    shutil.copy2(ASSETS.tube_holder_combo_usd, REFINED_HOLDER_USD)
    stage = Usd.Stage.Open(str(REFINED_HOLDER_USD))
    if stage is None:
        raise RuntimeError(f"Unable to open {REFINED_HOLDER_USD}")

    material = _physics_material(
        stage,
        UsdPhysics,
        UsdShade,
        "/World/Looks/HolderPlasticPhysics",
        static=0.20,
        dynamic=0.10,
        restitution=0.0,
    )
    _disable_all_collisions(stage, Usd, UsdPhysics)
    scale = IMPORTED_LAB_ASSETS.holder_scale
    for slot_index, slot_center in enumerate(IMPORTED_LAB_ASSETS.holder_slot_centers_local_m):
        _add_open_well_boxes(
            stage,
            Gf,
            UsdGeom,
            UsdPhysics,
            UsdShade,
            prefix=f"holder_slot_{slot_index}",
            center_xy_m=(slot_center[0], slot_center[1]),
            support_z_m=slot_center[2],
            scale=scale,
            inner_halfspan_xy_m=(0.030, 0.030),
            wall_thickness_m=0.006,
            wall_height_m=max(IMPORTED_LAB_ASSETS.holder_top_from_root_m - slot_center[2], 0.006),
            floor_thickness_m=0.004,
            material=material,
        )
    stage.GetRootLayer().Save()
    return REFINED_HOLDER_USD


def build_refined_vortexer(*, force: bool) -> Path:
    Gf, Usd, UsdGeom, UsdPhysics, UsdShade = _require_pxr()
    CONTACT_ASSET_DIR.mkdir(parents=True, exist_ok=True)
    if REFINED_VORTEXER_USD.exists() and not force:
        return REFINED_VORTEXER_USD
    shutil.copy2(ASSETS.vortexer_usd, REFINED_VORTEXER_USD)
    stage = Usd.Stage.Open(str(REFINED_VORTEXER_USD))
    if stage is None:
        raise RuntimeError(f"Unable to open {REFINED_VORTEXER_USD}")

    material = _physics_material(
        stage,
        UsdPhysics,
        UsdShade,
        "/World/Looks/VortexerPlasticPhysics",
        # The real vortexer top behaves more like a grippy elastomer seat than
        # smooth plastic. Higher support friction helps the tube settle instead
        # of skating and tipping after release.
        static=1.20,
        dynamic=0.90,
        restitution=0.0,
    )
    _disable_all_collisions(stage, Usd, UsdPhysics)
    scale = IMPORTED_LAB_ASSETS.scale
    support = IMPORTED_LAB_ASSETS.vortexer_support_center_local_m
    annular_floor_visual_color = (0.93, 0.93, 0.93)
    # The imported vortexer mesh exposes a small raised center boss at z = 145
    # mm, but the visible cup floor that supports the tube shell is the broader
    # annulus at z = 130 mm. Model the support as an annular ring so the tube
    # can seat down inside the hollow instead of balancing on a flat synthetic
    # floor or on the center boss. Add matching visible annulus pieces so the
    # render shows the same support the physics solver is actually using.
    cup_halfspan_xy_m = (0.0195, 0.0195)
    _add_open_well_boxes(
        stage,
        Gf,
        UsdGeom,
        UsdPhysics,
        UsdShade,
        prefix="vortexer_well",
        center_xy_m=(support[0], support[1]),
        support_z_m=support[2],
        scale=scale,
        inner_halfspan_xy_m=cup_halfspan_xy_m,
        wall_thickness_m=0.006,
        wall_height_m=max(IMPORTED_LAB_ASSETS.vortexer_top_from_root_m - support[2], 0.006),
        floor_thickness_m=0.0,
        material=material,
    )
    _add_annular_floor_boxes(
        stage,
        Gf,
        UsdGeom,
        UsdPhysics,
        UsdShade,
        prefix="vortexer_well",
        center_xy_m=(support[0], support[1]),
        floor_z_m=support[2],
        scale=scale,
        outer_halfspan_xy_m=cup_halfspan_xy_m,
        inner_void_halfspan_xy_m=(0.010, 0.010),
        thickness_m=0.006,
        material=material,
        visual_color=annular_floor_visual_color,
    )
    stage.GetRootLayer().Save()
    return REFINED_VORTEXER_USD


def build_refined_proxy_pad(*, force: bool) -> Path:
    # Legacy debug asset for older proxy-pad experiments. The baked-contact live
    # execution path no longer spawns these runtime helper bodies.
    Gf, Usd, UsdGeom, UsdPhysics, UsdShade = _require_pxr()
    CONTACT_ASSET_DIR.mkdir(parents=True, exist_ok=True)
    if REFINED_PROXY_PAD_USD.exists() and not force:
        return REFINED_PROXY_PAD_USD

    stage = Usd.Stage.CreateNew(str(REFINED_PROXY_PAD_USD))
    if stage is None:
        raise RuntimeError(f"Unable to create {REFINED_PROXY_PAD_USD}")
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    root = UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(root.GetPrim())

    material = _physics_material(
        stage,
        UsdPhysics,
        UsdShade,
        "/World/Looks/ProxyFingerRubberPhysics",
        static=2.2,
        dynamic=1.7,
        restitution=0.0,
    )
    UsdGeom.Xform.Define(stage, "/World/contact_colliders")
    UsdGeom.Xform.Define(stage, "/World/visuals")

    # The same pad asset is used on both sides of the proxy gripper, so the
    # collision geometry is symmetric around local Y. The broad vertical face
    # gives stable side contact, while the shallow shelf is positioned to catch
    # the refined tube collar during lift instead of depending on friction only.
    _add_collision_cube(
        stage,
        Gf,
        UsdGeom,
        UsdPhysics,
        UsdShade,
        "/World/contact_colliders/grip_face",
        center=(0.0, 0.0, -0.014),
        size=(0.064, 0.006, 0.046),
        material=material,
    )
    _add_collision_cube(
        stage,
        Gf,
        UsdGeom,
        UsdPhysics,
        UsdShade,
        "/World/contact_colliders/under_lip_shelf",
        center=(0.0, 0.0, 0.0),
        size=(0.064, 0.018, 0.004),
        material=material,
    )
    _add_collision_cube(
        stage,
        Gf,
        UsdGeom,
        UsdPhysics,
        UsdShade,
        "/World/contact_colliders/v_groove_neg_y",
        center=(0.0, -0.003, -0.010),
        size=(0.064, 0.0025, 0.036),
        material=material,
        rotate_xyz_deg=(12.0, 0.0, 0.0),
    )
    _add_collision_cube(
        stage,
        Gf,
        UsdGeom,
        UsdPhysics,
        UsdShade,
        "/World/contact_colliders/v_groove_pos_y",
        center=(0.0, 0.003, -0.010),
        size=(0.064, 0.0025, 0.036),
        material=material,
        rotate_xyz_deg=(-12.0, 0.0, 0.0),
    )
    # End stops bound the tube in the pad-length direction. Without these
    # lips the tube can remain squeezed but slide along local X during long
    # carries, which produces physically valid but off-center releases.
    _add_collision_cube(
        stage,
        Gf,
        UsdGeom,
        UsdPhysics,
        UsdShade,
        "/World/contact_colliders/x_stop_neg",
        center=(-0.028, 0.0, -0.012),
        size=(0.004, 0.018, 0.040),
        material=material,
    )
    _add_collision_cube(
        stage,
        Gf,
        UsdGeom,
        UsdPhysics,
        UsdShade,
        "/World/contact_colliders/x_stop_pos",
        center=(0.028, 0.0, -0.012),
        size=(0.004, 0.018, 0.040),
        material=material,
    )
    _add_visual_cube(
        stage,
        Gf,
        UsdGeom,
        "/World/visuals/rubber_pad",
        center=(0.0, 0.0, -0.014),
        size=(0.064, 0.018, 0.046),
        color=(0.04, 0.04, 0.035),
    )
    UsdPhysics.RigidBodyAPI.Apply(root.GetPrim())
    _set_mass(UsdPhysics, root.GetPrim(), 0.02)
    stage.GetRootLayer().Save()
    return REFINED_PROXY_PAD_USD


def _validate_asset(path: Path) -> None:
    _, Usd, _, UsdPhysics, _ = _require_pxr()
    stage = Usd.Stage.Open(str(path))
    if stage is None:
        raise RuntimeError(f"Unable to reopen {path}")
    collision_paths = []
    instance_collision_paths = []
    for prim in stage.Traverse():
        if prim.HasAPI(UsdPhysics.CollisionAPI):
            collision_paths.append(prim.GetPath().pathString)
            if prim.IsInstance() or prim.IsInstanceProxy():
                instance_collision_paths.append(prim.GetPath().pathString)
    print(f"{path}: collision_prims={len(collision_paths)} instance_collision_prims={len(instance_collision_paths)}")
    for collision_path in collision_paths:
        if (
            "contact_pad" in collision_path
            or "grip_sleeve" in collision_path
            or "proxy" in collision_path.lower()
            or "contact_colliders" in collision_path
        ):
            print(f"  refined_collision={collision_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build contact-refined USD assets for physical live execution.")
    parser.add_argument("--force", action="store_true", help="Overwrite existing refined assets.")
    args = parser.parse_args()
    tube = build_refined_tube(force=args.force)
    robot = build_refined_robot(force=args.force)
    holder = build_refined_holder(force=args.force)
    vortexer = build_refined_vortexer(force=args.force)
    _validate_asset(tube)
    _validate_asset(robot)
    _validate_asset(holder)
    _validate_asset(vortexer)


if __name__ == "__main__":
    main()
