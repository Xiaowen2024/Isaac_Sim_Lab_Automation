from __future__ import annotations

import argparse
from pathlib import Path

from isaacsim import SimulationApp


def _build_import_config(_urdf, args):
    import_config = _urdf.ImportConfig()
    import_config.convex_decomp = False
    import_config.fix_base = args.fix_base
    import_config.merge_fixed_joints = args.merge_joints
    import_config.make_default_prim = True
    import_config.self_collision = False
    import_config.create_physics_scene = False
    import_config.import_inertia_tensor = True
    import_config.distance_scale = 1.0
    import_config.density = 0.0
    return import_config


def main():
    parser = argparse.ArgumentParser(description="Convert xArm6 URDF to a saved USD file with Isaac Sim.")
    parser.add_argument("--urdf", required=True, help="Absolute path to xarm6.urdf")
    parser.add_argument("--out", required=True, help="Absolute output USD path")
    parser.set_defaults(fix_base=True)
    parser.add_argument("--fix-base", dest="fix_base", action="store_true", help="Import the robot as fixed-base.")
    parser.add_argument(
        "--floating-base",
        dest="fix_base",
        action="store_false",
        help="Import the robot as floating-base.",
    )
    parser.add_argument("--merge-joints", action="store_true", default=False)
    parser.add_argument("--headless", action="store_true", default=True)
    args = parser.parse_args()

    urdf_path = Path(args.urdf).expanduser().resolve()
    out_path = Path(args.out).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not urdf_path.exists():
        raise FileNotFoundError(f"URDF not found: {urdf_path}")

    simulation_app = SimulationApp({"headless": args.headless})
    try:
        import omni.kit.app
        import omni.kit.commands
        import omni.usd
        from isaacsim.asset.importer.urdf import _urdf

        omni.usd.get_context().new_stage()
        import_config = _build_import_config(_urdf, args)

        print(f"[convert_xarm6_urdf] urdf={urdf_path}")
        print(f"[convert_xarm6_urdf] out={out_path}")
        print("[convert_xarm6_urdf] importing into current stage...")

        result, robot_model = omni.kit.commands.execute(
            "URDFParseFile",
            urdf_path=str(urdf_path),
            import_config=import_config,
        )
        if not result:
            raise RuntimeError("URDFParseFile returned failure")

        result, prim_path = omni.kit.commands.execute(
            "URDFImportRobot",
            urdf_robot=robot_model,
            import_config=import_config,
        )
        if not result:
            raise RuntimeError("URDFImportRobot returned failure")

        # Let Isaac finish authoring the imported prims before saving the stage.
        app = omni.kit.app.get_app()
        for _ in range(8):
            simulation_app.update()
            app.update()

        stage = omni.usd.get_context().get_stage()
        if stage is None:
            raise RuntimeError("No active USD stage after URDF import")

        default_prim = stage.GetDefaultPrim()
        print(f"[convert_xarm6_urdf] imported_prim={prim_path}")
        print(f"[convert_xarm6_urdf] default_prim={default_prim.GetPath() if default_prim else '<none>'}")
        print("[convert_xarm6_urdf] saving stage...")

        if not omni.usd.get_context().save_as_stage(str(out_path), None):
            raise RuntimeError(f"save_as_stage failed for {out_path}")

        for _ in range(4):
            simulation_app.update()
            app.update()

        if not out_path.exists():
            raise RuntimeError(f"Import finished but USD was not created at {out_path}")

        print(f"[convert_xarm6_urdf] saved={out_path}")
    finally:
        simulation_app.close()


if __name__ == "__main__":
    main()
