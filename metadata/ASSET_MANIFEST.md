# Asset Manifest

This bundle contains only the assets planned for the clean contact-valid placement demo.

## Active Assets

All active assets are stored directly under `assets/`.

Files:

- `assets/xarm6_with_gripper_contact_refined.usd`
- `assets/4_50ml_conical_holder_contact_refined.usd`
- `assets/vortexer_contact_refined.usd`
- `assets/autobio_50ml_tube_contact_refined.usda`

## Sources

- `xarm6_with_gripper_contact_refined.usd`, `4_50ml_conical_holder_contact_refined.usd`, and `vortexer_contact_refined.usd` come from `Isaac_Sim_Lab_Automation/contact_assets`.
- `autobio_50ml_tube_contact_refined.usda` is generated from AutoBio 50ml tube dimensions and uses explicit USD primitives for visual and collision geometry.

## Tube Notes

- The old refined tube USD from `Isaac_Sim_Lab_Automation/contact_assets/50ml_conical_ep_tube_contact_refined.usd` is intentionally excluded.
- The active tube uses a transparent body cylinder, an orange cap cylinder, and collision cylinders for body, upper sleeve, and cap/collar.
- Tube rigid body mass is set to `0.008 kg`.

## License Notes

The current local AutoBio clone does not appear to include a root-level license file. Before publishing or sharing derived assets, confirm the license and asset usage terms from the upstream repository.
