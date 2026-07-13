"""Tube placement direct RL task.

This file will eventually register the Gym task id.
It is intentionally not registering yet because the environment class is still
being scaffolded.
"""

import gymnasium as gym


from . import agents


gym.register(
    id="ContactValid-TubePlace-Direct-v0",
    entry_point="contact_valid_place_rl.tasks.direct.tube_place.tube_place_env:TubePlaceEnv",
    kwargs={
        "env_cfg_entry_point": "contact_valid_place_rl.tasks.direct.tube_place.tube_place_env_cfg:TubePlaceEnvCfg",
        "rsl_rl_cfg_entry_point": "contact_valid_place_rl.tasks.direct.tube_place.agents.rsl_rl_ppo_cfg:TubePlacePPORunnerCfg",
    },
)
