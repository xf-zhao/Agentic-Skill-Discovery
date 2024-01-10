# Copyright (c) 2022-2023, The ORBIT Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configurations for the object lift environments."""

# from . import config


import gymnasium as gym

from envs.franka_table.config.franka import agents

##
# Register Gym environments.
##

##
# Joint Position Control
##

from envs.franka_table.env_cfg import franka_table_env_cfg

gym.register(
    id="Franka_Table",
    entry_point="omni.isaac.orbit.envs:RLTaskEnv",
    kwargs={
        "env_cfg_entry_point": franka_table_env_cfg.FrankaTableEnvCfg,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.PlayCubePPORunnerCfg,
    },
    disable_env_checker=True,
)