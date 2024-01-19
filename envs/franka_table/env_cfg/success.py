from __future__ import annotations
from omni.isaac.orbit.utils import configclass
from omni.isaac.orbit.managers import RewardTermCfg as RewTerm
from omni.isaac.orbit.envs import RLTaskEnv

import torch


def object_is_lifted(
    env: RLTaskEnv, obs_name: str, minimal_height: float
) -> torch.Tensor:
    """Sparse reward the agent for lifting the object above the minimal height."""
    obs = env.obs_buf["observations"]
    object_height = obs[obs_name][:, 2]  # x, y, z (height)
    is_success = torch.where(object_height > minimal_height, 1.0, 0.0)
    return is_success


@configclass
class SuccessCfg:
    # Success condition for cube_a to be lifted up to sufficient height, here, a 0.3 meter as an example.
    success = RewTerm(
        func=object_is_lifted,
        params={
            "obs_name": "cube_a_position",
            "minimal_height": 0.3,
        },  # Table plane as z = 0
        weight=30.0,
    )
