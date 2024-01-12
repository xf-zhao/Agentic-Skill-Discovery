from __future__ import annotations
from omni.isaac.orbit.utils import configclass
from omni.isaac.orbit.managers import SceneEntityCfg
from omni.isaac.orbit.managers import RewardTermCfg as RewTerm
from omni.isaac.orbit.assets import RigidObject
from omni.isaac.orbit.managers import SceneEntityCfg
from omni.isaac.orbit.sensors import FrameTransformer
from omni.isaac.orbit.utils.math import combine_frame_transforms
from envs.franka_table import mdp
import torch
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from omni.isaac.orbit.envs import RLTaskEnv
    

def object_is_lifted(env: RLTaskEnv, obs_name: str, minimal_height: float) -> torch.Tensor:
    """Sparse reward the agent for lifting the object above the minimal height."""
    obs = env.obs_buf["observations"]
    object_height = obs[obs_name][:, 2] # x, y, z (height)
    is_success = torch.where(object_height > minimal_height, 1.0, 0.0)
    return is_success


@configclass
class SuccessCfg:
    # Success condition for cube_a to be lifted up to sufficient height, here, a 0.3 meter as an example.
    success = RewTerm(
        func=object_is_lifted,
        params={"minimum_height": 0.3, "obs_name": "cube_a_position"}, # Table plane as z = 0
        weight=10.0,
    )