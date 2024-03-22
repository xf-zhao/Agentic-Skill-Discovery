from __future__ import annotations
from omni.isaac.orbit.utils import configclass
from omni.isaac.orbit.managers import SceneEntityCfg
from omni.isaac.orbit.managers import RewardTermCfg as RewTerm
from omni.isaac.orbit.managers import SceneEntityCfg
from omni.isaac.orbit.envs import RLTaskEnv
import torch

from envs.franka_table import mdp
from .success import SuccessCfg
from .termination import TerminationsCfg


def get_terminate_penalty(terminate_item):
    return RewTerm(
        func=terminate_item.func,
        params=terminate_item.params,
        weight=-10.0,
    )


def cube_a_ee_distance(
    env: RLTaskEnv,
    std: float = 0.1,
) -> torch.Tensor:
    # Distance of the end-effector to the object: (num_envs,)
    obs = env.obs_buf["observations"]
    distance = torch.norm(obs["cube_a_position"] - obs["ee_position"], dim=1)
    return 1 - torch.tanh(distance / std)


def object_is_grasped(
    env: RLTaskEnv,
) -> torch.Tensor:
    """Reward the agent for grasping the object"""
    obs = env.obs_buf["observations"]
    cube_a_ee_distance = torch.norm(obs["cube_a_position"] - obs["ee_position"], dim=1)
    is_near_cube = torch.where(cube_a_ee_distance < 0.02, 1.0, 0.0)
    encourage_close_reward = is_near_cube * (0.04 - obs["gripper_open_distance"].squeeze())
    return encourage_close_reward


def object_is_lifted(env: RLTaskEnv) -> torch.Tensor:
    """Reward the agent for lifting the object above the minimal height."""
    obs = env.obs_buf["observations"]
    minimal_height = 0.06
    cube_a_height = obs["cube_a_position"][:, 2]
    lifted_reward = torch.where(cube_a_height > minimal_height, 1.0, 0.0)
    return lifted_reward


@configclass
class RewardsCfg:
    try:
        success = SuccessCfg().success
    except:
        success = list(SuccessCfg().__dict__.values())[0]
    terminate_1 = get_terminate_penalty(TerminationsCfg().cube_a_dropping)
    terminate_2 = get_terminate_penalty(TerminationsCfg().cube_b_dropping)
    terminate_3 = get_terminate_penalty(TerminationsCfg().plate_dropping)

    reaching_object = RewTerm(
        func=cube_a_ee_distance,
        weight=1.0,
    )
    grasping_object = RewTerm(
        func=object_is_grasped,
        weight=25.0,
    )
    lifting_object = RewTerm(func=object_is_lifted, weight=10.0)

    # action penalty
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-1e-4)
    joint_vel = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-1e-4,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )


# @configclass
# class DrawerRewardsCfg:
#     """Reward terms for the MDP."""

#     dropping_cube_a = RewTerm(
#         func=mdp.base_height,
#         params={"minimum_height": -0.05, "asset_cfg": SceneEntityCfg("cube_a")},
#         weight=-10.0,
#     )
#     dropping_cube_b = RewTerm(
#         func=mdp.base_height,
#         params={"minimum_height": -0.05, "asset_cfg": SceneEntityCfg("cube_b")},
#         weight=-10.0,
#     )
#     dropping_plate = RewTerm(
#         func=mdp.base_height,
#         params={"minimum_height": -0.05, "asset_cfg": SceneEntityCfg("plate")},
#         weight=-10.0,
#     )

#     reaching_drawer = RewTerm(
#         func=mdp.drawer_ee_distance,
#         params={"std": 0.3, "object_cfg": SceneEntityCfg("cabinet")},
#         weight=1.0,
#     )

#     grasping_drawer = RewTerm(
#         func=mdp.drawer_is_grasped,
#         params={"object_cfg": SceneEntityCfg("cabinet")},
#         weight=25.0,
#     )

#     dragging_drawer = RewTerm(
#         func=mdp.drawer_is_dragged,
#         params={"object_cfg": SceneEntityCfg("cabinet"), "minimal_distance": 0.01},
#         weight=25.0,
#     )

#     # action penalty
#     action_rate = RewTerm(func=mdp.action_rate_l2, weight=-1e-4)
#     joint_vel = RewTerm(
#         func=mdp.joint_vel_l2,
#         weight=-1e-4,
#         params={"asset_cfg": SceneEntityCfg("robot")},
#     )
