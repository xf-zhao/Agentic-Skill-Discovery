# Copyright (c) 2022-2023, The ORBIT Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from omni.isaac.orbit.assets import RigidObject
from omni.isaac.orbit.managers import SceneEntityCfg
from omni.isaac.orbit.sensors import FrameTransformer
from omni.isaac.orbit.utils.math import combine_frame_transforms

if TYPE_CHECKING:
    from omni.isaac.orbit.envs import RLTaskEnv

def drawer_is_grasped(
    env: RLTaskEnv, object_cfg: SceneEntityCfg = SceneEntityCfg("cabinet"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg('robot'),
) -> torch.Tensor:
    """Reward the agent for grasping the object"""
    cabinet: RigidObject = env.scene[object_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    drawer_pos_w = cabinet.data.body_pos_w[:, -1]
    ee_w = ee_frame.data.target_pos_w[..., 0, :]
    object_ee_distance = torch.norm(drawer_pos_w - ee_w, dim=1)
    is_near = torch.where(object_ee_distance < 0.02, 1.0, 0.0)
    # Knonw that env.scene['robot'].data.body_names[:-2] -> ['panda_leftfinger', 'panda_rightfinger']
    left_right_finger_pos = env.scene[robot_cfg.name].data.joint_pos[:, -2:].mean(axis=-1)
    encourage_close_reward = is_near * (0.04 - left_right_finger_pos)
    # print(f'{object_ee_distance.min()} at {object_ee_distance.argmin()}')
    return encourage_close_reward



def drawer_is_dragged(
    env: RLTaskEnv, minimal_distance: float=0.01, object_cfg: SceneEntityCfg = SceneEntityCfg("cabinet")
) -> torch.Tensor:
    """Reward the agent for lifting the object above the minimal height."""
    cabinet: RigidObject = env.scene[object_cfg.name]
    reward = torch.where(cabinet.data.joint_pos[:, -1] > minimal_distance, 1.0, 0.0)
    return reward

def drawer_ee_distance(
    env: RLTaskEnv,
    std: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("cabinet"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Reward the agent for reaching the object using tanh-kernel."""
    # extract the used quantities (to enable type-hinting)
    cabinet: RigidObject = env.scene[object_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    # Target object position: (num_envs, 3)
    drawer_pos_w = cabinet.data.body_pos_w[:, -1]
    # End-effector position: (num_envs, 3)
    ee_w = ee_frame.data.target_pos_w[..., 0, :]
    # Distance of the end-effector to the object: (num_envs,)
    object_ee_distance = torch.norm(drawer_pos_w - ee_w, dim=1)

    return 1 - torch.tanh(object_ee_distance / std)




def object_goal_distance(
    env: RLTaskEnv,
    std: float,
    minimal_height: float,
    command_name: str,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Reward the agent for tracking the goal pose using tanh-kernel."""
    # extract the used quantities (to enable type-hinting)
    robot: RigidObject = env.scene[robot_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    command = env.command_manager.get_command(command_name)
    # compute the desired position in the world frame
    des_pos_b = command[:, :3]
    des_pos_w, _ = combine_frame_transforms(robot.data.root_state_w[:, :3], robot.data.root_state_w[:, 3:7], des_pos_b)
    # distance of the end-effector to the object: (num_envs,)
    distance = torch.norm(des_pos_w - object.data.root_pos_w[:, :3], dim=1)
    # rewarded if the object is lifted above the threshold
    return (object.data.root_pos_w[:, 2] > minimal_height) * (1 - torch.tanh(distance / std))
