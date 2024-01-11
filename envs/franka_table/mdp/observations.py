# Copyright (c) 2022-2023, The ORBIT Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from omni.isaac.orbit.assets import RigidObject
from omni.isaac.orbit.managers import SceneEntityCfg
from omni.isaac.orbit.utils.math import subtract_frame_transforms
from omni.isaac.orbit.assets import Articulation, RigidObject
from omni.isaac.orbit.sensors import FrameTransformer
from omni.isaac.orbit.utils.math import combine_frame_transforms
from omni.isaac.orbit.assets import RigidObject
from omni.isaac.orbit.managers import SceneEntityCfg
from omni.isaac.orbit.utils.math import subtract_frame_transforms
from omni.isaac.orbit.assets import Articulation, RigidObject

if TYPE_CHECKING:
    from omni.isaac.orbit.envs import RLTaskEnv
    from omni.isaac.orbit.envs import BaseEnv, RLTaskEnv


def ee_position_in_robot_root_frame(env: RLTaskEnv) -> torch.Tensor:
    """The position of the object in the robot's root frame."""
    robot: RigidObject = env.scene["robot"]
    ee_frame: FrameTransformer = env.scene["ee_frame"]
    ee_pos_w = ee_frame.data.target_pos_w[..., 0, :]
    ee_pos_b, _ = subtract_frame_transforms(
        robot.data.root_state_w[:, :3], robot.data.root_state_w[:, 3:7], ee_pos_w
    )
    return ee_pos_b


def gripper_open_range(env: RLTaskEnv) -> torch.Tensor:
    # Knonw that env.scene['robot'].data.body_names[:-2] -> ['panda_leftfinger', 'panda_rightfinger']
    robot: RigidObject = env.scene["robot"]
    open_range = robot.data.joint_pos[:, -2:].mean(axis=-1).unsqueeze(-1)
    return open_range


def object_position_in_robot_root_frame(
    env: RLTaskEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """The position of the object in the robot's root frame."""
    robot: RigidObject = env.scene[robot_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    object_pos_w = object.data.root_pos_w[:, :3]
    object_pos_b, _ = subtract_frame_transforms(
        robot.data.root_state_w[:, :3], robot.data.root_state_w[:, 3:7], object_pos_w
    )
    return object_pos_b


def drawer_joint_pos_rel(
    env: BaseEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("cabinet"),
    selected_joints: tuple[str,] = ("drawer_handle_top_joint"),
) -> torch.Tensor:
    """The joint positions of the asset w.r.t. the default joint positions."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    drawer_joint_pos_rel = (
        asset.data.joint_pos[:, -1] - asset.data.default_joint_pos[:, -1]
    )
    return drawer_joint_pos_rel.unsqueeze(-1)


def drawer_position_in_robot_root_frame(
    env: RLTaskEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("cabinet"),
) -> torch.Tensor:
    """The position of the object in the robot's root frame."""
    robot: RigidObject = env.scene[robot_cfg.name]
    object: RigidObject = env.scene[asset_cfg.name]
    object_pos_w = object.data.body_pos_w[:, -1]  # drawer_top_handle
    object_pos_b, _ = subtract_frame_transforms(
        robot.data.root_state_w[:, :3], robot.data.root_state_w[:, 3:7], object_pos_w
    )
    return object_pos_b
