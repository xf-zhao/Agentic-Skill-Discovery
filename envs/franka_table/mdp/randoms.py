from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from omni.isaac.orbit.assets import Articulation, RigidObject
from omni.isaac.orbit.managers import SceneEntityCfg
from omni.isaac.orbit.utils.math import quat_from_euler_xyz

if TYPE_CHECKING:
    from omni.isaac.orbit.envs import BaseEnv


def reset_root_state_group_uniform(
    env: BaseEnv,
    env_ids: torch.Tensor,
    pose_range: dict[str, tuple[float, float]],
    velocity_range: dict[str, tuple[float, float]],
    asset_cfgs: tuple[SceneEntityCfg,] = (SceneEntityCfg("object"),),
    min_distance=0.1,
):
    if hasattr(env, 'no_random'):
        return
    asset: RigidObject | Articulation = env.scene[asset_cfgs[0].name]
    root_states = asset.data.default_root_state[env_ids].clone()
    offsets = _compute_root_state_uniform(
        root_states, pose_range=pose_range, min_distance=min_distance
    )
    for offset, asset_cfg in zip(offsets, asset_cfgs):
        _reset_root_state_uniform_with_offset(
            env=env,
            env_ids=env_ids,
            pos_offset=offset,
            pose_range=pose_range,
            velocity_range=velocity_range,
            asset_cfg=asset_cfg,
        )


def _compute_root_state_uniform(
    root_states,
    pose_range: dict[str, tuple[float, float]],
    min_distance=0.1,
):
    # positions
    num_envs_needs_to_reset = root_states.shape[0]
    N = num_envs_needs_to_reset * 1024
    pos_offset = torch.zeros(N, 3 * 3).cuda()  # 3 objects x 3 xyz
    pos_offset[:, 0::3].uniform_(*pose_range.get("x", (0.0, 0.0)))
    pos_offset[:, 1::3].uniform_(*pose_range.get("y", (0.0, 0.0)))
    pos_offset[:, 2::3].uniform_(*pose_range.get("z", (0.0, 0.0)))
    dist0 = torch.norm(pos_offset[:, :3] - pos_offset[:, 3:6], dim=-1).unsqueeze(-1)
    dist1 = torch.norm(pos_offset[:, :3] - pos_offset[:, 6:], dim=-1).unsqueeze(-1)
    dist2 = torch.norm(pos_offset[:, 3:6] - pos_offset[:, 6:], dim=-1).unsqueeze(-1)
    dists = torch.cat([dist0, dist1, dist2], dim=-1)
    dist_conditions = dists.min(axis=-1).values >= min_distance
    position_offsets = pos_offset[dist_conditions][:num_envs_needs_to_reset]
    # print(f'position_offsets shape: {position_offsets.shape}')
    # It starts with (num_envs, 3), then (num_envs_needs_to_reset, 3)
    return position_offsets[:, :3], position_offsets[:, 3:6], position_offsets[:, 6:]


def _reset_root_state_uniform_with_offset(
    env: BaseEnv,
    env_ids: torch.Tensor,
    pos_offset: torch.Tensor,
    pose_range: dict[str, tuple[float, float]],
    velocity_range: dict[str, tuple[float, float]],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]
    # get default root state
    root_states = asset.data.default_root_state[env_ids].clone()
    positions = root_states[:, 0:3] + env.scene.env_origins[env_ids] + pos_offset
    # orientations
    euler_angles = torch.zeros_like(positions)
    euler_angles[:, 0].uniform_(*pose_range.get("roll", (0.0, 0.0)))
    euler_angles[:, 1].uniform_(*pose_range.get("pitch", (0.0, 0.0)))
    euler_angles[:, 2].uniform_(*pose_range.get("yaw", (0.0, 0.0)))
    orientations = quat_from_euler_xyz(
        euler_angles[:, 0], euler_angles[:, 1], euler_angles[:, 2]
    )
    # velocities
    velocities = root_states[:, 7:13]
    velocities[:, 0].uniform_(*velocity_range.get("x", (0.0, 0.0)))
    velocities[:, 1].uniform_(*velocity_range.get("y", (0.0, 0.0)))
    velocities[:, 2].uniform_(*velocity_range.get("z", (0.0, 0.0)))
    velocities[:, 3].uniform_(*velocity_range.get("roll", (0.0, 0.0)))
    velocities[:, 4].uniform_(*velocity_range.get("pitch", (0.0, 0.0)))
    velocities[:, 5].uniform_(*velocity_range.get("yaw", (0.0, 0.0)))

    # set into the physics simulation
    asset.write_root_pose_to_sim(
        torch.cat([positions, orientations], dim=-1), env_ids=env_ids
    )
    asset.write_root_velocity_to_sim(velocities, env_ids=env_ids)
