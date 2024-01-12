from __future__ import annotations

from dataclasses import MISSING

import omni.isaac.orbit.sim as sim_utils
from omni.isaac.orbit.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from omni.isaac.orbit.envs import RLTaskEnvCfg
from omni.isaac.orbit.managers import CurriculumTermCfg as CurrTerm
from omni.isaac.orbit.managers import ObservationGroupCfg as ObsGroup
from omni.isaac.orbit.managers import ObservationTermCfg as ObsTerm
from omni.isaac.orbit.managers import RandomizationTermCfg as RandTerm
from omni.isaac.orbit.managers import SceneEntityCfg
from omni.isaac.orbit.scene import InteractiveSceneCfg
from omni.isaac.orbit.sensors.frame_transformer.frame_transformer_cfg import (
    FrameTransformerCfg,
)
from omni.isaac.orbit.sim.spawners.from_files.from_files_cfg import (
    GroundPlaneCfg,
    UsdFileCfg,
)
from omni.isaac.orbit.utils import configclass
from omni.isaac.orbit.utils.assets import ISAAC_NUCLEUS_DIR
from omni.isaac.orbit.assets import ArticulationCfg, AssetBaseCfg
from omni.isaac.orbit.sim.schemas.schemas_cfg import (
    RigidBodyPropertiesCfg,
)
from omni.isaac.orbit.assets import RigidObjectCfg
from omni.isaac.orbit.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from omni.isaac.orbit.actuators import ImplicitActuatorCfg
from omni.isaac.orbit.sensors import FrameTransformer
from omni.isaac.orbit.utils.math import combine_frame_transforms
from omni.isaac.orbit.assets import RigidObject
from omni.isaac.orbit.managers import SceneEntityCfg
from omni.isaac.orbit.utils.math import subtract_frame_transforms
from omni.isaac.orbit.assets import Articulation, RigidObject
import torch
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from omni.isaac.orbit.envs import RLTaskEnv
    from omni.isaac.orbit.envs import BaseEnv, RLTaskEnv

from envs.franka_table import mdp


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy learning."""

        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        cube_a_position = ObsTerm(
            func=mdp.object_position_in_robot_root_frame,
            params={"object_cfg": SceneEntityCfg("cube_a")},
        )
        cube_b_position = ObsTerm(
            func=mdp.object_position_in_robot_root_frame,
            params={"object_cfg": SceneEntityCfg("cube_b")},
        )
        plate_position = ObsTerm(
            func=mdp.object_position_in_robot_root_frame,
            params={"object_cfg": SceneEntityCfg("plate")},
        )
        drawer_joint_position = ObsTerm(
            func=mdp.drawer_joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("cabinet")},
        )
        drawer_position = ObsTerm(
            func=mdp.drawer_position_in_robot_root_frame,
            params={"asset_cfg": SceneEntityCfg("cabinet")},
        )
        target_object_position = ObsTerm(
            func=mdp.generated_commands, params={"command_name": "object_pose"}
        )
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True


    @configclass
    class MyObservationCfg(ObsGroup):
        """My Observations that can be used to compose reward functions."""

        # robot joint positions
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)

        # robot joint velocities
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)

        # robot end-effector position
        ee_position = ObsTerm(func=mdp.ee_position_in_robot_root_frame)

        # robot end-effector two gripper open range
        gripper_open_range = ObsTerm(func=mdp.gripper_open_range)

        # cube A position
        cube_a_position = ObsTerm(
            func=mdp.object_position_in_robot_root_frame,
            params={"object_cfg": SceneEntityCfg("cube_a")},
        )

        # cube B position
        cube_b_position = ObsTerm(
            func=mdp.object_position_in_robot_root_frame,
            params={"object_cfg": SceneEntityCfg("cube_b")},
        )

        # plate position
        plate_position = ObsTerm(
            func=mdp.object_position_in_robot_root_frame,
            params={"object_cfg": SceneEntityCfg("plate")},
        )

        # drawer position
        drawer_position = ObsTerm(
            func=mdp.drawer_position_in_robot_root_frame,
            params={"asset_cfg": SceneEntityCfg("cabinet")},
        )

        # drawer prismatic joint position. Closed: 0; open: > 0 values.
        drawer_joint_position = ObsTerm(
            func=mdp.drawer_joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("cabinet")},
        )

        # # goal to reach
        # target_object_position = ObsTerm(
        #     func=mdp.generated_commands, params={"command_name": "object_pose"}
        # )

        # the last action the robot has taken
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    # observation group only for policy learning
    policy: PolicyCfg = PolicyCfg()

    # observation group for users, e.g. reward functions
    observations: MyObservationCfg = MyObservationCfg()
