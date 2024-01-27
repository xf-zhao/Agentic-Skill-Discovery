from __future__ import annotations
from omni.isaac.orbit.envs import RLTaskEnvCfg
from omni.isaac.orbit.managers import ObservationGroupCfg as ObsGroup
from omni.isaac.orbit.managers import ObservationTermCfg as ObsTerm
from omni.isaac.orbit.managers import SceneEntityCfg
from omni.isaac.orbit.utils import configclass
from omni.isaac.orbit.managers import SceneEntityCfg
import torch

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
        target_pose_on_table = ObsTerm(
            func=mdp.generated_commands, params={"command_name": "target_pose_on_table"}
        )
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class MyObservationCfg(ObsGroup):
        """My Observations that can be used to compose reward functions.

        Scene setting description: This is a table top manipulation environment. Here are the elements:
            - The table is with a range of: x in [0.0, 1.0], y in [-0.45, 0.45], z=0 (table top plane as the zero point).
            - A Franka robot is sitting on top of the table with a reaching range mentioned as the table surface. The robot has a gripper on its end-effector that can open or close.
            - The cabinet is near the table and fixed, with only one drawer visible and accessable by the robot. The drawer sits a litte bit higher than the table.
            - Two cubes with the same size but different appearance and id (cube A and cube B). The cubes' edge length are both 0.08 meter. The positions of cubes are randomly initialized within the table range every reset.
            - One rounded plate, with a radius of 0.1 meter. The position of the plate is randomly initialized every time of reset within the table range.
            - The robot gripper is initially open. It is bigger than the cubes but smaller than plates and the drawer.
            - The variable `target_pose_on_table` indicates a specific target position on the table, which is meant for goal-conditioned policy learning. Also use this one instead of coming up with a new variable. This pose variable contains 3 position and 4 quaternions.

        Use obs_buf and prev_obs_buf to access current and previous observations, respectively.
        For example:

        obs = env.obs_buf["observations"]
        prev_obs = env.prev_obs_buf["observations"]
        cube_a_height = obs["cube_a_position"][:, 2]
        prev_cube_a_height = prev_obs["cube_a_position"][:, 2]

        """

        # robot joint positions, in a shape of (9, ), including 7 DoF and 2 for the two finger-gripper joints.
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)

        # robot joint velocities, in a shape of (9, ), including 7 DoF and 2 for the two finger-gripper joints.
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)

        # robot end-effector position
        ee_position = ObsTerm(func=mdp.ee_position_in_robot_root_frame)

        # Robot end-effector (two-finger gripper) open distance
        # Tensor in a shape (num_envs, 1), from min 0 (closed) to max 0.04 (open)
        gripper_open_distance = ObsTerm(func=mdp.gripper_open_distance)

        # cube A position, Tensor in a shape (num_envs, 3)
        cube_a_position = ObsTerm(
            func=mdp.object_position_in_robot_root_frame,
            params={"object_cfg": SceneEntityCfg("cube_a")},
        )

        # cube B position, Tensor in a shape (num_envs, 3)
        cube_b_position = ObsTerm(
            func=mdp.object_position_in_robot_root_frame,
            params={"object_cfg": SceneEntityCfg("cube_b")},
        )

        # plate position, Tensor in a shape (num_envs, 3)
        plate_position = ObsTerm(
            func=mdp.object_position_in_robot_root_frame,
            params={"object_cfg": SceneEntityCfg("plate")},
        )

        # The drawer handle position (constrained by the drawer primistic joint), Tensor in a shape (num_envs, 3)
        drawer_handle_position = ObsTerm(
            func=mdp.drawer_position_in_robot_root_frame,
            params={"asset_cfg": SceneEntityCfg("cabinet")},
        )

        # to determine whether the drawer is open or not. Closed: 0; open: > 0 values.
        # Tensor in a shape (num_envs, 1)
        drawer_open_distance = ObsTerm(
            func=mdp.drawer_joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("cabinet")},
        )

        # Randomly initialized target pose to play with. It is in the shape of (num_envs, 7).
        # Later use `env.obs_buf['observations']['target_pose_on_table'][:, :3]` for position and `env.obs_buf['observations']['target_pose_on_table'][:, 3:]` for quatenion orentation inside the function computation.
        target_pose_on_table = ObsTerm(
            func=mdp.generated_commands, params={"command_name": "target_pose_on_table"}
        )

        # the last action the robot has taken
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    # observation group only for policy learning
    policy: PolicyCfg = PolicyCfg()

    # observation group for users, e.g. reward functions
    observations: MyObservationCfg = MyObservationCfg()
