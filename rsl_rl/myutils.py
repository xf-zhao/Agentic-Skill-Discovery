import torch
from rsl_rl.runners import OnPolicyRunner
from omni.isaac.orbit_tasks.utils import get_checkpoint_path


def set_current_status_as_default(env):
    # rigid bodies
    for rigid_object in env.env.scene.rigid_objects.values():
        # obtain default and deal with the offset for env origins
        default_root_state = rigid_object._data.root_state_w.clone()
        default_root_state[:, 0:3] -= env.env.scene.env_origins
        # set into the physics simulation
        rigid_object._data.default_root_state = default_root_state
    # articulations
    for articulation_asset in env.env.scene.articulations.values():
        # obtain default and deal with the offset for env origins
        default_root_state = articulation_asset._data.root_state_w.clone()
        default_root_state[:, 0:3] -= env.env.scene.env_origins
        # obtain default joint positions
        default_joint_pos = articulation_asset._data.joint_pos.clone()
        default_joint_vel = articulation_asset._data.joint_vel.clone()
        articulation_asset._data.default_root_state = default_root_state
        articulation_asset._data.default_joint_pos = default_joint_pos
        articulation_asset._data.default_joint_vel = default_joint_vel
    env.env.episode_length_buf = torch.zeros(
        env.env.num_envs, device=env.env.device, dtype=torch.long
    )
    env.env.no_random = True
    env.env.env.no_random = True
    return


def set_with_precedents(env, agent_cfg, log_dir, precedents=None):
    if precedents is not None:
        pre_runner = OnPolicyRunner(
            env,
            agent_cfg.to_dict(),
            log_dir=log_dir,
            device=agent_cfg.device,
        )
        for ith, precedent in enumerate(precedents):
            precedent_resume = get_checkpoint_path(
                precedent, agent_cfg.load_run, agent_cfg.load_checkpoint
            )
            print(
                f"[INFO]: Loading model checkpoint from: {precedent_resume} for the {ith}-th precedent skill policy."
            )
            pre_runner.load(precedent_resume)

            # obtain the trained policy for inference
            policy = pre_runner.get_inference_policy(device=env.unwrapped.device)

            obs, _ = env.get_observations()
            print("[INFO]: prepared env. Start simulating next step.")
            for istep in range(50):
                # run everything in inference mode
                with torch.inference_mode():
                    # agent stepping
                    actions = policy(obs)
                    # env stepping
                    obs, *_ = env.step(actions)
            set_current_status_as_default(env)
