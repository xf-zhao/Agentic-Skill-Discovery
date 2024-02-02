# Copyright (c) 2022-2023, The ORBIT Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint if an RL agent from RSL-RL."""

from __future__ import annotations

"""Launch Isaac Sim Simulator first."""


from tqdm import tqdm
import argparse
import os
import json

from omni.isaac.orbit.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument(
    "--cpu", action="store_true", default=False, help="Use CPU pipeline."
)
parser.add_argument(
    "--onnx", action="store_true", default=False, help="Export onnx model."
)
parser.add_argument(
    "--num_envs", type=int, default=None, help="Number of environments to simulate."
)
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--seed", type=int, default=None, help="Seed used for the environment"
)
parser.add_argument(
    "--video", action="store_true", default=False, help="Record videos during training."
)
parser.add_argument("--log_root", type=str, default=None, help="Saved model path.")
parser.add_argument(
    "--video_length",
    type=int,
    default=250,
    help="Length of the recorded video (in steps).",
)
parser.add_argument(
    "--video_interval",
    type=int,
    default=250,
    help="Interval between video recordings (in steps).",
)
parser.add_argument(
    "--steps",
    type=int,
    default=250,
    help="Total steps.",
)

from eurekaplus.utils.misc import set_freest_gpu

set_freest_gpu()

# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# load cheaper kit config in headless
if args_cli.headless:
    app_experience = f"{os.environ['EXP_PATH']}/omni.isaac.sim.python.gym.headless.kit"
else:
    app_experience = f"{os.environ['EXP_PATH']}/omni.isaac.sim.python.kit"

# launch omniverse app
app_launcher = AppLauncher(args_cli, experience=app_experience)
simulation_app = app_launcher.app

"""Rest everything follows."""


import gymnasium as gym
import os
import torch
import traceback

import carb
from rsl_rl.runners import OnPolicyRunner

import omni.isaac.contrib_tasks  # noqa: F401
import omni.isaac.orbit_tasks  # noqa: F401
from omni.isaac.orbit_tasks.utils import get_checkpoint_path, parse_env_cfg
from omni.isaac.orbit_tasks.utils.wrappers.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlVecEnvWrapper,
    export_policy_as_onnx,
)

import sys

dirname = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, dirname)

import envs, envs_gpt  # noqa: F401


def main():
    """Play with RSL-RL agent."""
    # parse configuration
    env_cfg = parse_env_cfg(
        args_cli.task, use_gpu=not args_cli.cpu, num_envs=args_cli.num_envs
    )
    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(
        args_cli.task, args_cli
    )

    # specify directory for logging experiments
    log_root_path = (
        os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
        if args_cli.log_root is None
        else args_cli.log_root
    )
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    resume_path = get_checkpoint_path(
        log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint
    )
    log_dir = resume_path.replace(".pt", "_videos")
    print(f"[INFO]: Loading model checkpoint from: {resume_path}")

    # adjust camera resolution and pose
    env_cfg.viewer.resolution = (512, 512)
    env_cfg.viewer.eye = (0.3, 1.0, 1.0)
    env_cfg.viewer.lookat = (0.5, 0.0, 0.0)

    # create isaac environment
    env = gym.make(
        args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None
    )
    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": log_dir,
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            # "video_length": args_cli.video_length,
            "video_length": env.unwrapped.max_episode_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during playing.")
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env)

    # load previously trained model
    ppo_runner = OnPolicyRunner(
        env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device
    )
    ppo_runner.load(resume_path)
    print(f"[INFO]: Loading model checkpoint from: {resume_path}")

    # obtain the trained policy for inference
    policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)

    # export policy to onnx
    if args_cli.onnx:
        export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
        export_policy_as_onnx(
            ppo_runner.alg.actor_critic, export_model_dir, filename="policy.onnx"
        )
        print(f"[INFO]: Exported policy.onnx")

    # reset environment
    user_obss = []
    obs, _ = env.get_observations()
    print("[INFO]: reset env. Start simulating next step.")
    # simulate environment
    # while simulation_app.is_running():
    for i in tqdm(range(env.max_episode_length)):
        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping
            actions = policy(obs)
            # env stepping
            obs, _, _, extras = env.step(actions)
            if i in (6, env.max_episode_length - 10):
                data = extras["observations"]["observations"]
                data_str = {}
                for k, v in data.items():
                    if k in ("actions",):
                        continue
                    if v.reshape(-1).shape[0] > 1:
                        _v = (
                            "["
                            + ", ".join(
                                [f"{vv:.2f}" for vv in v.squeeze().cpu().numpy()]
                            )
                            + "]"
                        )
                    else:
                        _v = f"{v.squeeze().cpu().numpy():.2f}"
                    data_str[k] = _v
                user_obss.append(data_str)

    # close the simulator
    env.close()

    obs_json = {"first_frame": user_obss[0], "end_frame": user_obss[-1]}
    obs_json_str = json.dumps(obs_json)
    obs_path = f"{log_dir}/rl-video-step-0-obs.json"
    with open(obs_path, "w") as obsf:
        obsf.write(obs_json_str)
    return


if __name__ == "__main__":
    try:
        # run the main execution
        main()
    except Exception as err:
        err_ = traceback.format_exc()
        carb.log_error(err)
        carb.log_error(err_)
        print(err)
        print(err_)
        raise
    finally:
        # close sim app
        simulation_app.close()
