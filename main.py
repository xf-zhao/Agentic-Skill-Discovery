import hydra
import numpy as np
import json
import logging
import matplotlib.pyplot as plt
import os
import openai
import re
import subprocess
from pathlib import Path
import shutil
import time
import uuid
from eurekaplus.utils.misc import *
from eurekaplus.utils.file_utils import find_files_with_substring, load_tensorboard_logs
from eurekaplus.utils.create_task import create_task
from eurekaplus.utils.extract_task_code import *

ZEROHERO_ROOT_DIR = f"{os.getcwd()}"
ORBIT_ROOT_DIR = f"/data/xufeng/workspace/isaac/orbit"
ISAAC_ROOT_DIR = f"{ORBIT_ROOT_DIR}/_isaac_sim"

MODULE_INIT = """
import gymnasium as gym
from envs.franka_table.config.franka import agents
from . import franka_table_env_cfg


gym.register(
    id="UUID_HEX",
    entry_point="omni.isaac.orbit.envs:RLTaskEnv",
    kwargs={
        "env_cfg_entry_point": franka_table_env_cfg.FrankaTableEnvCfg,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.PlayCubePPORunnerCfg,
    },
    disable_env_checker=True,
)
"""

SUCCESS_INIT = """
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

"""


REWARD_INIT = """
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


def get_terminate_penalty(terminate_item):
    return RewTerm(
        func=terminate_item.func,
        params=terminate_item.params,
        weight=-10.0,
    )
"""

REWARD_REPLACE_INPUT = """
@configclass
class RewardsCfg:
"""

REWARD_REPLACE_OUTPUT = """
@configclass
class RewardsCfg:

    success = SuccessCfg.success
    terminate_1 = get_terminate_penalty(TerminationsCfg.cube_a_dropping)
    terminate_2 = get_terminate_penalty(TerminationsCfg.cube_b_dropping)
    terminate_3 = get_terminate_penalty(TerminationsCfg.plate_dropping)

"""

TERMINATION_INIT = """
from omni.isaac.orbit.utils import configclass
from omni.isaac.orbit.managers import SceneEntityCfg
from omni.isaac.orbit.managers import TerminationTermCfg as DoneTerm

from envs.franka_table import mdp


@configclass
class TerminationsCfg:

    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    cube_a_dropping = DoneTerm(
        func=mdp.base_height,
        params={"minimum_height": -0.05, "asset_cfg": SceneEntityCfg("cube_a")},
    )
    cube_b_dropping = DoneTerm(
        func=mdp.base_height,
        params={"minimum_height": -0.05, "asset_cfg": SceneEntityCfg("cube_b")},
    )
    plate_dropping = DoneTerm(
        func=mdp.base_height,
        params={"minimum_height": -0.05, "asset_cfg": SceneEntityCfg("plate")},
    )
"""

FAKE_LLM_REWARD = """
I have to think ...

Here is the reward function.

```python
object_name = "cube_a"


@configclass
class RewardsCfg:

    dropping_cube_a = RewTerm(
        func=mdp.base_height,
        params={"minimum_height": -0.05, "asset_cfg": SceneEntityCfg("cube_a")},
        weight=-10.0,
    )
    dropping_cube_b = RewTerm(
        func=mdp.base_height,
        params={"minimum_height": -0.05, "asset_cfg": SceneEntityCfg("cube_b")},
        weight=-10.0,
    )
    dropping_plate = RewTerm(
        func=mdp.base_height,
        params={"minimum_height": -0.05, "asset_cfg": SceneEntityCfg("plate")},
        weight=-10.0,
    )

    reaching_object = RewTerm(
        func=mdp.object_ee_distance,
        params={"std": 0.1, "object_cfg": SceneEntityCfg(object_name)},
        weight=1.0,
    )

    grasping_object = RewTerm(
        func=mdp.object_is_grasped,
        params={"object_cfg": SceneEntityCfg(object_name)},
        weight=25.0,
    )

    lifting_object = RewTerm(
        func=mdp.object_is_lifted,
        params={"minimal_height": 0.06, "object_cfg": SceneEntityCfg(object_name)},
        weight=10.0,
    )

    object_goal_tracking = RewTerm(
        func=mdp.object_goal_distance,
        params={
            "std": 0.3,
            "minimal_height": 0.06,
            "command_name": "object_pose",
            "object_cfg": SceneEntityCfg(object_name),
        },
        weight=16.0,
    )

    object_goal_tracking_fine_grained = RewTerm(
        func=mdp.object_goal_distance,
        params={
            "std": 0.05,
            "minimal_height": 0.06,
            "command_name": "object_pose",
            "object_cfg": SceneEntityCfg(object_name),
        },
        weight=5.0,
    )

    # action penalty
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-1e-4)
    joint_vel = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-1e-4,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
```

Please have a look.
"""


def gpt_compose_function(
    signature,
    env_obs_code_string,
    task_description,
    initial_system,
    initial_user,
    code_output_tip,
    model="gpt-3.5-turbo",
    n_samples=1,
    temperature=0,
):
    responses = []
    total_samples = 0
    total_completion_token = 0
    total_token = 0

    initial_system = (
        initial_system.format(task_success_signature_string=signature) + code_output_tip
    )
    initial_user = initial_user.format(
        task_obs_code_string=env_obs_code_string, task_description=task_description
    )
    messages = [
        {"role": "system", "content": initial_system},
        {"role": "user", "content": initial_user},
    ]
    for msg in messages:
        print(msg["content"])

    chunk_size = n_samples if "gpt-3.5" in model else 4
    while True:
        if total_samples >= n_samples:
            break
        for attempt in range(1000):
            try:
                response_cur = openai.ChatCompletion.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    n=chunk_size,
                )
                total_samples += chunk_size
                break
            except Exception as e:
                if attempt >= 10:
                    chunk_size = max(int(chunk_size / 2), 1)
                    print("Current Chunk Size", chunk_size)
                logging.info(f"Attempt {attempt+1} failed with error: {e}")
                time.sleep(1)
        if response_cur is None:
            logging.info("Code terminated due to too many failed attempts!")
            exit()
        responses.extend(response_cur["choices"])
        prompt_tokens = response_cur["usage"]["prompt_tokens"]
        total_completion_token += response_cur["usage"]["completion_tokens"]
        total_token += response_cur["usage"]["total_tokens"]

    # Logging Token Information
    logging.info(
        f">>> Prompt Tokens: {prompt_tokens}, Completion Tokens: {total_completion_token}, Total Tokens: {total_token}"
    )

    if n_samples == 1:
        logging.info(
            f"Iteration {iter}: GPT Output:\n "
            + responses[0]["message"]["content"]
            + "\n"
        )

    return responses, total_samples, total_completion_token, total_token


def extract_code_string(responses, replace_input=None, replace_output=None):
    patterns = [
        r"```python(.*?)```",
        r"```(.*?)```",
        r'"""(.*?)"""',
        r'""(.*?)""',
        r'"(.*?)"',
    ]
    code_runs = []
    for response in responses:
        content = response["message"]["content"]
        # Regex patterns to extract python code enclosed in GPT response
        for pattern in patterns:
            code_string = re.search(pattern, content, re.DOTALL)
            if code_string is not None:
                code_string = code_string.group(1).strip()
                break
        code_string = content if not code_string else code_string
        if replace_input is not None and replace_output is not None:
            code_string = code_string.replace(replace_input, replace_output)
        code_runs.append(code_string)
    return code_runs


@hydra.main(config_path="cfg", config_name="config", version_base="1.1")
def main(cfg):
    trace_history = {}

    workspace_dir = Path.cwd()
    logging.info(f"Workspace: {workspace_dir}")
    logging.info(f"Project Root: {ZEROHERO_ROOT_DIR}")

    openai.api_key = os.getenv("OPENAI_API_KEY")

    env = cfg.env
    env_description = cfg.env.description
    task_description = "Reach cube A."
    suffix = cfg.suffix
    model = cfg.model
    logging.info(f"Using LLM: {model}")
    logging.info("Env: " + env.env_name)
    logging.info("Env description: " + env_description)

    env_name = cfg.env.env_name.lower()
    env_file = f"{ZEROHERO_ROOT_DIR}/envs/{env_name}/env_cfg/{env_name}_env_cfg.py"
    env_obs_file = f"{ZEROHERO_ROOT_DIR}/envs/{env_name}/env_cfg/observation.py"
    termination_file = f"{ZEROHERO_ROOT_DIR}/envs/{env_name}/env_cfg/termination.py"

    env_obs_code_string = file_to_string(env_obs_file)
    # Loading all text prompts
    prompt_dir = f"{ZEROHERO_ROOT_DIR}/eurekaplus/utils/prompts"
    initial_system_success = file_to_string(f"{prompt_dir}/initial_system_success.txt")
    initial_user_success = file_to_string(f"{prompt_dir}/initial_user_success.txt")
    success_signature = file_to_string(f"{prompt_dir}/success_signature.txt")

    initial_system_reward = file_to_string(f"{prompt_dir}/initial_system_reward.txt")
    initial_user_reward = file_to_string(f"{prompt_dir}/initial_user_reward.txt")
    code_output_tip = file_to_string(f"{prompt_dir}/code_output_tip.txt")
    code_feedback = file_to_string(f"{prompt_dir}/code_feedback.txt")
    reward_signature = file_to_string(f"{prompt_dir}/reward_signature.txt")
    policy_feedback = file_to_string(f"{prompt_dir}/policy_feedback.txt")
    execution_error_feedback = file_to_string(
        f"{prompt_dir}/execution_error_feedback.txt"
    )

    DUMMY_FAILURE = -10000.0
    max_successes = []
    max_successes_reward_correlation = []
    execute_rates = []
    best_code_paths = []
    max_success_overall = DUMMY_FAILURE
    max_success_reward_correlation_overall = DUMMY_FAILURE
    max_reward_code_path = None

    # Eureka generation loop
    for iter in range(cfg.iteration):
        # Get Eureka-plus response
        # total_tokens = 0
        # total_samples = 0
        # total_completion_tokens = 0

        logging.info(f"Iteration {iter}: Generating with {cfg.model}")

        # Success condition function
        responses_success, *_ = gpt_compose_function(
            signature=success_signature,
            env_obs_code_string=env_obs_code_string,
            task_description=task_description,
            initial_system=initial_system_success,
            initial_user=initial_user_success,
            code_output_tip=code_output_tip,
            model=cfg.model,
            n_samples=cfg.n_success_samples,
            temperature=cfg.temperature,
        )
        codes_success = extract_code_string(responses_success)
        if len(codes_success) == 1:
            codes_success *= cfg.n_reward_samples

        # reward function
        responses_reward, *_ = gpt_compose_function(
            signature=reward_signature,
            env_obs_code_string=env_obs_code_string,
            task_description=task_description,
            initial_system=initial_system_reward,
            initial_user=initial_user_reward,
            code_output_tip=code_output_tip,
            model=cfg.model,
            n_samples=cfg.n_reward_samples,
            temperature=cfg.temperature,
        )
        codes_reward = extract_code_string(
            responses_reward,
            replace_input=REWARD_REPLACE_INPUT,
            replace_output=REWARD_REPLACE_OUTPUT,
        )

        # total_samples += total_sample
        # total_tokens +=total_token
        # total_completion_tokens+=total_completion_token

        rl_runs = []
        for i_sample, (code_success, code_reward) in enumerate(zip(codes_success, codes_reward)):

            logging.info(f"Iteration {iter}: Processing Code Run {i_sample}, Idx: {idx}")

            idx = f"UID{uuid.uuid4().hex[:16]}"
            cur_env_dir = f"{ZEROHERO_ROOT_DIR}/envs_gpt/{env_name}/{idx}"
            if not os.path.exists(cur_env_dir):
                os.makedirs(cur_env_dir, exist_ok=True)
            shutil.copy(env_file, cur_env_dir)
            shutil.copy(env_obs_file, cur_env_dir)
            shutil.copy(termination_file, cur_env_dir)
            with open(f"{cur_env_dir}/__init__.py", "w") as f:
                f.write(MODULE_INIT.replace("UUID_HEX", idx))

            reward_file = f"{cur_env_dir}/reward.py"
            with open(reward_file, "w") as file:
                file.write(REWARD_INIT)
                file.writelines(code_reward + "\n")

            success_file = f"{cur_env_dir}/success.py"
            with open(success_file, "w") as file:
                file.write(SUCCESS_INIT)
                file.writelines(code_success + "\n")

        pass
            # # Find the freest GPU to run GPU-accelerated RL
            # set_freest_gpu()

            # # Execute the python file with flags
            # rl_filepath = f"env_iter{iter}_response{response_id}.txt"
            # with open(rl_filepath, "w") as f:
            #     process = subprocess.Popen(
            #         [
            #             f"{ORBIT_ROOT_DIR}/orbit.sh",
            #             "-p",
            #             f"{ZEROHERO_ROOT_DIR}/rsl_rl/train.py",
            #             "--task",
            #             f"{idx}",
            #             "--num_envs",
            #             f"{cfg.num_envs}",
            #         ],
            #         stdout=f,
            #         stderr=f,
            #     )
            # block_until_training(
            #     rl_filepath, log_status=True, iter_num=iter, response_id=response_id
            # )
            # rl_runs.append(process)

        # Gather RL training results and construct reward reflection
        code_feedbacks = []
        contents = []
        successes = []
        reward_correlations = []
        code_paths = []

        exec_success = False
        for response_id, (code_run, rl_run) in enumerate(zip(code_runs, rl_runs)):
            rl_run.communicate()
            rl_filepath = f"env_iter{iter}_response{response_id}.txt"
            code_paths.append(f"env_iter{iter}_response{response_id}.py")
            try:
                with open(rl_filepath, "r") as f:
                    stdout_str = f.read()
            except:
                content = execution_error_feedback.format(
                    traceback_msg="Code Run cannot be executed due to function signature error! Please re-write an entirely new reward function!"
                )
                content += code_output_tip
                contents.append(content)
                successes.append(DUMMY_FAILURE)
                reward_correlations.append(DUMMY_FAILURE)
                continue

            content = ""
            traceback_msg = filter_traceback(stdout_str)

            if traceback_msg == "":
                # If RL execution has no error, provide policy statistics feedback
                exec_success = True
                lines = stdout_str.split("\n")
                for i, line in enumerate(lines):
                    if line.startswith("Tensorboard Directory:"):
                        break
                tensorboard_logdir = line.split(":")[-1].strip()
                tensorboard_logs = load_tensorboard_logs(tensorboard_logdir)
                max_iterations = np.array(tensorboard_logs["gt_reward"]).shape[0]
                epoch_freq = max(int(max_iterations // 10), 1)

                content += policy_feedback.format(epoch_freq=epoch_freq)

                # Compute Correlation between Human-Engineered and GPT Rewards
                if "gt_reward" in tensorboard_logs and "gpt_reward" in tensorboard_logs:
                    gt_reward = np.array(tensorboard_logs["gt_reward"])
                    gpt_reward = np.array(tensorboard_logs["gpt_reward"])
                    reward_correlation = np.corrcoef(gt_reward, gpt_reward)[0, 1]
                    reward_correlations.append(reward_correlation)

                # Add reward components log to the feedback
                for metric in tensorboard_logs:
                    if "/" not in metric:
                        metric_cur = [
                            "{:.2f}".format(x)
                            for x in tensorboard_logs[metric][::epoch_freq]
                        ]
                        metric_cur_max = max(tensorboard_logs[metric])
                        metric_cur_mean = sum(tensorboard_logs[metric]) / len(
                            tensorboard_logs[metric]
                        )
                        if "consecutive_successes" == metric:
                            successes.append(metric_cur_max)
                        metric_cur_min = min(tensorboard_logs[metric])
                        if metric != "gt_reward" and metric != "gpt_reward":
                            if metric != "consecutive_successes":
                                metric_name = metric
                            else:
                                metric_name = "task_score"
                            content += f"{metric_name}: {metric_cur}, Max: {metric_cur_max:.2f}, Mean: {metric_cur_mean:.2f}, Min: {metric_cur_min:.2f} \n"
                        else:
                            # Provide ground-truth score when success rate not applicable
                            if "consecutive_successes" not in tensorboard_logs:
                                content += f"ground-truth score: {metric_cur}, Max: {metric_cur_max:.2f}, Mean: {metric_cur_mean:.2f}, Min: {metric_cur_min:.2f} \n"
                code_feedbacks.append(code_feedback)
                content += code_feedback
            else:
                # Otherwise, provide execution traceback error feedback
                successes.append(DUMMY_FAILURE)
                reward_correlations.append(DUMMY_FAILURE)
                content += execution_error_feedback.format(traceback_msg=traceback_msg)

            content += code_output_tip
            contents.append(content)

        # Repeat the iteration if all code generation failed
        if not exec_success and cfg.sample != 1:
            execute_rates.append(0.0)
            max_successes.append(DUMMY_FAILURE)
            max_successes_reward_correlation.append(DUMMY_FAILURE)
            best_code_paths.append(None)
            logging.info(
                "All code generation failed! Repeat this iteration from the current message checkpoint!"
            )
            continue

        # Select the best code sample based on the success rate
        best_sample_idx = np.argmax(np.array(successes))
        best_content = contents[best_sample_idx]

        max_success = successes[best_sample_idx]
        max_success_reward_correlation = reward_correlations[best_sample_idx]
        execute_rate = np.sum(np.array(successes) >= 0.0) / cfg.sample

        # Update the best Eureka Output
        if max_success > max_success_overall:
            max_success_overall = max_success
            max_success_reward_correlation_overall = max_success_reward_correlation
            max_reward_code_path = code_paths[best_sample_idx]

        execute_rates.append(execute_rate)
        max_successes.append(max_success)
        max_successes_reward_correlation.append(max_success_reward_correlation)
        best_code_paths.append(code_paths[best_sample_idx])

        logging.info(
            f"Iteration {iter}: Max Success: {max_success}, Execute Rate: {execute_rate}, Max Success Reward Correlation: {max_success_reward_correlation}"
        )
        logging.info(f"Iteration {iter}: Best Generation ID: {best_sample_idx}")
        logging.info(
            f"Iteration {iter}: GPT Output Content:\n"
            + responses[best_sample_idx]["message"]["content"]
            + "\n"
        )
        logging.info(f"Iteration {iter}: User Content:\n" + best_content + "\n")

        # Plot the success rate
        fig, axs = plt.subplots(2, figsize=(6, 6))
        fig.suptitle(f"{cfg.env.task}")

        x_axis = np.arange(len(max_successes))

        axs[0].plot(x_axis, np.array(max_successes))
        axs[0].set_title("Max Success")
        axs[0].set_xlabel("Iteration")

        axs[1].plot(x_axis, np.array(execute_rates))
        axs[1].set_title("Execute Rate")
        axs[1].set_xlabel("Iteration")

        fig.tight_layout(pad=3.0)
        plt.savefig("summary.png")
        np.savez(
            "summary.npz",
            max_successes=max_successes,
            execute_rates=execute_rates,
            best_code_paths=best_code_paths,
            max_successes_reward_correlation=max_successes_reward_correlation,
        )

        if len(messages) == 2:
            messages += [
                {
                    "role": "assistant",
                    "content": responses[best_sample_idx]["message"]["content"],
                }
            ]
            messages += [{"role": "user", "content": best_content}]
        else:
            assert len(messages) == 4
            messages[-2] = {
                "role": "assistant",
                "content": responses[best_sample_idx]["message"]["content"],
            }
            messages[-1] = {"role": "user", "content": best_content}

        # Save dictionary as JSON file
        with open("messages.json", "w") as file:
            json.dump(messages, file, indent=4)

    # Evaluate the best reward code many times
    if max_reward_code_path is None:
        logging.info("All iterations of code generation failed, aborting...")
        logging.info(
            "Please double check the output env_iter*_response*.txt files for repeating errors!"
        )
        exit()
    logging.info(
        f"Task: {task}, Max Training Success {max_success_overall}, Correlation {max_success_reward_correlation_overall}, Best Reward Code Path: {max_reward_code_path}"
    )
    logging.info(f"Evaluating best reward code {cfg.num_eval} times")
    shutil.copy(max_reward_code_path, output_file)

    eval_runs = []
    for i in range(cfg.num_eval):
        set_freest_gpu()

        # Execute the python file with flags
        rl_filepath = f"reward_code_eval{i}.txt"
        with open(rl_filepath, "w") as f:
            process = subprocess.Popen(
                [
                    "python",
                    "-u",
                    f"{ISAAC_ROOT_DIR}/train.py",
                    "hydra/output=subprocess",
                    f"task={task}{suffix}",
                    f"wandb_activate={cfg.use_wandb}",
                    f"wandb_entity={cfg.wandb_username}",
                    f"wandb_project={cfg.wandb_project}",
                    f"headless={not cfg.capture_video}",
                    f"capture_video={cfg.capture_video}",
                    "force_render=False",
                    f"seed={i}",
                ],
                stdout=f,
                stderr=f,
            )

        block_until_training(rl_filepath)
        eval_runs.append(process)

    reward_code_final_successes = []
    reward_code_correlations_final = []
    for i, rl_run in enumerate(eval_runs):
        rl_run.communicate()
        rl_filepath = f"reward_code_eval{i}.txt"
        with open(rl_filepath, "r") as f:
            stdout_str = f.read()
        lines = stdout_str.split("\n")
        for i, line in enumerate(lines):
            if line.startswith("Tensorboard Directory:"):
                break
        tensorboard_logdir = line.split(":")[-1].strip()
        tensorboard_logs = load_tensorboard_logs(tensorboard_logdir)
        max_success = max(tensorboard_logs["consecutive_successes"])
        reward_code_final_successes.append(max_success)

        if "gt_reward" in tensorboard_logs and "gpt_reward" in tensorboard_logs:
            gt_reward = np.array(tensorboard_logs["gt_reward"])
            gpt_reward = np.array(tensorboard_logs["gpt_reward"])
            reward_correlation = np.corrcoef(gt_reward, gpt_reward)[0, 1]
            reward_code_correlations_final.append(reward_correlation)

    logging.info(
        f"Final Success Mean: {np.mean(reward_code_final_successes)}, Std: {np.std(reward_code_final_successes)}, Raw: {reward_code_final_successes}"
    )
    logging.info(
        f"Final Correlation Mean: {np.mean(reward_code_correlations_final)}, Std: {np.std(reward_code_correlations_final)}, Raw: {reward_code_correlations_final}"
    )
    np.savez(
        "final_eval.npz",
        reward_code_final_successes=reward_code_final_successes,
        reward_code_correlations_final=reward_code_correlations_final,
    )


if __name__ == "__main__":
    main()
