import hydra
import numpy as np
import json
import psutil
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
import tempfile
from eurekaplus.utils.misc import *
from eurekaplus.utils.file_utils import find_files_with_substring, load_tensorboard_logs
from eurekaplus.utils.create_task import create_task
from eurekaplus.utils.extract_task_code import *
from typing import List


DUMMY_FAILURE = -10000.0
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
from omni.isaac.orbit.envs import RLTaskEnv
from envs.franka_table import mdp
import torch

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
from omni.isaac.orbit.envs import RLTaskEnv
from envs.franka_table import mdp
import torch

from .success import SuccessCfg
from .termination import TerminationsCfg


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

    success = SuccessCfg().success
    terminate_1 = get_terminate_penalty(TerminationsCfg().cube_a_dropping)
    terminate_2 = get_terminate_penalty(TerminationsCfg().cube_b_dropping)
    terminate_3 = get_terminate_penalty(TerminationsCfg().plate_dropping)

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
I have to think ..

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


def gpt_call(
    messages,
    model="gpt-3.5-turbo",
    n_samples=1,
    temperature=0,
):
    responses = []
    total_samples = 0
    total_completion_token = 0
    total_token = 0
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
    return responses, total_samples, total_completion_token, total_token


def extract_code_string(response):
    patterns = [
        r"```python(.*?)```",
        r"```(.*?)```",
        r'"""(.*?)"""',
        r'""(.*?)""',
        r'"(.*?)"',
    ]
    content = response["message"]["content"]
    # Regex patterns to extract python code enclosed in GPT response
    for pattern in patterns:
        code_string = re.findall(pattern, content, re.DOTALL)
        if code_string is not None:
            code_string = code_string[-1].strip()
            break
    code_string = content if not code_string else code_string
    return code_string


class Node:
    def __init__(
        self,
        messages=None,
        response=None,
        code=None,
        root_dir=None,
        iterations=1,
        env_name="franka_table",
        model="gpt-3.5-turbo",
        n_samples=1,
        temperature=0,
    ) -> None:
        self.root_dir = root_dir if root_dir is not None else ZEROHERO_ROOT_DIR
        self.prompt_dir = f"{self.root_dir}/eurekaplus/utils/prompts"
        self.env_name = env_name
        self.type = None
        self.parent = None
        self.children = []
        self.iterations = iterations
        self.runable = False
        self.model = model
        self.n_samples = n_samples
        self.temperature = temperature
        self.idx = None
        self.messages = messages
        self.response = response
        self.code = code
        self.summary = None
        self.ite = 1
        self.env_file = (
            f"{self.root_dir}/envs/{self.env_name}/env_cfg/{self.env_name}_env_cfg.py"
        )
        self.env_obs_file = (
            f"{self.root_dir}/envs/{self.env_name}/env_cfg/observation.py"
        )
        self.termination_file = (
            f"{self.root_dir}/envs/{self.env_name}/env_cfg/termination.py"
        )
        self.env_obs_code = self._extract_env_obs()

    def init(self):
        self.idx = f"{self.type[0]}{uuid.uuid4().hex[:8]}"
        return self

    def add_child(self, child):
        child.parent = self
        self.children.append(child)

    def propose(self):
        raise NotImplementedError

    def collect(self):
        raise NotImplementedError

    def reflect(self):
        raise NotImplementedError

    def unlink(self):
        raise NotImplementedError

    def remove(self):
        raise NotImplementedError

    def _extract_env_obs(self):
        env_obs_code_string = file_to_string(self.env_obs_file)
        pattern = r"class MyObservationCfg.*:(.*?)def __post_init__.*"
        code_string = re.search(pattern, env_obs_code_string, re.DOTALL)
        if code_string is not None:
            code_string = code_string.group(0).strip()
        return code_string

    def _loop_until_no_syntax_err(self, messages, response=None, replacements={}):
        messages = messages.copy()
        for _ in range(5):
            if response is None:
                response, *_ = gpt_call(
                    messages=messages,
                    model=self.model,
                    n_samples=1,
                    temperature=0,
                )[0]
            code = extract_code_string(response=response)
            no_err, err_feedback = self._syntax_examine(code)
            if not no_err:
                messages.extend(
                    [response["message"], self._wrap_user_message(err_feedback)]
                )
                response = None
            else:
                for k, v in replacements.items():
                    code = code.replace(k, v)
                break
        return messages, response, code, no_err

    def _syntax_examine(
        self,
        code: str,
        replacements={"RewTerm": "dict", "@configclass": "", "RLTaskEnv": "object"},
        prefix_codes="import torch\n\n",
        remove_temp=False,
    ):
        err_feedback = None
        for k, v in replacements.items():
            code = code.replace(k, v)
        try:
            with tempfile.NamedTemporaryFile(delete=False) as temp:
                code = prefix_codes + code
                temp.write(code.encode("utf-8"))
                logging.info(f"Examining syntax with temp file: {temp.name}...")
            subprocess.check_output(["python", temp.name], stderr=subprocess.STDOUT)
            if remove_temp:
                os.remove(temp.name)
            no_err = True
            logging.info(f"No syntax error, continue to build graph.")
        except subprocess.CalledProcessError as e:
            exec_msg = e.output.decode()
            traceback_msg = filter_traceback(exec_msg)
            assert traceback_msg != ""
            logging.warning(f"Temp syntax error found: {traceback_msg}, loop to fix.")
            err_feedback = self.execution_error_feedback.format(
                traceback_msg=traceback_msg
            )
            no_err = False
        return no_err, err_feedback

    def _wrap_system_message(self, content):
        return self._wrap_message(content=content, role="system")

    def _wrap_user_message(self, content):
        return self._wrap_message(content=content, role="user")

    def _wrap_assistant_message(self, content):
        return self._wrap_message(content=content, role="assistant")

    def _wrap_message(self, content, role="user"):
        return {"role": role, "content": content}


class RewardNode(Node):
    def __init__(self, num_envs=11, task=None, headless=False, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.type = "Reward"
        self.task = task
        self.headless = headless
        self.runable = True
        self.num_envs = num_envs
        self.rl_run = None
        self.rl_filepath = None
        self.policy_feedback = file_to_string(
            f"{self.prompt_dir}/reward/policy_feedback.txt"
        )
        self.code_feedback = file_to_string(
            f"{self.prompt_dir}/reward/code_feedback.txt"
        )
        self.cur_env_dir = None

    def init(self):
        super().init()
        cur_env_dir = f"{self.root_dir}/envs_gpt/{self.env_name}/{self.idx}"
        self.rl_filepath = (
            f"{self.parent.parent.idx}-{self.parent.idx}-{self.idx}-{self.ite}.txt"
        )
        self.cur_env_dir = cur_env_dir
        if os.path.exists(cur_env_dir):
            os.removedirs(cur_env_dir)
            logging.info(f"Remove directory {cur_env_dir}.")
        os.makedirs(cur_env_dir)

        shutil.copy(self.env_file, cur_env_dir)
        shutil.copy(self.env_obs_file, cur_env_dir)
        shutil.copy(self.termination_file, cur_env_dir)
        with open(f"{cur_env_dir}/__init__.py", "w") as f:
            f.write(MODULE_INIT.replace("UUID_HEX", self.idx))

        success_file = f"{cur_env_dir}/success.py"
        with open(success_file, "w") as file:
            file.write(SUCCESS_INIT)
            file.writelines(self.parent.code + "\n")

        reward_file = f"{cur_env_dir}/reward.py"
        with open(reward_file, "w") as file:
            file.write(REWARD_INIT)
            file.writelines(self.code + "\n")

        return self

    def unlink(self):
        # self.parent.children.pop(self's index)
        self.parent = None
        self.children = []
        return

    def remove(self):
        self.unlink()
        cur_env_dir = self.cur_env_dir
        if os.path.exists(cur_env_dir):
            os.removedirs(cur_env_dir)
            logging.info(f"Remove node {self.idx} and its directory {cur_env_dir}.")
        return

    def run(self):
        # Only run when memory is enough
        max_waiting = 60 * 60 * 3 // 10
        for i in range(
            max_waiting
        ):  # maximum 3 hour waiting time, long enough for finishing one run
            available_mem = psutil.virtual_memory().free / 1024 / 1024 / 1024
            if available_mem > 16:  # 16 GB is the minimum mem for a new instance
                break
            else:
                j = i % 60
                if j == 0:
                    logging.info(
                        f"Waiting for enough mem to run node {self.parent.parent.idx}-{self.parent.idx}-{self.idx}. Time elapsed: {j} x 10 minutes."
                    )
            time.sleep(10)
        assert i < max_waiting - 1

        # Find the freest GPU to run GPU-accelerated RL
        set_freest_gpu()

        # Execute the python file with flags
        rl_run_command = [
            f"{ORBIT_ROOT_DIR}/orbit.sh",
            "-p",
            f"{self.root_dir}/rsl_rl/train.py",
            "--task",
            f"{self.idx}",
            "--num_envs",
            f"{self.num_envs}",
        ]
        if self.headless:
            rl_run_command.append("--headless")
        with open(self.rl_filepath, "w") as f:
            self.rl_run = subprocess.Popen(
                rl_run_command,
                stdout=f,
                stderr=f,
            )
        self._block_until_training(log_status=True)
        return self

    # def propose(self):

    def summarize(self):
        summary = self._summarize_runlog()
        self.summary = summary
        return self

    def _block_until_training(self, log_status=False):
        # Ensure that the RL training has started before moving on
        while True:
            rl_log = file_to_string(self.rl_filepath)
            if "Learning iteration 0/" in rl_log or "Traceback" in rl_log:
                if log_status and "Learning iteration 0/" in rl_log:
                    logging.info(
                        f"Iteration {self.ite}: Code Run {self.idx} successfully training!"
                    )
                if log_status and "Traceback" in rl_log:
                    logging.info(
                        f"Iteration {self.ite}: Code Run {self.idx} execution error!"
                    )
                logging.info(f"Log at {self.rl_filepath}")
                break
        return

    def _summarize_runlog(self):
        self.rl_run.communicate()
        exec_success = False
        content = ""
        success = DUMMY_FAILURE
        summary = {
            "exec_success": exec_success,
            "content": content,
            "success": success,
        }

        try:
            with open(self.rl_filepath, "r") as f:
                stdout_str = f.read()
        except:
            content = self.parent.execution_error_feedback.format(
                traceback_msg="Code Run cannot be executed due to function signature error! Please re-write an entirely new reward function!"
            )
            return summary

        traceback_msg = filter_traceback(stdout_str)
        if traceback_msg == "":
            # If RL execution has no error, provide policy statistics feedback
            exec_success = True
            success_reward_key = "Episode Reward"
            lines = stdout_str.split("\n")
            for i, line in enumerate(lines):
                if line.startswith("Log Directory:"):
                    break
            tensorboard_logdir = line.split(":")[-1].strip()
            tensorboard_logs = load_tensorboard_logs(tensorboard_logdir)
            max_iterations = np.array(tensorboard_logs[success_reward_key]).shape[0]
            epoch_freq = max(int(max_iterations // 10), 1)

            content += self.policy_feedback.format(epoch_freq=epoch_freq)

            # Add reward components log to the feedback
            for metric in tensorboard_logs:
                if metric.startswith("Episode Reward/") and "/terminate_" not in metric:
                    metric_cur = [
                        "{:.2f}".format(x)
                        for x in tensorboard_logs[metric][::epoch_freq]
                    ]
                    metric_cur_max = max(tensorboard_logs[metric])
                    metric_cur_min = min(tensorboard_logs[metric])
                    metric_cur_mean = sum(tensorboard_logs[metric]) / len(
                        tensorboard_logs[metric]
                    )
                    if success_reward_key == metric:
                        success = metric_cur_max
                        metric_name = f"Reward component `{metric}`"
                    else:
                        metric_name = "Success score"
                        content += f"{metric_name}: {metric_cur}, Max: {metric_cur_max:.2f}, Mean: {metric_cur_mean:.2f}, Min: {metric_cur_min:.2f} \n"
        else:
            # Otherwise, provide execution traceback error feedback
            success = DUMMY_FAILURE
            content += self.execution_error_feedback.format(traceback_msg=traceback_msg)

        summary = {
            "exec_success": exec_success,
            "content": content,
            "success": success,
        }
        return summary


class SuccessNode(Node):
    def __init__(self, task=None, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.type = "Success"
        self.task = task
        self.initial_system = file_to_string(
            f"{self.prompt_dir}/reward/initial_system.txt"
        )
        self.initial_user = file_to_string(f"{self.prompt_dir}/reward/initial_user.txt")
        self.signature = file_to_string(f"{self.prompt_dir}/reward/signature.txt")
        self.execution_error_feedback = file_to_string(
            f"{self.prompt_dir}/reward/execution_error_feedback.txt"
        )
        self.code_feedback = file_to_string(
            f"{self.prompt_dir}/reward/code_feedback.txt"
        )
        self.code_output_tip = file_to_string(
            f"{self.prompt_dir}/reward/code_output_tip.txt"
        )
        self.max_success_overall = DUMMY_FAILURE
        self.max_reward_idx = None
        self.stats = {
            "max_success": [],
            "execute_rate": [],
            "max_reward_idx": [],
        }

    def init(self):
        super().init()
        initial_system = (
            self.initial_system.format(signature_string=self.signature)
            + self.code_output_tip
        )
        initial_user = self.initial_user.format(
            task_obs_code_string=self.env_obs_code,
            task_description=self.task.code,
        )
        self.messages = [
            self._wrap_system_message(initial_system),
            self._wrap_user_message(initial_user),
        ]
        return self

    def propose(self, num_envs=2048, headless=False) -> List[RewardNode]:
        self.children: List[RewardNode] = []
        responses, *_ = gpt_call(
            messages=self.messages,
            model=self.model,
            n_samples=self.n_samples,
            temperature=self.temperature,
        )
        if self.n_samples == 1:
            logging.info(f"GPT Output:\n " + responses[0]["message"]["content"] + "\n")

        for response in responses:
            messages, response, code, no_err = self._loop_until_no_syntax_err(
                messages=self.messages,
                response=response,
                replacements={REWARD_REPLACE_INPUT: REWARD_REPLACE_OUTPUT},
            )
            if not no_err:
                continue
            child = RewardNode(
                num_envs=num_envs,
                task=self.task,
                messages=messages,
                response=response,
                code=code,
                headless=headless,
            )
            self.add_child(child)
            child.init()
        return self.children

    def collect(self):
        for child in self.children:
            child.summarize()
        exec_successes = [child.summary["exec_success"] for child in self.children]
        any_success = np.sum(exec_successes) > 0
        # Repeat the iteration if all code generation failed
        if not any_success:  # and cfg.sample != 1:
            stat = {
                "execute_rate": 0.0,
                "max_success": DUMMY_FAILURE,
                "max_reward_idx": None,
            }
            logging.info(
                "All code generation failed! Repeat this iteration from the current message checkpoint!"
            )
            return any_success, stat

        successes = [child.summary["success"] for child in self.children]
        # Select the best code sample based on the success rate
        best_sample_idx = np.argmax(np.array(successes))
        best_node = self.children[best_sample_idx]
        for i, child in enumerate(self.children):
            if i != best_sample_idx:
                child.remove()
        best_node.unlink()
        self.children = []
        feedback = self._wrap_user_message(
            best_node.summary["content"] + self.code_feedback
        )
        self.messages = [*best_node.messages, best_node.response["message"], feedback]
        self.response = None

        # some statistic report
        max_success = best_node.summary["success"]
        execute_rate = np.sum(np.array(successes) >= 0.0) / self.n_samples

        # Update the best Eureka Output
        if max_success > self.max_success_overall:
            self.max_success_overall = max_success
            self.max_reward_idx = best_node.idx

        logging.info(
            f"Iteration {self.ite}: Max Success: {max_success}, Execute Rate: {execute_rate}"
        )
        logging.info(f"Iteration {self.ite}: Best Generation ID: {best_sample_idx}")
        logging.info(
            f"Iteration {self.ite}: GPT Output Content:\n"
            + best_node.response["message"]["content"]
            + "\n"
        )
        logging.info(
            f"Iteration {self.ite}: User Content:\n"
            + best_node.summary["content"]
            + "\n"
        )
        self.ite += 1
        self._collect_stat(execute_rate, max_success, self.max_reward_idx)
        return any_success, stat

    def analyze_stats(self):
        stats = self.stats
        # Plot the success rate
        fig, axs = plt.subplots(2, figsize=(6, 6))
        fig.suptitle(f"{self.task.code}")

        max_successes = stats["max_success"]
        execute_rates = stats["execute_rate"]
        max_reward_idxs = stats["max_reward_idx"]

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
            max_reward_idx=max_reward_idxs,
        )
        return

    def _collect_stat(self, execute_rate, max_success, max_reward_idx):
        stat = {
            "execute_rate": execute_rate,
            "max_success": max_success,
            "max_reward_idx": max_reward_idx,
        }
        for k, v in stat.items():
            self.stats[k].append(v)
        return


class TaskNode(Node):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.type = "Task"
        self.initial_system = file_to_string(
            f"{self.prompt_dir}/success/initial_system.txt"
        )
        self.initial_user = file_to_string(
            f"{self.prompt_dir}/success/initial_user.txt"
        )
        self.signature = file_to_string(f"{self.prompt_dir}/success/signature.txt")
        self.code_output_tip = file_to_string(
            f"{self.prompt_dir}/success/code_output_tip.txt"
        )
        self.execution_error_feedback = file_to_string(
            f"{self.prompt_dir}/success/execution_error_feedback.txt"
        )
        self.code_feedback = file_to_string(
            f"{self.prompt_dir}/success/code_feedback.txt"
        )

    def init(self):
        super().init()
        initial_system = (
            self.initial_system.format(signature_string=self.signature)
            + self.code_output_tip
        )
        initial_user = self.initial_user.format(
            task_obs_code_string=self.env_obs_code,
            task_description=self.code,
        )
        self.messages = [
            self._wrap_system_message(initial_system),
            self._wrap_user_message(initial_user),
        ]

        return self

    def propose(self, iterations=3, n_samples=3, temperature=0) -> List[SuccessNode]:
        responses, *_ = gpt_call(
            messages=self.messages,
            model=self.model,
            n_samples=self.n_samples,
            temperature=self.temperature,
        )
        if self.n_samples == 1:
            logging.info(f"GPT Output:\n " + responses[0]["message"]["content"] + "\n")

        for response in responses:
            messages, response, code, no_err = self._loop_until_no_syntax_err(
                messages=self.messages,
                response=response,
                replacements={"weight=": "weight=30.0, #"},
            )
            if not no_err:
                continue
            child: SuccessNode = SuccessNode(
                task=self,
                messages=messages,
                response=response,
                code=code,
                iterations=iterations,
                n_samples=n_samples,
                temperature=temperature,
            )
            self.add_child(child)
            child.init()
        return self.children


class EnvNode(Node):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.type = "Env"
        self.code = None
        self.initial_system = file_to_string(
            f"{self.prompt_dir}/task/initial_system.txt"
        )
        self.initial_user = file_to_string(f"{self.prompt_dir}/task/initial_user.txt")
        self.signature = file_to_string(f"{self.prompt_dir}/success/signature.txt")
        self.code_output_tip = file_to_string(
            f"{self.prompt_dir}/task/code_output_tip.txt"
        )

    def init(self):
        super().init()
        initial_system = self.initial_system + self.code_output_tip
        initial_user = self.initial_user.format(
            env_obs_code_string=self.env_obs_code,
        )
        self.messages = [
            self._wrap_system_message(initial_system),
            self._wrap_user_message(initial_user),
        ]
        return self

    def propose_fake(self, n_samples=1, temperature=0) -> List[TaskNode]:
        pattern = r"([Tt]ask\s+\d+:.*)"
        content = """
Task 1: Move Cube A to a specific target position on the table.
Task 2: Move Cube B to a specific target position on the table.
Task 3: Move the plate to a specific target position on the table.
Task 4: Open the drawer.
Task 5: Close the drawer.
Task 6: Pick up Cube A and place it inside the drawer.
Task 7: Pick up Cube B and place it inside the drawer.
Task 8: Pick up the plate and place it inside the drawer.
Task 9: Pick up Cube A and place it on top of Cube B.
Task 10: Pick up Cube B and place it on top of Cube A.
"""
        tasks = re.findall(pattern, content)

        for task in tasks:
            code = task.split(": ")[-1]
            child: TaskNode = TaskNode(
                code=code,
                n_samples=n_samples,
                temperature=temperature,
            )
            self.add_child(child)
            child.init()
        return self.children

    def propose(self, n_samples=1, temperature=0) -> List[TaskNode]:
        messages = self.messages
        responses, *_ = gpt_call(
            messages=messages,
            model=self.model,
            n_samples=1,
            temperature=self.temperature,
        )
        if self.n_samples == 1:
            logging.info(f"GPT Output:\n " + responses[0]["message"]["content"] + "\n")
        pattern = r"([Tt]ask\s+\d+:.*)"
        content = responses[0]["message"]["content"]
        tasks = re.findall(pattern, content)
        for task in tasks:
            code = task.split(": ")[-1]
            child: TaskNode = TaskNode(
                code=code, n_samples=n_samples, temperature=temperature
            )
            self.add_child(child)
            child.init()
        return self.children


@hydra.main(config_path="cfg", config_name="config", version_base="1.1")
def main(cfg):
    workspace_dir = Path.cwd()
    logging.info(f"Workspace: {workspace_dir}")
    logging.info(f"Project Root: {ZEROHERO_ROOT_DIR}")
    openai.api_key = os.getenv("OPENAI_API_KEY")
    env = cfg.env
    model = cfg.model
    logging.info(f"Using LLM: {model}")
    logging.info("Env: " + env.env_name)

    env_name = cfg.env.env_name.lower()
    num_envs = 11 if cfg.debug else cfg.num_envs
    env_node = EnvNode(
        env_name=env_name,
        model="gpt-3.5-turbo",
        # model="gpt-4-1106-preview",
        n_samples=1,
        temperature=cfg.temperature,
    ).init()

    # Eureka-plus generation loop
    for i in range(cfg.iteration):
        logging.info(f"Iteration {i}: Generating with {cfg.model}")
        task_nodes = env_node.propose_fake(n_samples=2, temperature=cfg.temperature)
        # task_nodes = env_node.propose()
        for task_node in task_nodes:
            break
        success_nodes = task_node.propose(
            n_samples=3, iterations=2, temperature=cfg.temperature
        )  # params for child init
        for i in range(2):
            for success_node in success_nodes:
                reward_nodes = success_node.propose(
                    num_envs=num_envs, headless=cfg.headless
                )
                for node in reward_nodes:
                    node.run()
            for success_node in success_nodes:
                success_node.collect()
        task_node.collect()  # check behavior caption

    # # Evaluate the best reward code many times
    # # if max_reward_idxs is None:
    #     logging.info("All iterations of code generation failed, aborting...")
    #     logging.info(
    #         "Please double check the output env_iter*_response*.txt files for repeating errors!"
    #     )
    #     exit()
    # logging.info(
    #     f"Task: {task}, Max Training Success {max_success_overall}, Correlation {max_success_reward_correlation_overall}, Best Reward Code Path: {max_reward_code_path}"
    # )
    # logging.info(f"Evaluating best reward code {cfg.num_eval} times")
    # shutil.copy(max_reward_code_path, output_file)

    # eval_runs = []
    # for i in range(cfg.num_eval):
    #     set_freest_gpu()

    #     # Execute the python file with flags
    #     rl_filepath = f"reward_code_eval{i}.txt"
    #     with open(rl_filepath, "w") as f:
    #         process = subprocess.Popen(
    #             [
    #                 "python",
    #                 "-u",
    #                 f"{ISAAC_ROOT_DIR}/train.py",
    #                 "hydra/output=subprocess",
    #                 f"task={task}{suffix}",
    #                 f"wandb_activate={cfg.use_wandb}",
    #                 f"wandb_entity={cfg.wandb_username}",
    #                 f"wandb_project={cfg.wandb_project}",
    #                 f"headless={not cfg.capture_video}",
    #                 f"capture_video={cfg.capture_video}",
    #                 "force_render=False",
    #                 f"seed={i}",
    #             ],
    #             stdout=f,
    #             stderr=f,
    #         )

    #     self._block_until_training(rl_filepath)
    #     eval_runs.append(process)

    # reward_code_final_successes = []
    # reward_code_correlations_final = []
    # for i, rl_run in enumerate(eval_runs):
    #     rl_run.communicate()
    #     rl_filepath = f"reward_code_eval{i}.txt"
    #     with open(rl_filepath, "r") as f:
    #         stdout_str = f.read()
    #     lines = stdout_str.split("\n")
    #     for i, line in enumerate(lines):
    #         if line.startswith("Tensorboard Directory:"):
    #             break
    #     tensorboard_logdir = line.split(":")[-1].strip()
    #     tensorboard_logs = load_tensorboard_logs(tensorboard_logdir)
    #     max_success = max(tensorboard_logs["consecutive_successes"])
    #     reward_code_final_successes.append(max_success)

    #     if "gt_reward" in tensorboard_logs and "gpt_reward" in tensorboard_logs:
    #         gt_reward = np.array(tensorboard_logs["gt_reward"])
    #         gpt_reward = np.array(tensorboard_logs["gpt_reward"])
    #         reward_correlation = np.corrcoef(gt_reward, gpt_reward)[0, 1]
    #         reward_code_correlations_final.append(reward_correlation)

    # logging.info(
    #     f"Final Success Mean: {np.mean(reward_code_final_successes)}, Std: {np.std(reward_code_final_successes)}, Raw: {reward_code_final_successes}"
    # )
    # logging.info(
    #     f"Final Correlation Mean: {np.mean(reward_code_correlations_final)}, Std: {np.std(reward_code_correlations_final)}, Raw: {reward_code_correlations_final}"
    # )
    # np.savez(
    #     "final_eval.npz",
    #     reward_code_final_successes=reward_code_final_successes,
    #     reward_code_correlations_final=reward_code_correlations_final,
    # )


if __name__ == "__main__":
    main()
