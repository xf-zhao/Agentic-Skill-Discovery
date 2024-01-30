import hydra
import numpy as np
import json
import networkx as nx
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
from behavior import BehaviorCaptioner, video_to_frames


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

    # chunk_size = n_samples if "gpt-3.5" in model else 4
    chunk_size = n_samples
    while True:
        if total_samples >= n_samples:
            break
        for attempt in range(10):
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
                err_msg = f"Attempt {attempt+1} failed with error: {e}"
                logging.info(err_msg)
                if "maximum context length" in err_msg:
                    responses = None
                    return
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
    return responses


def extract_code_string(response, combine_all=False):
    patterns = [
        r"```python(.*?)```",
        r"```(.*?)```",
        r'"""(.*?)"""',
        r'""(.*?)""',
        r'"(.*?)"',
    ]
    content = response["message"]["content"]
    # Regex patterns to extract python code enclosed in GPT response
    code_string = ""
    for pattern in patterns:
        code_strings = re.findall(pattern, content, re.DOTALL)
        if len(code_strings) > 0:
            if combine_all:
                for cs in code_strings:
                    code_string += cs.strip() + "\n"
            else:
                code_string = code_strings[
                    -1
                ].strip()  # assume the last is a combined one.
            break
        else:
            code_string = None
    return code_string


class Node:
    def __init__(
        self,
        type=None,
        messages=None,
        response=None,
        code=None,
        root_dir=None,
        iterations=1,
        env_name="franka_table",
        model="gpt-3.5-turbo",
        n_samples=1,
        temperature=0,
        ite=0,
    ) -> None:
        self.root_dir = root_dir if root_dir is not None else ZEROHERO_ROOT_DIR
        self.prompt_dir = f"{self.root_dir}/eurekaplus/utils/prompts"
        self.env_name = env_name
        self.type = type
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
        self.ite = ite
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
        self.code_format_feedback = file_to_string(
            f"{self.prompt_dir}/reward/code_format_feedback.txt"
        )

    def init(self):
        if self.idx is None:
            self.idx = f"{self.type[0]}{uuid.uuid4().hex[:8]}"
        self.children = []
        return self

    def add_child(self, child):
        child.parent = self
        self.children.append(child)

    def _update_self_with_node(self, node):
        self.type = node["type"]
        self.idx = node["id"]
        self.ite = node["ite"]
        self.messages = node["messages"]
        self.response = node["response"]
        self.code = node["code"]
        return

    def _add_node_to_graph(self, node):
        node_type = node.type
        data = dict(
            type=node_type,
            messages=node.messages,
            response=node.response,
            code=node.code,
            ite=node.ite,
        )
        if node_type == "Success":
            if node.best_reward is not None:
                data.update(
                    {
                        "best_reward": {
                            "idx": node.best_reward.idx,
                            "priors": node.best_reward.priors,
                            "summary": node.best_reward.summary,
                        },
                        "stats": node.stats,
                    }
                )
        elif node_type == "Task":
            if node.num_variants > 0:
                data.update({"variants": [variant.idx for variant in node.variants]})
        else:
            pass
        self.G.add_node(node.idx, **data)
        return

    def propose(self):
        raise NotImplementedError

    def collect(self):
        raise NotImplementedError

    def reflect(self):
        raise NotImplementedError

    def unlink(self):
        # self.parent.children.remove(self)
        self.parent = None
        self.children = []
        return

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
        init_messages = messages.copy()
        for __ in range(10):
            messages = init_messages.copy()
            gpt_err = False
            for _ in range(5):
                if response is None:
                    responses = gpt_call(
                        messages=messages,
                        model=self.model,
                        n_samples=1,
                        temperature=self.temperature,
                    )
                    if responses is None:
                        gpt_err = True
                        break
                    else:
                        gpt_err = False
                        response = responses[0]
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
            if not gpt_err:
                break
        return messages, response, code, no_err

    def _syntax_examine(
        self,
        code: str,
        replacements={
            "RewTerm": "dict",
            "@configclass": "",
            "RLTaskEnv": "object",
            "@torch.jit.script": "",
            "@staticmethod": "",
        },
        prefix_codes="import torch\n\n",
        remove_temp=False,
    ):
        if code is None:
            return False, self.code_format_feedback
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
    def __init__(
        self,
        num_envs=11,
        task=None,
        headless=False,
        video=False,
        memory_requirement=16,
        max_iterations=2000,
        priors=None,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.type = "Reward"
        self.task = task
        self.headless = headless
        self.video = video
        self.runable = True
        self.num_envs = num_envs
        self.rl_run = None
        self.rl_filepath = None
        self.play_filepath = None
        self.memory_requirement = memory_requirement
        self.policy_feedback = file_to_string(
            f"{self.prompt_dir}/reward/policy_feedback.txt"
        )
        self.code_feedback = file_to_string(
            f"{self.prompt_dir}/reward/code_feedback.txt"
        )
        self.execution_error_feedback = file_to_string(
            f"{self.prompt_dir}/reward/execution_error_feedback.txt"
        )
        self.cur_env_dir = None
        self.max_iterations = max_iterations
        self.priors = [] if priors is None else priors

    def init(self):
        super().init()
        cur_env_dir = f"{self.root_dir}/envs_gpt/{self.env_name}/{self.idx}"
        self.rl_filepath = (
            f"{self.parent.parent.idx}-{self.parent.idx}-{self.idx}-{self.ite}.txt"
        )
        self.cur_env_dir = cur_env_dir
        self.log_dir = f"{self.cur_env_dir}/logs"
        if os.path.exists(cur_env_dir):
            shutil.rmtree(cur_env_dir)
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

    def remove(self):
        self.unlink()
        cur_env_dir = self.cur_env_dir
        if os.path.exists(cur_env_dir):
            shutil.rmtree(cur_env_dir)
            logging.info(f"Removed node {self.idx} and its directory {cur_env_dir}.")
        return

    def _prepare_launch(self):
        # Only run when memory is enough
        max_waiting = 60 * 60 * 4 // 10
        for i in range(
            max_waiting
        ):  # Maximum 4 hour waiting time, long enough for finishing one run
            available_mem = psutil.virtual_memory().free / 1024 / 1024 / 1024
            if (
                available_mem > self.memory_requirement
            ):  # 16 GB is the minimum mem for a new instance
                break
            else:
                if i % 60 == 0:
                    logging.info(f"")
                    logging.info(
                        f"Available mem: {available_mem}. (Require {self.memory_requirement}). Waiting for enough mem to run node {self.parent.parent.idx}-{self.parent.idx}-{self.idx}. Time elapsed: {i//6} minutes."
                    )
                time.sleep(10)
        assert i < max_waiting - 1

        # Find the freest GPU to run GPU-accelerated RL
        set_freest_gpu()
        return

    def run(self):
        self._prepare_launch()

        # Execute the python file with flags
        rl_run_command = [
            f"{ORBIT_ROOT_DIR}/orbit.sh",
            "-p",
            f"{self.root_dir}/rsl_rl/train.py",
            "--task",
            f"{self.idx}",
            "--num_envs",
            f"{self.num_envs}",
            "--max_iterations",
            f"{self.max_iterations}",
            "--log_dir",
            os.path.dirname(self.log_dir),
        ]
        if self.headless:
            rl_run_command.append("--headless")
        if self.video:
            rl_run_command.append("--video")
            if self.headless:
                rl_run_command.append("--offscreen_render")
        print(f"Executing commands: {rl_run_command}")
        with open(self.rl_filepath, "w") as f:
            self.rl_run = subprocess.Popen(
                rl_run_command,
                stdout=f,
                stderr=f,
            )
        self._block_until_training()
        return self

    def play(self):
        self._prepare_launch()

        # Execute the python file with flags
        rl_run_command = [
            f"{ORBIT_ROOT_DIR}/orbit.sh",
            "-p",
            f"{self.root_dir}/rsl_rl/play.py",
            "--task",
            f"{self.idx}",
            "--num_envs",
            "1",
            "--video",
            "--log_root",
            self.log_dir,
        ]
        if self.headless:
            rl_run_command.append("--headless")
            rl_run_command.append("--offscreen_render")
        self.play_filepath = self.rl_filepath.rstrip(".txt") + "_play.txt"
        print(f"Executing commands: {rl_run_command}")
        with open(self.play_filepath, "w") as f:
            self.rl_run = subprocess.Popen(
                rl_run_command,
                stdout=f,
                stderr=f,
            )
        behavior_image_paths = self._block_until_play_recorded()
        return behavior_image_paths

    def summarize(self):
        summary = self._summarize_runlog()
        self.summary = summary
        return self

    def _block_until_training(self):
        # Ensure that the RL training has started before moving on
        while True:
            time.sleep(3)
            rl_log = file_to_string(self.rl_filepath)
            msg = filter_traceback(rl_log)
            if msg is None:
                continue
            elif msg == "":
                logging.info(
                    f"Iteration {self.iterations} - node {self.idx}: successfully launched RL training."
                )
            else:
                logging.error(
                    f"Iteration {self.iterations} - node {self.idx}: execution error!"
                )
            logging.info(f"Log at {self.rl_filepath}")
            break
        return

    def _block_until_play_recorded(self):
        image_paths = None
        for _ in range(60 * 5):  # 5 mins throw error
            time.sleep(1)
            play_video = self.log_dir + "/rl-video-step-0.mp4"
            if os.path.exists(play_video):
                image_paths = video_to_frames(play_video)
                break
        return image_paths

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
            content = self.execution_error_feedback.format(
                traceback_msg="Code Run cannot be executed due to function signature error! Please re-write an entirely new reward function!"
            )
            return summary

        traceback_msg = filter_traceback(stdout_str)
        assert traceback_msg is not None
        if traceback_msg == "":
            # If RL execution has no error, provide policy statistics feedback
            exec_success = True
            success_reward_key = "Episode Reward/success"
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
                    if success_reward_key != metric:
                        metric_name = f"Reward component `{metric}`"
                    else:
                        success = metric_cur_max
                        metric_name = "Success score"
                    content += f"{metric_name}: {metric_cur}, Max: {metric_cur_max:.2f}, Mean: {metric_cur_mean:.2f}, Min: {metric_cur_min:.2f} \n"
        else:
            # Otherwise, provide execution traceback error feedback
            success = DUMMY_FAILURE
            content += self.execution_error_feedback.format(traceback_msg=traceback_msg)
            self.remove()

        summary = {
            "exec_success": exec_success,
            "content": content,
            "success": success,
        }
        return summary


class SuccessNode(Node):
    def __init__(
        self,
        task=None,
        best_reward=None,
        stats=None,
        *args,
        **kwargs,
    ) -> None:
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

        self.best_reward = (
            RewardNode(**best_reward) if best_reward is not None else None
        )
        self.stats = (
            {
                "max_success_overall": DUMMY_FAILURE,
                "execute_rate": [],
                "max_success": [],
            }
            if stats is None
            else stats
        )

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

    def propose(
        self,
        num_envs=2048,
        max_iterations=2000,
        headless=False,
        video=False,
        memory_requirement=10,
    ) -> List[RewardNode]:
        self.children: List[RewardNode] = []
        responses = gpt_call(
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
                replacements={
                    REWARD_REPLACE_INPUT: REWARD_REPLACE_OUTPUT,
                    "@torch.jit.script": "",
                    "@staticmethod": "",
                },
            )
            if not no_err:
                continue
            child = RewardNode(
                num_envs=num_envs,
                max_iterations=max_iterations,
                task=self.task,
                messages=messages,
                response=response,
                code=code,
                headless=headless,
                video=video,
                memory_requirement=memory_requirement,
            )
            self.add_child(child)
            child.init()
        return self.children

    def collect(self):
        for child in self.children:
            child.summarize()
        exec_successes = [child.summary["exec_success"] for child in self.children]
        any_success = np.sum(exec_successes) > 0
        stat = {
            "execute_rate": 0.0,
            "max_success": DUMMY_FAILURE,
        }
        if not any_success:  # and cfg.sample != 1:
            logging.info(
                "All code generation failed! Repeat this iteration from the current message checkpoint!"
            )
            self._collect_stat(stat)
            return any_success, stat

        successes = [child.summary["success"] for child in self.children]
        # Select the best code sample based on the success rate
        best_sample_idx = np.argmax(np.array(successes))
        best_reward = self.children[best_sample_idx]
        for i, child in enumerate(self.children):
            if i != best_sample_idx:
                child.remove()
        best_reward.unlink()
        self.children = []
        feedback = self._wrap_user_message(
            best_reward.summary["content"] + self.code_feedback
        )
        self.messages = [
            *best_reward.messages,
            best_reward.response["message"],
            feedback,
        ]
        self.response = None

        # some statistic report
        max_success = best_reward.summary["success"]
        execute_rate = np.sum(np.array(successes) >= 0.0) / self.n_samples

        self.ite += 1
        logging.info(
            f"Iteration {self.ite}: Max Success: {max_success}, Execute Rate: {execute_rate}"
        )
        logging.info(f"Iteration {self.ite}: Best Generation ID: {best_sample_idx}")
        logging.info(
            f"Iteration {self.ite}: GPT Output Content:\n"
            + best_reward.response["message"]["content"]
            + "\n"
        )
        logging.info(
            f"Iteration {self.ite}: User Content:\n"
            + best_reward.summary["content"]
            + "\n"
        )
        if self.best_reward is not None:
            best_reward.priors = [*self.best_reward.priors, self.best_reward.idx]
        # Update the best Eureka Output
        if max_success > self.stats["max_success_overall"]:
            self.stats["max_success_overall"] = max_success
            self.best_reward = best_reward

        stat = {
            "execute_rate": execute_rate,
            "max_success": max_success,
        }
        self._collect_stat(stat)
        return any_success, stat

    def analyze_stats(self):
        stats = self.stats
        # Plot the success rate
        fig, axs = plt.subplots(2, figsize=(6, 6))
        fig.suptitle(f"{self.task.code}")

        max_successes = stats["max_success"]
        execute_rates = stats["execute_rate"]

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
        )
        return

    def _collect_stat(self, stat):
        for k, v in stat.items():
            self.stats[k].append(v)
        return


class TaskNode(Node):
    def __init__(self, variants=None, *args, **kwargs) -> None:
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
        self.variants = [] if variants is None else variants

    @property
    def num_variants(self):
        return len(self.variants)

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

    def propose(
        self, iterations=3, n_samples=3, temperature=0, model="gpt-3.5-turbo"
    ) -> List[SuccessNode]:
        responses = gpt_call(
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
                # replacements={"weight=": "weight=30.0, #", "@torch.jit.script": ""},
                replacements={"@torch.jit.script": ""},
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
                model=model,
            )
            self.add_child(child)
            child.init()
        return self.children

    def collect(self, behavior_captioner: BehaviorCaptioner = None):
        children_bak = self.children.copy()
        self.children = []
        for success_child in children_bak:
            behavior_image_paths = success_child.best_reward.play()
            succ = behavior_captioner.conclude(behavior_image_paths, task=self.code)
            if succ:
                self._collect_variant(success_child)
                self.add_child(success_child)
            else:
                success_child.unlink()
        return

    def _collect_variant(self, child):
        self.variants.append(child)
        logging.info(
            f"GPT-4v verified and collected task {self.idx} with variant success {child.idx}, best reward {child.best_reward.idx}. Current variant count: {self.num_variants}"
        )
        return


class EnvNode(Node):
    def __init__(
        self,
        idx="E00",
        skills=None,
        impossibles=None,
        resume=True,
        graph_output=None,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.idx = idx
        self.resume = resume
        self.G = nx.DiGraph()
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
        if graph_output is None:
            graph_dir = f"{self.root_dir}/envs_gpt/graphs"
            graph_output = f"{graph_dir}/{self.env_name}_{self.idx}.json"
            if not os.path.exists(graph_dir):
                os.makedirs(graph_dir)
        self.graph_output = graph_output
        self.skills = [] if skills is None else skills
        self.impossibles = [] if impossibles is None else impossibles

    @property
    def num_skills(self):
        return len(self.skills)

    @property
    def num_impossibles(self):
        return len(self.impossibles)

    def init(self):
        super().init()
        if self.num_skills > 0:
            _skill_list_str = "\n".join(
                [f"({i+1}) {skill}" for i, skill in enumerate(self.skills)]
            )
            skill_list_str = f"We have already acquired {self.num_skills} skills:\n{_skill_list_str}\n"
        else:
            skill_list_str = ""
        if self.num_impossibles > 0:
            _im_list_str = "\n".join(
                [f"({i+1}) {im}" for i, im in enumerate(self.impossibles)]
            )
            im_list_str = f"Previously we tried but failed to learn the following {self.num_impossibles} skills:\n{_im_list_str}\nMaybe consider to propose easier or clearer tasks.\n"
        else:
            im_list_str = ""
        initial_system = (
            self.initial_system.format(
                skills_count=self.num_skills,
                skills=skill_list_str,
                impossibles=im_list_str,
            )
            + self.code_output_tip
        )
        initial_user = self.initial_user.format(
            env_obs_code_string=self.env_obs_code,
        )
        self.messages = [
            self._wrap_system_message(initial_system),
            self._wrap_user_message(initial_user),
        ]
        return self

    def propose(
        self, n_samples=1, temperature=0, model="gpt-3.5-turbo"
    ) -> List[TaskNode]:
        tasks = None
        for i in range(5):
            tasks = self._propose(temperature_increase=i * 0.1)
            if len(tasks) > 0:
                break
        assert tasks is not None
        for task in tasks:
            code = task.split(": ")[-1]
            child: TaskNode = TaskNode(
                code=code, n_samples=n_samples, temperature=temperature, model=model
            )
            self.add_child(child)
            child.init()
        return self.children

    def save_graph(self):
        G = self.G
        self._add_node_to_graph(self)
        for task_node in self.children:
            self._add_node_to_graph(task_node)
            G.add_edge(self.idx, task_node.idx)
            for success_node in task_node.children:
                self._add_node_to_graph(success_node)
                G.add_edge(task_node.idx, success_node.idx)
                for reward_node in success_node.children:
                    self._add_node_to_graph(reward_node)
                    G.add_edge(success_node.idx, reward_node)
        G_dict = nx.node_link_data(G)
        with open(self.graph_output, "w") as fout:
            data_json = json.dumps(G_dict)
            fout.write(data_json + "\n")
            logging.info(f"Saved graph {self.idx} to {self.graph_output}")
        return

    def load_graph(self, graph_input=None):
        if not self.resume:
            logging.info(
                f"Run in no-resume mode. Creating/Overwriting a new graph {graph_input}."
            )
            return self
        if graph_input is None:
            graph_input = self.graph_output
        if not os.path.exists(graph_input):
            logging.info(f"No graph found in {graph_input}, creating a new one.")
            return self
        with open(graph_input, "r") as fin:
            data = json.load(fin)
        if graph_input is not None:
            for node in data["nodes"]:
                assert node["type"] == "Env"
                self._update_self_with_node(node)
                break
        G = nx.node_link_graph(data)
        self.G = G
        for task_idx in G.neighbors(self.idx):
            task_node = TaskNode(**G.nodes[task_idx])
            self.add_child(task_node)
            for success_idx in G.neighbors(task_idx):
                success_node = SuccessNode(**G.nodes[success_idx])
                task_node.add_child(success_node)
                for reward_idx in G.neighbors(success_idx):
                    reward_node = RewardNode(**G.nodes[reward_idx])
                    success_node.add_child(reward_node)
        logging.info(f"Loaded graph {graph_input}.")
        return self

    def collect(self):
        for task_child in self.children:
            self._collect_skill(task_child)
        return

    def _propose(self, temperature_increase=0) -> List[TaskNode]:
        self.init()
        messages = self.messages
        responses = gpt_call(
            messages=messages,
            model=self.model,
            n_samples=1,
            temperature=self.temperature + temperature_increase,
        )
        if self.n_samples == 1:
            logging.info(f"GPT Output:\n " + responses[0]["message"]["content"] + "\n")
        pattern = r"([Tt]ask\s+\d+:.*)"
        content = responses[0]["message"]["content"]
        tasks = re.findall(pattern, content)
        return tasks

    def _update_self_with_node(self, node):
        super()._update_self_with_node(node)
        if "skills" in node.keys():
            self.skills = node["skills"]
        if "impossibles" in node.keys():
            self.impossibles = node["impossibles"]
        return

    def _collect_skill(self, child):
        if child.num_variants > 0:
            self.skills.append(child.code)
            logging.info(
                f"Collected new skill {child.code} with {child.num_variants} variants."
            )
        else:
            self.impossibles.append(child.code)
            logging.info(f"Mission impossible on {child.code}.")
        return


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
    env_node = (
        EnvNode(
            env_name=env_name,
            resume=cfg.resume,
            model=model,
            n_samples=1,
            temperature=cfg.temperature,
            skills=[
                "Move cube A to target position",
            ],
            impossibles=[
                "Pick up the plate",
            ],
        ).init()
        # .load_graph()
    )

    bc = BehaviorCaptioner(
        init_sys_prompt=f"{env_node.prompt_dir}/task/behavior_context.txt",
    )
    # Eureka-plus generation loop
    for i in range(cfg.iteration):
        logging.info(f"Iteration {i}: Generating with {model}")
        # task_nodes = env_node.propose_fake( n_samples=cfg.n_success_samples, temperature=cfg.temperature, model=model)
        task_nodes = env_node.propose(
            n_samples=cfg.n_success_samples, temperature=cfg.temperature, model=model
        )
        for task_node in task_nodes:
            break
        success_nodes = task_node.propose(
            n_samples=cfg.n_reward_samples,
            iterations=2,
            temperature=cfg.temperature,
            model=model,
        )  # params for child init
        for i in range(2):
            for success_node in success_nodes:
                reward_nodes = success_node.propose(
                    num_envs=num_envs,
                    headless=cfg.headless,
                    video=cfg.video,
                    memory_requirement=cfg.memory_requirement,
                    max_iterations=cfg.max_iterations,
                )
                for node in reward_nodes:
                    node.run()
            for success_node in success_nodes:
                success_node.collect()
        task_node.collect(behavior_captioner=bc)  # check behavior caption
    env_node.collect()
    env_node.save_graph()
    logging("All done!")


if __name__ == "__main__":
    main()
