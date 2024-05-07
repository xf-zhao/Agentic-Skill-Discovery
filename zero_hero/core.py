import numpy as np
import pandas as pd
import wandb
import json
import networkx as nx
import psutil
import logging
import matplotlib.pyplot as plt
import os
import openai
import re
import subprocess
import shutil
import time
import uuid
import tempfile
from typing import List
from evolution.utils.misc import *
from evolution.utils.file_utils import load_tensorboard_logs
from evolution.utils.extract_task_code import *
from .behavior import BehaviorCaptioner, video_to_frames


DUMMY_FAILURE = -10000.0
ORBIT_ROOT_DIR = os.environ["ORBIT_ROOT_DIR"]
ZEROHERO_ROOT_DIR = os.environ["ZEROHERO_ROOT_DIR"]
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

SUCCESS_MUST_CONTAIN = """
@configclass
class SuccessCfg:
"""

REWARD_REPLACE_OUTPUT = """
@configclass
class RewardsCfg:

    try:
        success = SuccessCfg().success
    except:
        success = list(SuccessCfg().__dict__.values())[0]
    terminate_1 = get_terminate_penalty(TerminationsCfg().cube_a_dropping)
    terminate_2 = get_terminate_penalty(TerminationsCfg().cube_b_dropping)
    terminate_3 = get_terminate_penalty(TerminationsCfg().plate_dropping)

"""


def extract_tasks(content, pattern=r"([Tt]ask(\s+\d+)?:.*)"):
    tasks = re.findall(pattern, content)
    codes = None
    if len(tasks) > 0:
        codes = [
            (
                task.split(": ")[-1]
                .replace("specific", "target")
                .replace("specified", "target")
                .replace("target target", "target")
                .replace("**", "").strip()
            )
            for task, __ in tasks
        ]
    return codes


def _wrap_message(content, role="user"):
    return {"role": role, "content": content}


def wrap_system_message(content):
    return _wrap_message(content=content, role="system")


def wrap_user_message(content):
    return _wrap_message(content=content, role="user")


def wrap_assistant_message(content):
    return _wrap_message(content=content, role="assistant")


def gpt_call(
    messages,
    model="gpt-3.5-turbo",
    n_samples=1,
    temperature=0,
):
    choices = []
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
                    return []
                time.sleep(1)
        if response_cur is None:
            logging.info("Code terminated due to too many failed attempts!")
            exit()
        choices.extend(response_cur["choices"])
        prompt_tokens = response_cur["usage"]["prompt_tokens"]
        total_completion_token += response_cur["usage"]["completion_tokens"]
        total_token += response_cur["usage"]["total_tokens"]

    # Logging Token Information
    logging.info(
        f">>> Prompt Tokens: {prompt_tokens}, Completion Tokens: {total_completion_token}, Total Tokens: {total_token}"
    )
    return choices


def extract_code_string(responses, combine_all=False, log=False):
    patterns = [
        r"```python(.*?)```",
        r"```(.*?)```",
        r'"""(.*?)"""',
        r'""(.*?)""',
        r'"(.*?)"',
    ]
    content = ""
    for resp in responses:
        content += resp["content"]
    # Regex patterns to extract python code enclosed in GPT response
    for pattern in patterns:
        code_string = ""
        code_strings = re.findall(pattern, content, re.DOTALL)
        if len(code_strings) > 0:
            if combine_all:
                for cs in code_strings:
                    code_string += cs.strip() + "\n"
            else:
                # assume the last is a combined one in this case
                code_string = code_strings[-1].strip()
            break
        else:
            code_string = None
    if log:
        print("=" * 100)
        print(code_string)
        print("=" * 100)
    return code_string


class Database:
    def get_attr(self, attr, index=None, task=None):
        if index is not None:
            variants = self.df.loc[index][attr]
        elif task is not None:
            variants = self.df[self.df["command"] == task][attr]
        else:
            raise NotImplementedError
        return variants.values


class TaskDatabase(Database):
    def __init__(
        self,
        env_name,
        env_idx,
        store_path=None,
        target_num_skills=64,
        failed_tolerance=None,
        proposal_batch=10,
    ) -> None:
        self.store_path = (
            f'{ZEROHERO_ROOT_DIR}/envs_gpt/tasks/{env_name.replace(" ","_")}_{env_idx}.csv'
            if store_path is None
            else store_path
        )
        self.target_num_skills = target_num_skills
        self.failed_tolerance = (
            failed_tolerance if failed_tolerance is not None else target_num_skills * 2
        )
        self.proposal_batch = proposal_batch
        self.columns = ["command", "status", "variants"]
        self.load()

    def met_target(self):
        is_met = (
            self.num_skills >= self.target_num_skills
            or self.num_failed >= self.failed_tolerance
        )
        return is_met

    def should_wait(self):
        return self.num_wait >= self.proposal_batch

    def reset_tasks(self, tasks):
        df = pd.DataFrame(columns=self.columns)
        self.df = df
        self.add_tasks(tasks)
        return self

    def load(self):
        store_path = self.store_path
        if os.path.exists(store_path):
            df = pd.read_csv(store_path)
        else:
            os.makedirs(os.path.dirname(store_path), exist_ok=True)
            df = pd.DataFrame(columns=self.columns)
        self.df = df
        return self

    @property
    def commands(self):
        return self.df["commands"]

    @property
    def status(self):
        return self.df["status"]

    @property
    def variants(self):
        return self.df["variants"]

    @property
    def num_wait(self):
        return (self.df["status"] == "todo").sum() + (
            self.df["status"] == "doing"
        ).sum()

    @property
    def num_skills(self):
        return (self.df["status"] == "completed").sum()

    @property
    def num_failed(self):
        return (self.df["status"] == "failed").sum()

    def add_task(self, task: str):
        df = self.df
        if task in self.df["command"].values:
            self.refresh_task(task)
        else:
            row = pd.Series({"command": task, "status": "todo", "variants": ""})
            df = pd.concat([df, pd.DataFrame([row], columns=row.index)]).reset_index(
                drop=True
            )
            self.df = df
        return

    def add_tasks(self, tasks):
        for task in tasks:
            self.add_task(task)
        return

    def drop_tasks(self, task_indicates, reduce=1):
        df = self.df
        indices_to_drop = []
        for task_indicate in task_indicates:
            index_to_drop = re.search(
                pattern=r"Task\s+(\d+)", string=task_indicate
            ).group(1)
            index = int(index_to_drop) - reduce
            indices_to_drop.append(index)
        df = df.drop(index=indices_to_drop).reset_index(drop=True)
        self.df = df
        return

    def update_task(self, task: dict):
        command = task["command"]
        if command not in self.df["command"].values:
            self.add_task(command)
        df = self.df
        df.loc[df.command == command, "status"] = task["status"]
        df.loc[df.command == command, "variants"] = task["variants"]
        self.df = df
        self.save()
        return

    def refresh_task(self, task: str):
        self.update_task({"command": task, "status": "todo", "variants": ""})
        return

    def save(self):
        self.df.to_csv(self.store_path, index=False)
        print(f"self.df:\n {self.df}")
        print(f"Saved data to {self.store_path}")

    def render(self, pure=False):
        df = self.df
        if not pure:
            numbered_tasks = "\n".join(
                [
                    f"({i+1}) Task: {row.command} Status: {row.status}. Variants: {row.variants}"
                    for i, row in df.iterrows()
                ]
            )
        else:
            numbered_tasks = "\n".join(
                [
                    f"({i+1}) Task: {row.command} Status: {row.status}."
                    for i, row in df.iterrows()
                ]
            )
        numbered_tasks += "\n"
        print(numbered_tasks)
        return numbered_tasks

    def pop_wait(self, timeout=60 * 60 * 10):
        itime = 0
        while task is None and itime < timeout:
            itime += 1
            time.sleep(1)
            task = self.pop()
            if task is not None:
                break
        return task

    def pop(self):
        df = self.df
        indices = df.loc[df.status == "todo"].index
        if len(indices) > 0:
            index_to_pop = indices[0]
            df.loc[index_to_pop, "status"] = "doing"
            task = df.loc[index_to_pop, "command"]
            self.df = df
            self.save()
        else:
            task = None
        return task

    @property
    def skills(self):
        df = self.df
        skill_df = df[df["status"] == "completed"]
        return skill_df


class CenteralizedTask:
    def __init__(
        self,
        env_name,
        store_path=None,
        model="gpt-3.5-turbo-0125",
        temperature=0.7,
    ) -> None:
        self.env_name = env_name
        self.root_dir = ZEROHERO_ROOT_DIR
        self.prompt_dir = f"{self.root_dir}/evolution/utils/prompts"
        self.center_tasks = TaskDatabase(
            env_name=env_name, env_idx="centralized_task", store_path=store_path
        ).load()
        self.messages = []
        self.env_obs_file = (
            f"{self.root_dir}/envs/{self.env_name}/env_cfg/observation.py"
        )
        self.initial_system = file_to_string(
            f"{self.prompt_dir}/design/initial_system.txt"
        )
        self.initial_user = file_to_string(f"{self.prompt_dir}/design/initial_user.txt")
        self.followup_user = file_to_string(
            f"{self.prompt_dir}/design/followup_user.txt"
        )
        self.model = model
        self.temperature = temperature

    def consume(self, task_database: TaskDatabase):
        tdb = self.center_tasks
        df = pd.concat([tdb.df, task_database.df]).reset_index(drop=True)
        tdb.df = df
        logging.info(
            f"Updated centralized task database {tdb.store_path} with {len(task_database.df)} new tasks."
        )
        tdb.save()
        return self

    def filter(self, task_database: TaskDatabase):
        tdb = self.center_tasks
        tasks_rm = self._filter(task_database)
        if tasks_rm is not None:
            task_database.drop_tasks(tasks_rm)
            task_database.render()
            task_database.save()
            logging.info(
                f"Rm {len(tasks_rm)} tasks for task database {task_database.store_path}."
            )
        else:
            logging.info(f"No duplicates for database {tdb.store_path}.")
        self.consume(task_database)
        return task_database

    def _filter(self, task_database):
        tdb = self.center_tasks
        env_obs_code_string = file_to_string(self.env_obs_file)
        initial_system = self.initial_system.format(
            known_tasks=tdb.render(), source_code=env_obs_code_string
        )
        initial_user = self.initial_user.format(
            coming_tasks=task_database.render(pure=True)
        )
        followup_user = self.followup_user
        messages = [
            wrap_system_message(initial_system),
            wrap_user_message(initial_user),
        ]
        init_resp = self._gpt_call(messages)
        print(init_resp["content"])
        messages.extend([init_resp, wrap_user_message(followup_user)])
        resp = self._gpt_call(messages)
        print(resp["content"])
        filtered_tasks = extract_tasks(resp["content"], pattern=r"(Task(\s+\d+)).*")
        self.messages = messages
        return filtered_tasks

    def _gpt_call(self, messages):
        choices = gpt_call(
            messages=messages,
            model=self.model,
            n_samples=1,
            temperature=self.temperature,
        )
        resp = choices[0]["message"]
        return resp


class Node:
    def __init__(
        self,
        root_dir=None,
        idx=None,
        type=None,
        messages=None,
        response=None,
        code=None,
        iterations=1,
        env_name="franka_table",
        model="gpt-3.5-turbo",
        n_samples=1,
        temperature=0,
        ite=0,
        resume=True,
        precedents=None,
        local_num_syntax_error=0,
        conversation_dir=None,
        *args,
        **kwargs,
    ) -> None:
        self.resume = resume
        self.root_dir = root_dir if root_dir is not None else ZEROHERO_ROOT_DIR
        self.prompt_dir = f"{self.root_dir}/evolution/utils/prompts"
        self.env_name = env_name
        self.precedents = precedents
        self.type = type
        self.parent = None
        self.children = []
        self.iterations = iterations
        self.runable = False
        self.model = model
        self.n_samples = n_samples
        self.temperature = temperature
        self.idx = idx
        self.messages = messages
        self.response = [] if response is None else response
        self.code = code
        self.summary = None
        self.ite = ite
        self.local_num_syntax_error = local_num_syntax_error
        self.conversation_dir = (
            f"{self.root_dir}/envs_gpt/converstions"
            if conversation_dir is None
            else conversation_dir
        )
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
        self.num_syntax_error = 0

    def init(self):
        if self.idx is None:
            self.idx = f"{self.type[0]}{uuid.uuid4().hex[:8]}"
        if self.precedents is not None and len(self.precedents)>0:
            self.precedent_skills = '\n'.join([f'({i}) {skill}' for i, skill in enumerate(self.precedents.values())])
        else:
            self.precedent_skills = ''
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
            if node.num_candidates > 0:
                data.update(
                    {"candidates": [candidate.idx for candidate in node.candidates]}
                )
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

    def say(self):
        words = self.words()
        print(words)
        return self

    def words(self):
        word = ""
        for msg in [*self.messages, *self.response]:
            role = msg["role"]
            word = word + "=" * 50 + f"<<< {role} >>>" + "=" * 50 + "\n"
            word = word + msg["content"] + "\n"
        return word

    def _extract_env_obs(self):
        env_obs_code_string = file_to_string(self.env_obs_file)
        pattern = r"class MyObservationCfg.*:(.*?)def __post_init__.*"
        code_string = re.search(pattern, env_obs_code_string, re.DOTALL)
        if code_string is not None:
            code_string = code_string.group(0).strip()
        return code_string

    def _loop_until_no_syntax_err(self, messages, response=None, replacements={}):
        if response is None:
            response = []
            syntax_valid = False
        else:
            response = [response]
            syntax_valid = True
        local_num_syntax_error = 0
        messages = messages.copy()
        for t in range(10):
            gpt_err = False
            for _ in range(5):
                if not syntax_valid:
                    choices = gpt_call(
                        messages=[*messages, *response],
                        model=self.model,
                        n_samples=1,
                        temperature=self.temperature + t * 0.1,
                    )
                    if choices is None or len(choices) == 0:
                        gpt_err = True
                        break
                    else:
                        gpt_err = False
                        response.append(choices[0]["message"])
                code = extract_code_string(responses=response, combine_all=True)
                syntax_valid, err_feedback = self._syntax_examine(code)
                if not syntax_valid:
                    response.append(self._wrap_user_message(err_feedback))
                    self.num_syntax_error += 1
                    local_num_syntax_error += 1
                else:
                    for k, v in replacements.items():
                        code = code.replace(k, v)
                    break
            if not gpt_err:
                break
        return messages, response, code, syntax_valid, local_num_syntax_error

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
        prefix_codes="import torch\nfrom typing import Dict, List, Tuple\n",
        remove_temp=False,
    ):
        if code is None:
            return False, self.code_format_feedback
        if self.type == "Success":
            must_contain = REWARD_REPLACE_INPUT
        elif self.type == "Task":
            must_contain = SUCCESS_MUST_CONTAIN
        else:
            must_contain = ""
        if must_contain.strip("\n").strip() not in code:
            return (
                False,
                f"Please always configure with exact `{must_contain}`! No other new names.",
            )
        err_feedback = None
        for k, v in replacements.items():
            code = code.replace(k, v)
        try:
            with tempfile.NamedTemporaryFile(delete=False) as temp:
                code = prefix_codes + code
                temp.write(code.encode("utf-8"))
                logging.info(f"Examining syntax with temp file: {temp.name}...")
            subprocess.check_output(["python3", temp.name], stderr=subprocess.STDOUT)
            if remove_temp:
                os.remove(temp.name)
            valid = True
            logging.info(f"No syntax error, continue to build graph.")
        except subprocess.CalledProcessError as e:
            exec_msg = e.output.decode()
            traceback_msg = filter_traceback(exec_msg)
            assert traceback_msg != ""
            logging.warning(f"Temp syntax error found: {traceback_msg}, loop to fix.")
            err_feedback = self.execution_error_feedback.format(
                traceback_msg=traceback_msg
            )
            valid = False
        return valid, err_feedback

    def _wrap_system_message(self, content):
        return self._wrap_message(content=content, role="system")

    def _wrap_user_message(self, content):
        return self._wrap_message(content=content, role="user")

    def _wrap_assistant_message(self, content):
        return self._wrap_message(content=content, role="assistant")

    def _wrap_message(self, content, role="user"):
        return {"role": role, "content": content}

    def _write_record_line(self, data, record_file):
        if os.path.exists(record_file):
            df = pd.read_csv(record_file)
            df_udpate = pd.concat(
                [df, pd.DataFrame([data], columns=df.columns)]
            ).reset_index(drop=True)
        else:
            df_udpate = pd.DataFrame([data])
        df_udpate.to_csv(record_file, index=False, header=True)
        return

    def _save_conversation(self):
        conversation_path = self.conversation_dir + f"/{self.idx}.txt"
        if not os.path.exists(self.conversation_dir):
            os.makedirs(self.conversation_dir, exist_ok=True)
        with open(conversation_path, "w") as f:
            for msg in self.messages:
                role, content = msg["role"], msg["content"]
                line = f"<<< {role} START >>>\n{content} \n<<< {role} END >>>"
                f.write(line)
        return


class RewardNode(Node):
    def __init__(
        self,
        num_envs=11,
        task=None,
        headless=False,
        video=False,
        memory_requirement=16,
        min_gpu=90,
        max_iterations=2000,
        priors=None,
        record=None,
        best_record=None,
        task_ite=1,
        reward_ite=1,
        behavior_captioner: BehaviorCaptioner = None,
        finetune =False,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.type = "Reward"
        self.behavior_captioner = behavior_captioner
        self.task = task
        self.headless = headless
        self.task_ite = task_ite
        self.reward_ite = reward_ite
        self.finetune = finetune
        self.video = video
        self.runable = True
        self.success_idx = None
        self.num_envs = num_envs
        self.rl_run = None
        self.play_run = None
        self.rl_filepath = None
        self.play_filepath = None
        self.memory_requirement = memory_requirement
        self.min_gpu = min_gpu
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
        self.playbacks = None
        if record is None:
            record_dir = f"{self.root_dir}/envs_gpt/records"
            self.record = f"{record_dir}/{self.env_name}_record.csv"
            if not os.path.exists(record_dir):
                os.makedirs(record_dir)
        if best_record is None:
            best_record_dir = f"{self.root_dir}/envs_gpt/records"
            self.best_record = f"{best_record_dir}/{self.env_name}_best_record.csv"
            if not os.path.exists(best_record_dir):
                os.makedirs(best_record_dir)
        # For gpt-4-v
        self.caption_data = None
        self.s_exec_success = True
        self.r_exec_success = True

    def init(self):
        self.caption_data = None
        super().init()
        cur_env_dir = f"{self.root_dir}/envs_gpt/{self.env_name}/{self.idx}"
        self.rl_filepath = f"{self.idx}.txt"
        if self.parent is not None:
            self.success_idx = self.parent.idx
        self.cur_env_dir = cur_env_dir
        self.log_dir = f"{self.cur_env_dir}/logs"

        reward_file = f"{cur_env_dir}/reward.py"
        if os.path.exists(cur_env_dir):
            if not self.resume:
                shutil.rmtree(cur_env_dir)
                logging.info(f"Remove directory {cur_env_dir}.")
            with open(reward_file, "r") as file:
                self.code = file.read()
        else:
            os.makedirs(cur_env_dir, exist_ok=True)

            shutil.copy(self.env_file, cur_env_dir)
            shutil.copy(self.env_obs_file, cur_env_dir)
            shutil.copy(self.termination_file, cur_env_dir)
            with open(f"{cur_env_dir}/__init__.py", "w") as f:
                f.write(MODULE_INIT.replace("UUID_HEX", self.idx))

            success_file = f"{cur_env_dir}/success.py"
            with open(success_file, "w") as file:
                file.write(SUCCESS_INIT)
                file.writelines(self.parent.code + "\n")

            with open(reward_file, "w") as file:
                file.write(REWARD_INIT)
                file.writelines(self.code + "\n")
        if self.precedents is not None and len(self.precedents)>0 and self.finetune:
            precedent_idx = list(self.precedents.keys())[0]
            precedent_logpath = f'{self.root_dir}/envs_gpt/{self.env_name}/{precedent_idx}/logs'
            pts = [thing.lstrip('model_').rstrip('.pt') for thing in os.listdir(precedent_logpath) if thing.startswith('model_') and thing.endswith('.pt')]
            max_epoch = max([int(pt) for pt in pts])
            pt_finetune = f'model_{max_epoch}.pt'
            os.makedirs(f'{cur_env_dir}/logs', exist_ok=True)
            shutil.copy(f'{precedent_logpath}/{pt_finetune}', f'{cur_env_dir}/logs/')
            shutil.copytree(f'{precedent_logpath}/params', f'{cur_env_dir}/logs/params')

        return self

    def remove(self):
        self.unlink()
        cur_env_dir = self.cur_env_dir
        if os.path.exists(cur_env_dir):
            shutil.rmtree(cur_env_dir)
            logging.info(f"Removed node {self.idx} and its directory {cur_env_dir}.")
        return

    def _prepare_launch(self, mode="RTX"):
        # Only run when memory is enough
        # Maximum 24 hour waiting time, long enough for finishing one run
        max_waiting = 60 * 12  # mins, so here 12h
        for i in range(max_waiting):
            available_mem = psutil.virtual_memory().available / 1024 / 1024 / 1024
            is_enough = (
                available_mem > self.memory_requirement
            )  # 16 GB is the minimum mem for a new instance
            # Find the freest GPU to run GPU-accelerated RL
            gpu_avi = set_freest_gpu(mode=mode)
            is_valid = gpu_avi >= self.min_gpu
            if is_enough and is_valid:
                break
            else:
                if i % 10 == 0:
                    logging.info(
                        f"Available RAM: {available_mem} (require {self.memory_requirement}); Available GPU: {gpu_avi} (require {self.min_gpu}). Waiting for enough resouces to run node {self.idx}. Time elapsed: {i} minutes."
                    )
                time.sleep(60)
        assert i < max_waiting - 1
        return

    def run(self):
        self._prepare_launch(mode="GTX")

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
            self.log_dir,
        ]
        if self.video:
            rl_run_command.append("--video")
            if self.headless:
                rl_run_command.append("--offscreen_render")
        rl_run_command = self._fill_command(rl_run_command)
        with open(self.rl_filepath, "w") as f:
            self.rl_run = subprocess.Popen(
                rl_run_command,
                stdout=f,
                stderr=f,
            )
        self._block_until_training()
        return self

    def play(self, suffix="_videos"):
        self._prepare_launch(mode="RTX")

        # Execute the python file with flags
        play_run_command = [
            f"{ORBIT_ROOT_DIR}/orbit.sh",
            "-p",
            f"{self.root_dir}/rsl_rl/play.py",
            "--task",
            f"{self.idx}",
            "--suffix",
            suffix,
            "--num_envs",
            "1",
            "--video",
            "--log_root",
            os.path.dirname(self.log_dir),
        ]
        if self.headless:
            play_run_command.append("--offscreen_render")
        play_run_command = self._fill_command(play_run_command)
        self.play_filepath = self.rl_filepath.rstrip(".txt") + "_play.txt"
        with open(self.play_filepath, "w") as f:
            self.play_run = subprocess.Popen(play_run_command, stdout=f, stderr=f)
        playbacks = self._block_until_play_recorded()
        self.playbacks = playbacks
        return playbacks

    def caption(self):
        assert self.behavior_captioner is not None
        for i in range(3):
            if self.playbacks is None:
                self.playbacks = self.play()
            if i > 0:
                logging.warning(f"Tried but failed to record playbacks {i}-th times.")
        description, v_succ = self.behavior_captioner.conclude(
            playbacks=self.playbacks, task=self.task
        )
        if description is None:
            return
        data = {
            **self.playbacks,
            "gpt-4v-succ": v_succ,
        }
        data_extra = data.copy()
        data_extra["gpt-4v-description"] = description
        # only caption for best node of succ node
        self._write_record_line(data, self.best_record)
        self.caption_data = data_extra
        return data_extra

    def summarize(self):
        summary = self._summarize_runlog()
        self.summary = summary
        return self

    def _fill_command(self, run_command):
        if self.headless:
            run_command.append("--headless")
        if self.precedents is not None and len(self.precedents) > 0:
            run_command.append("--precedents")
            for precedent in self.precedents:
                if not precedent.startswith("/"):
                    precedent = (
                        f"{ZEROHERO_ROOT_DIR}/envs_gpt/{self.env_name}/{precedent}"
                    )
                run_command.append(precedent)
        print(f"Executing commands: {run_command}")
        return run_command

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
                    f"[Task iter {self.task_ite} - Reward iter {self.reward_ite} - Node {self.idx}]: Successfully launched RL training."
                )
                self.s_exec_success = True
                self.r_exec_success = True
            else:
                logging.error(
                    f"[Task iter {self.task_ite} - Reward iter {self.reward_ite} - Node {self.idx}]: Execution error!"
                )
                if "/success.py" in msg:
                    self.s_exec_success = False
                    self.r_exec_success = True
                else:
                    self.s_exec_success = True
                    self.r_exec_success = False
                self.exec_success = False
                self.success = DUMMY_FAILURE
            logging.info(f"Log at {self.rl_filepath}")
            break
        return

    @property
    def record_data(self):
        data = {
            "task": self.task,
            "task_ite": self.task_ite,
            "reward_ite": self.reward_ite,
            "success_idx": self.success_idx,
            "reward_idx": self.idx,
            "reward_num_syntax_error": self.local_num_syntax_error,
            "exec_success": self.exec_success,
            "s_exec_success": self.s_exec_success,
            "r_exec_success": self.r_exec_success,
            "success": self.success,
        }
        return data

    def _block_until_play_recorded(self):
        self.play_run.communicate()
        pattern = r".*(Loading model checkpoint from.*)"
        play_log = file_to_string(self.play_filepath)
        model_path_reg = re.search(pattern=pattern, string=play_log)
        if model_path_reg is None:
            return
        video_dir = (
            model_path_reg.group(1).split(":")[1].strip().replace(".pt", "_videos")
        )
        video_path = f"{video_dir}/rl-video-step-0.mp4"
        obs_path = f"{video_dir}/rl-video-step-0-obs.json"
        if not os.path.exists(video_path):
            return
        image_paths = video_to_frames(video_path)
        playbacks = {
            "reward_idx": self.idx,
            "video_dir": video_dir,
            "image_paths": image_paths,
            "video_path": video_path,
            "state_path": obs_path,
        }
        return playbacks

    def _summarize_runlog(self):
        self.rl_run.communicate()
        exec_success = False
        content = ""
        success = DUMMY_FAILURE
        summary = {
            "s_exec_success": self.s_exec_success,
            "r_exec_success": self.r_exec_success,
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
            # but may no logs at all
            try:
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
                    if (
                        metric.startswith("Episode Reward/")
                        and "/terminate_" not in metric
                    ):
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
            except Exception as e:
                logging.error(f"Failed to analyze tensorboard logs!")
                # Otherwise, provide execution traceback error feedback
                success = DUMMY_FAILURE
                content += self.execution_error_feedback.format(
                    traceback_msg=traceback_msg
                )
                self.remove()
        else:
            # Otherwise, provide execution traceback error feedback
            success = DUMMY_FAILURE
            content += self.execution_error_feedback.format(traceback_msg=traceback_msg)
            self.remove()

        summary = {
            "s_exec_success": self.s_exec_success,
            "r_exec_success": self.r_exec_success,
            "exec_success": exec_success,
            "content": content,
            "success": success,
        }
        self.success = success
        self.exec_success = exec_success
        self._write_record_line(self.record_data, self.record)
        self._save_conversation()
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
        self.gpt4v_tip = file_to_string(f"{self.prompt_dir}/reward/gpt4v_tip.txt")
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
                "syntax_error": [],
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
            task_description=self.task,
            precedent_skills = self.precedent_skills,
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
        memory_requirement=16,
        min_gpu=90,
        task_ite=1,
        reward_ite=1,
        behavior_captioner=None,
        finetune=False,
    ) -> List[RewardNode]:
        self.children: List[RewardNode] = []
        choices = gpt_call(
            messages=self.messages,
            model=self.model,
            n_samples=self.n_samples,
            temperature=self.temperature,
        )
        if choices is None:
            choices = gpt_call(
                messages=self.messages,
                model=self.model,
                n_samples=1,
                temperature=self.temperature + 0.5,
            )

        if len(choices) == 1:
            logging.info(f"GPT Output:\n " + choices[0]["message"]["content"] + "\n")

        for choice in choices:
            response = choice["message"]
            messages, response, code, syntax_valid, local_num_syntax_error = (
                self._loop_until_no_syntax_err(
                    messages=self.messages,
                    response=response,
                    replacements={
                        REWARD_REPLACE_INPUT: REWARD_REPLACE_OUTPUT,
                        "@torch.jit.script": "",
                        "@staticmethod": "",
                    },
                )
            )
            if not syntax_valid:
                continue
            child = RewardNode(
                root_dir=self.root_dir,
                num_envs=num_envs,
                max_iterations=max_iterations,
                task=self.task,
                messages=messages,
                response=response,
                code=code,
                headless=headless,
                video=video,
                memory_requirement=memory_requirement,
                min_gpu=min_gpu,
                task_ite=task_ite,
                reward_ite=reward_ite,
                behavior_captioner=behavior_captioner,
                precedents=self.precedents,
                finetune=finetune,
                local_num_syntax_error=local_num_syntax_error,
            )
            self.add_child(child)
            child.init()
        return self.children

    def collect(self):
        for child in self.children:
            child.summarize()
        # only succ_func() will terminate iteration to save unnecessary gpt call cost.
        s_exec_successes = [child.summary["s_exec_success"] for child in self.children]
        any_success = np.sum(s_exec_successes) > 0
        stat = {
            "syntax_error": 0.0,
            "execute_rate": 0.0,
            "max_success": DUMMY_FAILURE,
        }
        if not any_success:  # and cfg.sample != 1:
            logging.info(
                "All code generation failed! Repeat this iteration from the current message checkpoint!"
            )
            self._collect_stat(stat)
            self.children = []
            return any_success, stat

        # reward_func exec error will be iterated to evolve
        successes = [child.summary["success"] for child in self.children]
        syntax_errors = [child.local_num_syntax_error for child in self.children]
        # Select the best code sample based on the success rate
        best_sample_idx = np.argmax(np.array(successes))
        best_reward = self.children[best_sample_idx]
        for i, child in enumerate(self.children):
            if i != best_sample_idx:
                child.remove()
        best_reward.unlink()
        self.children = []
        gpt4v_feedback = ""
        if best_reward.exec_success:
            best_reward.caption()
            if best_reward.caption_data is not None:
                gpt4v_description = best_reward.caption_data["gpt-4v-description"]
                gpt4v_feedback = self.gpt4v_tip.format(
                    gpt4v_description=gpt4v_description
                )

        feedback = self._wrap_user_message(
            best_reward.summary["content"] + gpt4v_feedback + self.code_feedback
        )
        self.messages = [
            *best_reward.messages,
            *best_reward.response,
            feedback,
        ]
        self.response = []

        # some statistic report
        max_success = best_reward.summary["success"]
        execute_rate = np.array(successes).mean()
        avg_syntax_error = np.array(syntax_errors).mean()

        self.ite += 1
        logging.info(
            f"Iteration {self.ite}: Max Success: {max_success}, Execute Rate: {execute_rate}, Avg Syntax Error: {avg_syntax_error}"
        )
        logging.info(f"Iteration {self.ite}: Best Generation ID: {best_reward.idx}")
        logging.info(f"Iteration {self.ite}: Conversation:\n" + best_reward.words())
        logging.info(
            f"Iteration {self.ite}: User Content:\n"
            + best_reward.summary["content"]
            + "\n"
        )
        if self.best_reward is not None:
            best_reward.priors = [*self.best_reward.priors, self.best_reward.idx]
        if max_success > self.stats["max_success_overall"]:
            self.stats["max_success_overall"] = max_success
        # Update the best but not like Eureka
        self.best_reward = best_reward

        stat = {
            "syntax_error": avg_syntax_error,
            "execute_rate": execute_rate,
            "max_success": max_success,
        }
        self._collect_stat(stat)
        return any_success, stat

    def analyze_stats(self):
        stats = self.stats
        # Plot the success rate
        fig, axs = plt.subplots(2, figsize=(6, 6))
        fig.suptitle(f"{self.task}")

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
    def __init__(
        self,
        variants=None,
        status_output=None,
        candidates=None,
        *args,
        **kwargs,
    ) -> None:
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
        self.candidates = [] if candidates is None else candidates
        if status_output is None:
            status_dir = f"{self.root_dir}/envs_gpt/status"
            status_output = f"{status_dir}/{self.env_name}_{self.idx}.json"
            if not os.path.exists(status_dir):
                os.makedirs(status_dir)

    @property
    def num_variants(self):
        return len(self.variants)

    @property
    def num_candidates(self):
        return len(self.candidates)

    def init(self):
        super().init()
        initial_system = (
            self.initial_system.format(signature_string=self.signature)
            + self.code_output_tip
        )
        initial_user = self.initial_user.format(
            task_obs_code_string=self.env_obs_code,
            task_description=self.code,
            precedent_skills=self.precedent_skills,
        )
        self.messages = [
            self._wrap_system_message(initial_system),
            self._wrap_user_message(initial_user),
        ]

        return self

    def provide(self, *args, **kwargs):
        if len(self.children) == 0:
            return self.propose(*args, **kwargs)
        return self.children

    def propose(
        self, iterations=3, n_samples=3, temperature=0, model="gpt-3.5-turbo"
    ) -> List[SuccessNode]:
        choices = gpt_call(
            messages=self.messages,
            model=self.model,
            n_samples=self.n_samples,
            temperature=self.temperature,
        )
        if choices is None:
            choices = gpt_call(
                messages=self.messages,
                model=self.model,
                n_samples=1,
                temperature=self.temperature + 0.5,
            )
        if len(choices) == 1:
            logging.info(f"GPT Output:\n " + choices[0]["message"]["content"] + "\n")

        for choice in choices:
            response = choice["message"]
            messages, response, code, syntax_valid, local_num_syntax_error = (
                self._loop_until_no_syntax_err(
                    messages=self.messages,
                    response=response,
                    # replacements={"weight=": "weight=30.0, #", "@torch.jit.script": ""},
                    replacements={"@torch.jit.script": ""},
                )
            )
            if not syntax_valid:
                continue
            child: SuccessNode = SuccessNode(
                root_dir=self.root_dir,
                task=self.code,
                messages=messages,
                response=response,
                code=code,
                iterations=iterations,
                n_samples=n_samples,
                temperature=temperature,
                model=model,
                precedents=self.precedents,
                local_num_syntax_error=local_num_syntax_error,
            )
            self.add_child(child)
            child.init()
        return self.children

    def collect(self):
        children_bak = self.children.copy()
        self.children = []
        num_optimized = []
        num_v_succ = []
        num_f_succ = []
        variant_videos, candidate_videos = [], []
        variant_video_captions, candidate_video_captions = [], []
        for success_child in children_bak:
            if success_child.best_reward is not None:
                # control whether to re-use good success functions
                self.add_child(success_child)
                f_succ = int(success_child.best_reward.summary["success"] > 0)
                num_f_succ.append(f_succ)
                f_optimized = int(success_child.best_reward.summary["success"] >= 0)
                num_optimized.append(f_optimized)
                caption_data = success_child.best_reward.caption_data
                if caption_data is not None:
                    v_succ = caption_data["gpt-4v-succ"]
                    video_path = caption_data["video_path"]
                    video_caption = caption_data["gpt-4v-description"]
                else:
                    v_succ, video_path, video_caption = False, None, None
                if v_succ:
                    num_v_succ.append(1)
                else:
                    num_v_succ.append(0)
                if f_succ:
                    if v_succ:
                        self._collect_variant(success_child)
                        if video_path is not None:
                            variant_videos.append(video_path)
                            variant_video_captions.append(video_caption)
                    else:
                        self._collect_candidate(success_child)
                        if video_path is not None:
                            candidate_videos.append(video_path)
                            candidate_video_captions.append(video_caption)
            else:
                num_f_succ.append(0)
                num_v_succ.append(0)
                num_optimized.append(0)
                success_child.unlink()

        def wrap_variant_video(variant_videos, captions=None, prefix="variants"):
            video_stats = {}
            for v_path, video_caption in zip(variant_videos, captions):
                v_idx = v_path.split("/")[-4]
                wandb_video = {
                    f"{prefix}_video_{v_idx}": wandb.Video(
                        v_path, caption=video_caption, fps=30, format="mp4"
                    )
                }
                video_stats.update(wandb_video)
            return video_stats

        stat = {
            "GPT-4v succ": np.array(num_v_succ).mean(),
            "Func succ": np.array(num_f_succ).mean(),
            "Succ consistant": np.array(
                [a == b for a, b in zip(num_f_succ, num_v_succ)]
            ).mean(),
            "num_optimized": np.array(num_optimized).mean(),
            **wrap_variant_video(
                variant_videos, variant_video_captions, prefix="variants"
            ),
            **wrap_variant_video(
                candidate_videos, candidate_video_captions, prefix="candidates"
            ),
        }

        return stat

    def _collect_variant(self, child):
        self.variants.append(child)
        logging.info(
            f"GPT-4v verified and collected task {self.idx} with variant success {child.idx}, best reward {child.best_reward.idx}. Current variant count: {self.num_variants}"
        )
        return

    def _collect_candidate(self, child):
        self.candidates.append(child)
        logging.info(
            f"GPT-4v verified but does not admit task {self.idx} with variant success {child.idx}, best reward {child.best_reward.idx}. Current candidates count: {self.num_candidates}"
        )
        return

    def save_status(self):
        def get_variant_tree(variant):
            tree = {
                "best_reward": {
                    variant.best_reward.idx: {
                        "priors": variant.best_reward.priors,
                        "summary": variant.best_reward.summary,
                    }
                },
                "stats": variant.stats,
            }
            return tree

        def get_variant_forest(variants):
            forest = {variant.idx: get_variant_tree(variant) for variant in variants}
            return forest

        def get_skill_tree(skill):
            tree = {
                "code": skill.code,
                "variants": get_variant_forest(skill.variants),
                "candidates": get_variant_forest(skill.candidates),
            }
            return tree

        def get_skill_forest(skills):
            forest = {skill.idx: get_skill_tree(skill) for skill in skills}
            return forest

        status = {
            "Env": self.env_name,
            "idx": self.idx,
            "skills": get_skill_forest(self.skills),
            "impossibles": get_skill_forest(self.impossibles),
        }
        with open(self.status_output, "w") as fout:
            data_json = json.dumps(status)
            fout.write(data_json + "\n")
            logging.info(f"Saved status {self.idx} to {self.status_output}")
        self.status = status
        return

    def load_status(self, status_input=None):
        if status_input is None:
            status_input = self.status_output
        if not os.path.exists(status_input):
            logging.info(f"No status found in {status_input}, creating a new one.")
            return self
        with open(status_input, "r") as fin:
            status = json.load(fin)
        self.idx = status["idx"]
        self.env_name = status["Env"]

        def build_status(key="skills"):
            skills = []
            for skill_idx, skill_tree in status[key].items():
                skill = TaskNode(idx=skill_idx, code=skill_tree["code"])
                variants = []
                for variant_idx, variant_tree in skill_tree["variants"].items():
                    variant = SuccessNode(idx=variant_idx, stats=variant_tree["stats"])
                    for reward_idx, reward_values in variant_tree[
                        "best_reward"
                    ].items():
                        best_reward_node = RewardNode(
                            idx=reward_idx, priors=reward_values["priors"]
                        )
                        best_reward_node.summary = reward_values["summary"]
                        break
                    variant.best_reward = best_reward_node
                    variants.append(variant)
                skill.variants = variants
                skills.append(skill)
            return skills

        self.skills = build_status("skills")
        self.impossibles = build_status("impossibles")
        return self


class EnvNode(Node):
    def __init__(
        self,
        task_database: TaskDatabase,
        idx="E00",
        skills=None,
        impossibles=None,
        graph_output=None,
        status_output=None,
        centralized_task: CenteralizedTask = None,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.task_database = task_database
        self.centralized_task = centralized_task
        self.idx = idx
        self.G = nx.DiGraph()
        self.type = "Env"
        self.code = None
        self.initial_system = file_to_string(
            f"{self.prompt_dir}/task/initial_system.txt"
        )
        self.initial_user = file_to_string(f"{self.prompt_dir}/task/initial_user.txt")
        self.followup_user = file_to_string(f"{self.prompt_dir}/task/followup_user.txt")
        self.signature = file_to_string(f"{self.prompt_dir}/success/signature.txt")
        self.code_output_tip = file_to_string(
            f"{self.prompt_dir}/task/code_output_tip.txt"
        )
        if graph_output is None:
            graph_dir = f"{self.root_dir}/envs_gpt/graphs"
            graph_output = f"{graph_dir}/{self.env_name}_{self.idx}.json"
            if not os.path.exists(graph_dir):
                os.makedirs(graph_dir)
        if status_output is None:
            status_dir = f"{self.root_dir}/envs_gpt/status"
            status_output = f"{status_dir}/{self.env_name}_{self.idx}.json"
            if not os.path.exists(status_dir):
                os.makedirs(status_dir)
        self.graph_output = graph_output
        self.status_output = status_output
        self.status = None
        self.skills = [] if skills is None else skills
        self.impossibles = [] if impossibles is None else impossibles

    @property
    def num_skills(self):
        return len(self.skills)

    @property
    def num_impossibles(self):
        return len(self.impossibles)

    def get_skill_list(self):
        _skill_list_str = "\n".join(
            [f"({i+1}) {skill.code}" for i, skill in enumerate(self.skills)]
        )
        return _skill_list_str

    def get_impossible_list(self):
        _im_list_str = "\n".join(
            [f"({i+1}) {im.code}" for i, im in enumerate(self.impossibles)]
        )
        return _im_list_str

    def init(self):
        super().init()
        tasks_list = self.task_database.render()
        task_prompt = f"Recently known tasks are:\n{tasks_list}\n"
        if self.centralized_task is not None:
            centeral_task_list = self.centralized_task.center_tasks.render()
            task_prompt = (
                f"Previously known tasks are:\n{centeral_task_list}\n" + task_prompt
            )
        initial_system = (
            self.initial_system.format(tasks=task_prompt) + self.code_output_tip
        )
        initial_user = self.initial_user.format(
            env_obs_code_string=self.env_obs_code,
        )
        self.messages = [
            self._wrap_system_message(initial_system),
            self._wrap_user_message(initial_user),
        ]
        return self

    def propose(self):
        self.init().render()
        for i in range(5):
            msg, codes = self._propose(temperature_increase=i * 0.1)
            if codes is not None and len(codes) > 0:
                self.messages.extend(msg)
                break
        return codes

    def save_status(self):
        def get_variant_tree(variant):
            tree = {
                "best_reward": {
                    variant.best_reward.idx: {
                        "priors": variant.best_reward.priors,
                        "summary": variant.best_reward.summary,
                    }
                },
                "stats": variant.stats,
            }
            return tree

        def get_variant_forest(variants):
            forest = {variant.idx: get_variant_tree(variant) for variant in variants}
            return forest

        def get_skill_tree(skill):
            tree = {
                "code": skill.code,
                "variants": get_variant_forest(skill.variants),
                "candidates": get_variant_forest(skill.candidates),
            }
            return tree

        def get_skill_forest(skills):
            forest = {skill.idx: get_skill_tree(skill) for skill in skills}
            return forest

        status = {
            "Env": self.env_name,
            "idx": self.idx,
            "skills": get_skill_forest(self.skills),
            "impossibles": get_skill_forest(self.impossibles),
        }
        with open(self.status_output, "w") as fout:
            data_json = json.dumps(status)
            fout.write(data_json + "\n")
            logging.info(f"Saved status {self.idx} to {self.status_output}")
        self.status = status
        return

    def load_status(self, status_input=None):
        if status_input is None:
            status_input = self.status_output
        if not os.path.exists(status_input):
            logging.info(f"No status found in {status_input}, creating a new one.")
            return self
        with open(status_input, "r") as fin:
            status = json.load(fin)
        self.idx = status["idx"]
        self.env_name = status["Env"]

        def build_status(key="skills"):
            skills = []
            for skill_idx, skill_tree in status[key].items():
                skill = TaskNode(idx=skill_idx, code=skill_tree["code"])
                variants = []
                for variant_idx, variant_tree in skill_tree["variants"].items():
                    variant = SuccessNode(idx=variant_idx, stats=variant_tree["stats"])
                    for reward_idx, reward_values in variant_tree[
                        "best_reward"
                    ].items():
                        best_reward_node = RewardNode(
                            idx=reward_idx, priors=reward_values["priors"]
                        )
                        best_reward_node.summary = reward_values["summary"]
                        break
                    variant.best_reward = best_reward_node
                    variants.append(variant)
                skill.variants = variants
                skills.append(skill)
            return skills

        self.skills = build_status("skills")
        self.impossibles = build_status("impossibles")
        return self

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
        stat = {
            "num_skills": self.num_skills,
            "num_impossibles": self.num_impossibles,
        }
        return stat

    def render(self):
        for msg in self.messages:
            print("*" * 50 + f'role: {msg["role"]}' + "*" * 50)
            print(msg["content"])
        return

    def _gpt_call(self, messages, temperature_increase=0):
        choices = gpt_call(
            messages=messages,
            model=self.model,
            n_samples=1,
            temperature=self.temperature + temperature_increase,
        )
        if self.n_samples == 1:
            logging.info(f"GPT Output:\n " + choices[0]["message"]["content"] + "\n")
        resp = choices[0]["message"]
        return resp

    def _propose(self, temperature_increase=0) -> List[TaskNode]:
        messages = self.messages
        init_resp = self._gpt_call(messages, temperature_increase=temperature_increase)
        messages.extend([init_resp, wrap_user_message(self.followup_user)])
        resp = self._gpt_call(messages, temperature_increase=temperature_increase)
        codes = extract_tasks(resp["content"])
        self.messages = messages
        return resp, codes

    def _update_self_with_node(self, node):
        super()._update_self_with_node(node)
        if "skills" in node.keys():
            self.skills = node["skills"]
        if "impossibles" in node.keys():
            self.impossibles = node["impossibles"]
        return

    def _collect_skill(self, child):
        if child.num_variants > 0:
            self.skills.append(child)
            logging.info(
                f"Collected new skill {child.code} with {child.num_variants} variants."
            )
        elif child.num_variants == 0:
            self.impossibles.append(child)
            logging.info(f"Mission impossible on {child.code}.")
        else:
            pass
        return
