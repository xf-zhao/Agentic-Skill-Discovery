import hydra
import pandas as pd
import wandb
import logging
import os
import openai
from omegaconf import OmegaConf
from pathlib import Path
from eurekaplus.utils.misc import *
from eurekaplus.utils.extract_task_code import *
from zero_hero.core import (
    ZEROHERO_ROOT_DIR,
    wrap_system_message,
    wrap_user_message,
    gpt_call,
    extract_tasks,
)
from zero_hero.task import TaskDatabase
from zero_hero.core import ZEROHERO_ROOT_DIR


class CenteralizedTask:
    def __init__(
        self,
        store_path="centeralized_task.csv",
        env_name=None,
        idx=0,
        model="gpt-3.5-turbo-0125",
        temperature=0.7,
    ) -> None:
        self.env_name = env_name
        self.root_dir = ZEROHERO_ROOT_DIR
        self.prompt_dir = f"{self.root_dir}/eurekaplus/utils/prompts"
        self.center_tasks = TaskDatabase(store_path=store_path).load()
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
        self.df = df
        logging.info(
            f"Updated centralized task database {tdb.store_path} with {len(task_database.df)} new tasks."
        )
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


@hydra.main(config_path="cfg", config_name="config", version_base="1.1")
def main(cfg):
    workspace_dir = Path.cwd()
    logging.info(f"Workspace: {workspace_dir}")
    openai.api_key = os.getenv("OPENAI_API_KEY")
    logging.info(cfg)
    env_name = cfg.env.env_name.lower()
    ct = CenteralizedTask(
        store_path=f'{ZEROHERO_ROOT_DIR}/envs_gpt/tasks/{env_name.replace(" ","_")}_centraliaze_task.csv',
        env_name=env_name,
        idx="DESIGN",
        model=cfg.design.model,
        temperature=cfg.design.temperature,
    )
    seeds = cfg.design.seeds
    for seed in seeds:
        env_idx = f"E{seed:02d}"
        tdb = TaskDatabase(
            store_path=f'{ZEROHERO_ROOT_DIR}/envs_gpt/tasks/{env_name.replace(" ","_")}_{env_idx}.csv'
        )
        tdb = ct.filter(task_database=tdb)
        pass


if __name__ == "__main__":
    main()
