import hydra
import wandb
import logging
import os
import openai
from omegaconf import OmegaConf
from pathlib import Path
from eurekaplus.utils.misc import *
from eurekaplus.utils.extract_task_code import *
from zero_hero.behavior import BehaviorCaptioner
from zero_hero.core import EnvNode
from zero_hero.task import TaskDatabase


ZEROHERO_ROOT_DIR = f"{os.getcwd()}"


@hydra.main(config_path="cfg", config_name="config", version_base="1.1")
def main(cfg):
    workspace_dir = Path.cwd()
    logging.info(f"Workspace: {workspace_dir}")
    openai.api_key = os.getenv("OPENAI_API_KEY")
    my_cfg = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    wandbrun = wandb.init(
        project=cfg.wandb_project,
        config=my_cfg,
    )

    env = cfg.env
    model = cfg.model
    logging.info(cfg)
    logging.info(f"Using LLM: {model}")

    env_name = cfg.env.env_name.lower()
    tdb = TaskDatabase(
        store_path=f'{ZEROHERO_ROOT_DIR}/envs_gpt/tasks/{env_name.replace(" ","_")}.csv'
    )
    env_node = (
        EnvNode(
            task_database=tdb,
            idx=f"E{cfg.seed:02d}",
            root_dir=ZEROHERO_ROOT_DIR,
            env_name=env_name,
            resume=cfg.resume,
            model=model,
            n_samples=1,
            temperature=cfg.temperature,
            skills=[],
            impossibles=[],
        ).init()
        # .load_status()
    )
    logging.info(f"Env: {env.env_name} / {env_node.idx}")

    for i_task in range(cfg.target_num_skills * 10):
        print("-" * 100)
        logging.info(
            f"{env_node.idx}/{i_task}: Acquired {env_node.num_skills} skills, gave up on {env_node.num_impossibles} impossibles."
        )
        if env_node.num_skills >= cfg.target_num_skills:
            logging.info(f"Finished accumulating {env_node.num_skills} skills.")
            break

    logging.info("All done!")


if __name__ == "__main__":
    main()
