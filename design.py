import hydra
import pandas as pd
import wandb
import logging
import os
import openai
from omegaconf import OmegaConf
from pathlib import Path
from zero_hero.core import TaskDatabase, CenteralizedTask



@hydra.main(config_path="cfg", config_name="config", version_base="1.1")
def main(cfg):
    workspace_dir = Path.cwd()
    logging.info(f"Workspace: {workspace_dir}")
    openai.api_key = os.getenv("OPENAI_API_KEY")
    logging.info(cfg)
    env_name = cfg.env.env_name.lower()
    ct = CenteralizedTask(
        env_name=env_name,
        model=cfg.design.model,
        temperature=cfg.design.temperature,
    )
    seeds = cfg.design.seeds
    for seed in seeds:
        env_idx = f"E{seed:02d}"
        tdb = TaskDatabase(
            env_name=env_name,
            env_idx=env_idx,
        )
        tdb = ct.filter(task_database=tdb)
        pass


if __name__ == "__main__":
    main()
