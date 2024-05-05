import hydra
import wandb
import logging
import os
import openai
from omegaconf import OmegaConf
from pathlib import Path
from evolution.utils.misc import *
from evolution.utils.extract_task_code import *
from zero_hero.behavior import BehaviorCaptioner
from zero_hero.core import TaskNode, TaskDatabase
from zero_hero.utils import FakeWandb


@hydra.main(config_path="cfg", config_name="config", version_base="1.1")
def main(cfg):
    workspace_dir = Path.cwd()
    logging.info(f"Workspace: {workspace_dir}")
    openai.api_key = os.getenv("OPENAI_API_KEY")
    logging.info(cfg)
    env_name = cfg.env.env_name.lower()
    task = cfg.task
    specified_task = task is not None and len(task) > 0
    seed = 99 if specified_task else cfg.seed
    env_idx = f"E{seed:02d}"
    tdb = TaskDatabase(
        env_name=env_name,
        env_idx=env_idx,
    )
    if specified_task:
        tdb.add_task(task)
    tdb.render()
    task = tdb.pop()
    if task is None or task == "":
        logging.info(f"Nothing to do with task database {tdb.store_path}!")
        return
    cfg.task = task
    cfg.seed = seed
    my_cfg = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    if cfg.use_wandb:
        wandbrun = wandb.init(
            project=cfg.wandb_project,
            config=my_cfg,
        )
    else:
        wandbrun = FakeWandb(my_cfg)
    precedents = cfg.precedents
    task_node: TaskNode = TaskNode(
        code=cfg.task,
        n_samples=cfg.n_success_samples,
        temperature=cfg.temperature,
        model=cfg.model,
        precedents=precedents,
    ).init()
    bc = BehaviorCaptioner(
        init_sys_prompt=f"{task_node.prompt_dir}/task/behavior_context.txt",
    )
    logging.info(f"Learning skill: {task}.")
    for task_ite in range(cfg.task_iterations):
        task_node.temperature += 0.2
        if task_node.num_variants >= cfg.num_variants:
            break
        task_node.propose(
            n_samples=cfg.n_reward_samples,
            iterations=2,
            temperature=cfg.temperature + task_ite * 0.2,
            model=cfg.model,
        )  # params for child init
        for reward_ite in range(cfg.reward_iterations):
            success_nodes = task_node.children
            for success_node in success_nodes:
                reward_nodes = success_node.propose(
                    num_envs=cfg.num_envs,
                    headless=cfg.headless,
                    video=cfg.video,
                    memory_requirement=cfg.memory_requirement,
                    min_gpu=cfg.min_gpu,
                    max_iterations=cfg.max_iterations,
                    task_ite=task_ite,
                    reward_ite=reward_ite,
                    behavior_captioner=bc,
                )
                for node in reward_nodes:
                    node.run()
            for success_node in success_nodes:
                _, succ_stat = success_node.collect()
                wandbrun.log(
                    {
                        **succ_stat,
                        "reward_ite": reward_ite,
                        "task_ite": task_ite,
                    }
                )
            task_stat = task_node.collect()
            wandbrun.log(
                {
                    **task_stat,
                    "task_ite": task_ite,
                    "reward_ite": reward_ite,
                }
            )

    if task_node.num_variants > 0:
        task_status = "completed"
        variants = [v.best_reward.idx for v in task_node.variants]
        logging.info(
            f"Collected new skill {task} with {task_node.num_variants} variants: {variants}."
        )
    else:
        task_status = "failed"
        logging.info(f"Mission impossible on {task}.")
        variants = [""]
    tdb.load()
    tdb.update_task({"command": task, "status": task_status, "variants": variants[0]})
    tdb.render()
    logging.info(f"Done! for task: {task}.")


if __name__ == "__main__":
    main()
