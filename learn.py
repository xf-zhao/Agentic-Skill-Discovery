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
from zero_hero.core import EnvNode, TaskNode
from zero_hero.task import TaskDatabase


ZEROHERO_ROOT_DIR = f"{os.getcwd()}"


@hydra.main(config_path="cfg", config_name="config", version_base="1.1")
def main(cfg):
    workspace_dir = Path.cwd()
    logging.info(f"Workspace: {workspace_dir}")
    openai.api_key = os.getenv("OPENAI_API_KEY")
    logging.info(cfg)
    env_name = cfg.env.env_name.lower()
    env_idx = f"E{cfg.seed:02d}"
    tdb = TaskDatabase(
        store_path=f'{ZEROHERO_ROOT_DIR}/envs_gpt/tasks/{env_name.replace(" ","_")}_{env_idx}.csv'
    )
    tdb.render()
    task = tdb.pop()
    if task is None:
        logging.info(f"Nothing to do with task database {tdb.store_path}!")
        tdb.render()
        return
    cfg.task = task
    my_cfg = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    wandbrun = wandb.init(
        project=cfg.wandb_project,
        config=my_cfg,
    )
    task_node = TaskNode(
        code=cfg.task,
        n_samples=cfg.n_success_samples,
        temperature=cfg.temperature,
        model=cfg.model,
    ).init()
    bc = BehaviorCaptioner(
        init_sys_prompt=f"{task_node.prompt_dir}/task/behavior_context.txt",
    )
    logging.info(f"Learning skill: {task}.")
    for task_ite in range(cfg.task_iterations):
        if task_node.num_variants >= cfg.num_variants:
            break
        success_nodes = task_node.propose(
            n_samples=cfg.n_reward_samples,
            iterations=2,
            temperature=cfg.temperature,
            model=cfg.model,
        )  # params for child init
        for reward_ite in range(cfg.reward_iterations):
            for success_node in success_nodes:
                reward_nodes = success_node.propose(
                    num_envs=cfg.num_envs,
                    headless=cfg.headless,
                    video=cfg.video,
                    memory_requirement=cfg.memory_requirement,
                    max_iterations=cfg.max_iterations,
                    task_ite=task_ite,
                    reward_ite=reward_ite,
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
            task_stat = task_node.collect(
                behavior_captioner=bc
            )  # check behavior caption
            wandbrun.log(
                {
                    **task_stat,
                    "task_ite": task_ite,
                    "reward_ite": reward_ite,
                }
            )

    if task_node.num_variants > 0:
        task_status = "completed"
        logging.info(
            f"Collected new skill {task} with {task_node.num_variants} variants."
        )
    elif task_node.num_variants == 0:
        task_status = "failed"
        logging.info(f"Mission impossible on {task}.")
    else:
        raise NotImplementedError
    tdb.load()
    tdb.update_task({"command": task, "status": task_status})
    tdb.save()
    logging.info(f"Done! for task: {task}.")
    tdb.render()


if __name__ == "__main__":
    main()
