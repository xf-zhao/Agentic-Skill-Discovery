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
from zero_hero.core import TaskNode, ZEROHERO_ROOT_DIR
from zero_hero.task import TaskDatabase


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
        store_path=f'{ZEROHERO_ROOT_DIR}/envs_gpt/tasks/{env_name.replace(" ","_")}_{env_idx}.csv'
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
    wandbrun = wandb.init(
        project=cfg.wandb_project,
        config=my_cfg,
    )
    precedents = cfg.precedents
    if precedents is not None:
        if isinstance(precedents, str):
            if len(precedents) > 0:
                precedents = precedents.split(",")
            else:
                precedents = None
        elif isinstance(precedents, list):
            pass
        else:
            raise NotImplementedError
    task_node = TaskNode(
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
    for task_ite in range(cfg.task_iterations): # task_node.temperature += 0.2
        if task_node.num_variants >= cfg.num_variants:
            break
        success_nodes = task_node.propose(
            n_samples=cfg.n_reward_samples,
            iterations=2,
            temperature=cfg.temperature + task_ite * 0.2,
            model=cfg.model,
        )  # params for child init
        for reward_ite in range(cfg.reward_iterations):
            for success_node in success_nodes:
                reward_nodes = success_node.propose(
                    num_envs=cfg.num_envs,
                    headless=cfg.headless,
                    video=cfg.video,
                    memory_requirement=cfg.memory_requirement,
                    min_gpu_mem=cfg.min_gpu_mem,
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
        variants = task_node.variants[0]
        logging.info(
            f"Collected new skill {task} with {task_node.num_variants} variants: {variants}."
        )
    else:
        if task_node.num_candidates > 0:
            task_status = "compromised"
            variants = task_node.candidates
            logging.info(f"Mission compromised on {task} with candidates: {variants}.")
        else:
            task_status = "failed"
            logging.info(f"Mission impossible on {task}.")
            variants = ""
    tdb.load()
    tdb.update_task({"command": task, "status": task_status, "variants": variants})
    tdb.render()
    logging.info(f"Done! for task: {task}.")


if __name__ == "__main__":
    main()
