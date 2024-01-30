import hydra
import logging
import os
import openai
from pathlib import Path
from eurekaplus.utils.misc import *
from eurekaplus.utils.extract_task_code import *
from zero_hero.behavior import BehaviorCaptioner
from zero_hero.core import EnvNode


ZEROHERO_ROOT_DIR = f"{os.getcwd()}"

def learn_next_skill(env_node, cfg, bc):
    task_nodes = env_node.propose(
        n_samples=cfg.n_success_samples, temperature=cfg.temperature, model=cfg.model
    )
    for task_node in task_nodes:
        logging.info(f"Learn-Next-Skill: {task_node.code}.")
        break
    for _ in range(cfg.task_iterations):
        if task_node.num_variants >= cfg.num_variants:
            continue
        success_nodes = task_node.propose(
            n_samples=cfg.n_reward_samples,
            iterations=2,
            temperature=cfg.temperature,
            model=cfg.model,
        )  # params for child init
        for __ in range(cfg.reward_iterations):
            for success_node in success_nodes:
                reward_nodes = success_node.propose(
                    num_envs=cfg.num_envs,
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
    env_node.save_status()
    return env_node


@hydra.main(config_path="cfg", config_name="config", version_base="1.1")
def main(cfg):
    workspace_dir = Path.cwd()
    logging.info(f"Workspace: {workspace_dir}")
    openai.api_key = os.getenv("OPENAI_API_KEY")
    env = cfg.env
    model = cfg.model
    logging.info(cfg)
    logging.info(f"Using LLM: {model}")

    env_name = cfg.env.env_name.lower()
    env_node = (
        EnvNode(
            idx=f"E{cfg.seed:02d}",
            root_dir=ZEROHERO_ROOT_DIR,
            env_name=env_name,
            resume=cfg.resume,
            model=model,
            n_samples=1,
            temperature=cfg.temperature,
            skills=[],
            impossibles=[],
        )
        .init()
        .load_status()
    )
    logging.info(f"Env: {env.env_name} / {env_node.idx}")
    bc = BehaviorCaptioner(
        init_sys_prompt=f"{env_node.prompt_dir}/task/behavior_context.txt",
    )
    for i_task in range(cfg.target_num_skills * 10):
        print("-" * 100)
        logging.info(
            f"{env_node.idx}/{i_task}: Acquired {env_node.num_skills} skills, gave up on {env_node.num_impossibles} impossibles."
        )
        if env_node.num_skills >= cfg.target_num_skills:
            logging.info(f"Finished accumulating {env_node.num_skills} skills.")
            break
        learn_next_skill(env_node, cfg, bc)
    logging.info(f"Acquired {env_node.num_skills} skills: {env_node.get_skill_list()}")
    logging.info(
        f"Gave up on {env_node.num_impossibles} impossibiles: {env_node.get_impossible_list()}"
    )
    logging.info("All done!")


if __name__ == "__main__":
    main()
