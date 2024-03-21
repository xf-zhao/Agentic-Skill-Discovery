import hydra
import logging
import os
import openai
from zero_hero.core import EnvNode, ZEROHERO_ROOT_DIR
from zero_hero.task import TaskDatabase


@hydra.main(config_path="cfg", config_name="config", version_base="1.1")
def main(cfg):
    openai.api_key = os.getenv("OPENAI_API_KEY")
    model = cfg.proposal.model
    env_name = cfg.env.env_name.lower()
    env_idx = f"E{cfg.seed:02d}"
    tdb = TaskDatabase(
        store_path=f'{ZEROHERO_ROOT_DIR}/envs_gpt/tasks/{env_name.replace(" ","_")}_{env_idx}.csv',
        target_num_skills=cfg.proposal.target_num_skills,
        failed_tolerance=cfg.proposal.failed_tolerance,
        proposal_batch=cfg.proposal.proposal_batch,
    )
    env_node = EnvNode(
        task_database=tdb,
        idx=env_idx,
        root_dir=ZEROHERO_ROOT_DIR,
        env_name=env_name,
        resume=cfg.resume,
        model=model,
        n_samples=1,
        temperature=cfg.temperature,
        skills=[],
        impossibles=[],
    )
    while not tdb.met_target() and not tdb.should_wait():
        tasks = env_node.propose()
        tdb.add_tasks(tasks)
        tdb.render()
        tdb.save()
        logging.info(
            f"Updated task database {tdb.store_path} with {len(tasks)} new tasks."
        )
    logging.info(f"Finished!")


if __name__ == "__main__":
    main()
