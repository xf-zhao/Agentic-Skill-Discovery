import os, sys

dirname = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, dirname)

from zero_hero.task import TaskDatabase

skill = {}

task = {
    "command": "Move cube a to the target position.",
    "status": "todo",  # doing, completed, failed
}


cur_dir = os.path.dirname(__file__)
taskdb = TaskDatabase(store_path=f"{cur_dir}/tasks.csv")
taskdb.add_task(task)
taskdb.save()
taskdb.render()
taskdb.pop()
pass
