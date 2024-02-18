import time
import os
import pandas as pd


class TaskDatabase:
    def __init__(self, store_path="tasks.csv") -> None:
        self.store_path = store_path
        self.load()

    def load(self):
        store_path = self.store_path
        columns = ["command", "status"]
        if os.path.exists(store_path):
            df = pd.read_csv(store_path)
        else:
            os.makedirs(os.path.dirname(store_path), exist_ok=True)
            df = pd.DataFrame(columns=columns)
        self.df = df
        return self

    @property
    def commands(self):
        return self.df["commands"]

    @property
    def status(self):
        return self.df["status"]

    def num_skills(self):
        return self.df["status"] == "complete"

    def add_task(self, task: dict):
        df = self.df
        row = pd.Series({"command": task, "status": "todo"})
        df = pd.concat([df, pd.DataFrame([row], columns=row.index)]).reset_index(
            drop=True
        )
        self.df = df
        return

    def add_tasks(self, tasks):
        for task in tasks:
            self.add_task(task)
        return

    def update_task(self, task: dict):
        df = self.df
        df.loc[df.command == task["command"], "status"] = task["status"]
        self.df = df
        self.save()
        return

    def save(self):
        self.df.to_csv(self.store_path, index=False)

    def render(self):
        df = self.df
        numbered_tasks = "\n".join(
            [
                f"({i+1}) Task: {row.command.rstrip('.')}. Status: {row.status}"
                for i, row in df.iterrows()
            ]
        )
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
            df = self.df
            self.save()
        else:
            task = None
        return task
