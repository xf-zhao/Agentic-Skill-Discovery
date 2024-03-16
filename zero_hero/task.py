import time
import os
import pandas as pd


class Database:
    def get_attr(self, attr, index=None, task=None):
        if index is not None:
            variants = self.df.loc[index][attr]
        elif task is not None:
            variants = self.df[self.df["command"] == task][attr]
        else:
            raise NotImplementedError
        return variants.values


class TaskDatabase(Database):
    def __init__(
        self,
        store_path="tasks.csv",
        target_num_skills=64,
        failed_tolerance=None,
        proposal_batch=10,
    ) -> None:
        self.store_path = store_path
        self.target_num_skills = target_num_skills
        self.failed_tolerance = (
            failed_tolerance if failed_tolerance is not None else target_num_skills * 2
        )
        self.proposal_batch = proposal_batch
        self.load()

    def met_target(self):
        is_met = (
            self.num_skills >= self.target_num_skills
            or self.num_failed >= self.failed_tolerance
        )
        return is_met

    def should_wait(self):
        return self.num_wait >= self.proposal_batch

    def load(self):
        store_path = self.store_path
        columns = ["command", "status", "variants"]
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

    @property
    def variants(self):
        return self.df["variants"]

    @property
    def num_wait(self):
        return (self.df["status"] == "todo").sum() + (
            self.df["status"] == "doing"
        ).sum()

    @property
    def num_skills(self):
        return (self.df["status"] == "completed").sum()

    @property
    def num_failed(self):
        return (self.df["status"] == "failed").sum()

    def add_task(self, task: str):
        df = self.df
        if task in self.df["command"].values:
            self.refresh_task(task)
        else:
            row = pd.Series({"command": task, "status": "todo", "variants": ""})
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
        command = task['command']
        if command not in self.df["command"].values:
            self.add_task(command)
        df = self.df
        df.loc[df.command == command, "status"] = task["status"]
        df.loc[df.command == command, "variants"] = task["variants"]
        self.df = df
        self.save()
        return

    def refresh_task(self, task: str):
        self.update_task({"command": task, "status": "todo", "variants": ""})
        return

    def save(self):
        self.df.to_csv(self.store_path, index=False)
        print(f'self.df:\n {self.df}')
        print(f'Saved data to {self.store_path}')

    def render(self):
        df = self.df
        numbered_tasks = "\n".join(
            [
                f"({i+1}) Task: {row.command.rstrip('.')}. Status: {row.status}. Variants: {row.variants}"
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
            self.df = df
            self.save()
        else:
            task = None
        return task

    @property
    def skills(self):
        df = self.df
        skill_df = df[df["status"] == "completed"]
        return skill_df
