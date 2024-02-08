import pandas as pd
import os


class TaskDatabase:
    def __init__(self, store_path="tasks.csv") -> None:
        columns = ["command", "status"]
        if os.path.exists(store_path):
            df = pd.read_csv(store_path)
        else:
            df = pd.DataFrame(columns=columns)
        self.df = df
        if not os.path.exists(store_path):
            os.makedirs(store_path, exist_ok=True)
        self.store_path = store_path

    @property
    def commands(self):
        return self.df["commands"]

    @property
    def status(self):
        return self.df["status"]

    def add_task(self, task: dict):
        df = self.df
        row = pd.Series(task)
        df = pd.concat([df, pd.DataFrame([row], columns=row.index)]).reset_index(
            drop=True
        )
        self.df = df
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

    def pop(self):
        command = None
        df = self.df
        indices = df.loc[df.status == "todo"].index
        if len(indices) > 0:
            index_to_pop = indices[0]
            df.loc[index_to_pop, "status"] = "doing"
            command = df.loc[index_to_pop, "command"]
        df = self.df
        return command
