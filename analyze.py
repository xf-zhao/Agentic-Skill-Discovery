import pandas as pd
import matplotlib.pyplot as plt
import re
from zero_hero.core import ZEROHERO_ROOT_DIR

columns = ["command", "status", "variants"]
rec = pd.read_csv("rec.csv")
rec_best = pd.read_csv("rec_best.csv")


def find_succ_weight(idx):
    fp = f"{ZEROHERO_ROOT_DIR}/envs_gpt/franka_table/{idx}/success.py"
    try:
        with open(fp) as f:
            script = f.read()
            weight = re.findall(r"weight\s*=\s*(\d+)", script)[-1]
            weight = int(weight)
    except FileNotFoundError as e:
        weight = 1
    return weight


# normalize succ
success_weight = rec["reward_idx"].apply(find_succ_weight)
success = rec["success"] / success_weight
# tackle missing weight
rec["success"] = success.apply(lambda x: x / 100 if x > 1 else x).apply(
    lambda x: x if x > 0 else 0
)
rec["successful"] = rec["success"] > 0

recb = rec_best.join(rec.set_index("reward_idx"), on="reward_idx").dropna()

## reports
groups = ["task", "reward_ite"]
rg = rec.groupby(groups)
rgb = recb.groupby(groups)

# succ learned rate
success_rate = rg.apply(lambda x: x["successful"].mean())
success_rateb = rgb.apply(lambda x: x["successful"].mean())
task_list = success_rate.index.get_level_values(0).unique()

# succ learned value only for successful runs
success_value = (
    rec[rec["successful"]].groupby(groups).apply(lambda x: x["success"].mean())
)
success_valueb = (
    recb[recb["successful"]].groupby(groups).apply(lambda x: x["success"].mean())
)

# syntax error rate
reward_syntax_error = rg.apply(lambda x: x["reward_num_syntax_error"].mean())
# reward_syntax_errorb = rgb.apply(lambda x: x["reward_num_syntax_error"].mean())
exec_error = rg.apply(lambda x: (1 - x["exec_success"]).mean())
# exec_errorb = rgb.apply(lambda x: (1 - x["exec_success"]).mean())

# gpt-4v vs success
gpt_succ = rgb.apply(lambda x: x["gpt-4v-succ"].mean())
gpt_succ_agree = rgb.apply(lambda x: (x["gpt-4v-succ"] == x["successful"]).mean())

# variant / candidate count
variant = rgb[["gpt-4v-succ", "successful"]].apply(
    lambda x: (x["gpt-4v-succ"] & x["successful"]).sum()
)
candidate = rgb[["gpt-4v-succ", "successful"]].apply(
    lambda x: (~x["gpt-4v-succ"] & x["successful"]).sum()
)


for t in task_list:
    print("-" * 120)
    print(t)
    print(f"positive rate: {success_rate[t]}")
    print(f"success rate: {success_value[t]}")
    print(f"reward syntax error: {reward_syntax_error[t]}")
    print(f"exec succ: {exec_error[t]}")
    print("----")
    print(f"best positive rate: {success_rateb[t]}")
    print(f"best success rate: {success_valueb[t]}")
    print(f"gpt-4v-positive rate: {gpt_succ[t]}")
    print(f"succ func acc: {gpt_succ_agree[t]}")
    print("----")
    print(f"variants: {variant[t]}")
    print(f"candidates: {candidate[t]}")

pass
