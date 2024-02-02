import os
from zero_hero.behavior import BehaviorCaptioner

root_dir = os.getcwd()

from zero_hero.core import RewardNode

rn = RewardNode(
    idx="Rd67c82d3", root_dir=root_dir, headless=False, memory_requirement=4
).init()

behavior_image_paths, _ = rn.play()

bc = BehaviorCaptioner(
    init_sys_prompt="/data/xufeng/workspace/zero_hero/eurekaplus/utils/prompts/task/behavior_context.txt"
)

msg = bc.describe(
    behavior_image_paths,
    "/data/xufeng/workspace/zero_hero/envs_gpt/franka_table/Rd67c82d3/logs/model_1499_videos/rl-video-step-0-obs.json",
    "Move cube A to the target position",

)

bc2 = BehaviorCaptioner(
    init_sys_prompt="/data/xufeng/workspace/zero_hero/eurekaplus/utils/prompts/task/behavior_context.txt",
    caption_output="caption2.txt",
)
msg2 = bc2.describe(
    behavior_image_paths,
    "/data/xufeng/workspace/zero_hero/envs_gpt/franka_table/Rd67c82d3/logs/model_1499_videos/rl-video-step-0-obs.json",
    "Reach the target position",
)
pass
