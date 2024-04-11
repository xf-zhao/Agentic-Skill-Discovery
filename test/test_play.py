import os
from zero_hero.behavior import BehaviorCaptioner

root_dir = os.getcwd()

from zero_hero.core import RewardNode

rn = RewardNode(
    idx="Rd67c82d3", root_dir=root_dir, headless=False, memory_requirement=8
).init()

playbacks = rn.play()

bc = BehaviorCaptioner(
    init_sys_prompt="/data/xufeng/workspace/zero_hero/evolution/utils/prompts/task/behavior_context.txt"
)

msg = bc.conclude(
    playbacks,
    "Move cube A to the target position",
)

bc2 = BehaviorCaptioner(
    init_sys_prompt="/data/xufeng/workspace/zero_hero/evolution/utils/prompts/task/behavior_context.txt",
    caption_output="caption2.txt",
)
msg2 = bc2.conclude(
    playbacks,
    "Reach the target position",
)
pass
