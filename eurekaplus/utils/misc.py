import subprocess
import re
import os
import json
import logging

# from .extract_task_code import file_to_string

def set_freest_gpu():
    freest_gpu = get_freest_gpu()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(freest_gpu)

def get_freest_gpu():
    sp = subprocess.Popen(['gpustat', '--json'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out_str, _ = sp.communicate()
    gpustats = json.loads(out_str.decode('utf-8'))
    # Find GPU with most free memory
    freest_gpu = min(gpustats['gpus'], key=lambda x: x['memory.used'])

    return freest_gpu['index']

def filter_traceback(s, idx=None):
    if 'Learning iteration 0/' in s:
        return ''
    pattern = r'(Traceback.*Error:\s.*?\n)'
    traceback_msgs = re.findall(pattern, s, re.DOTALL)
    if len(traceback_msgs) > 0:
        logging.warning(f'Some env contains errors to import, may hinder this run.')
        if idx is not None:
            for msg in traceback_msgs:
                if idx in msg:
                    return msg
        return traceback_msgs[0]
    return None

s = '''
  (3): ELU(alpha=1.0)
  (4): Linear(in_features=128, out_features=64, bias=True)
  (5): ELU(alpha=1.0)
  (6): Linear(in_features=64, out_features=1, bias=True)
)
Setting seed: 42
2024-01-27 21:04:10 [23,028ms] [Error] [__main__] Can't pickle <function SuccessCfg.move_cube_a_to_target at 0x7f1a2d0d5a20>: attribute lookup SuccessCfg.move_cube_a_to_target on envs_gpt.franka_table.Rb99e6a85.success failed
2024-01-27 21:04:10 [23,028ms] [Error] [__main__] Traceback (most recent call last):
  File "/data/xufeng/workspace/zero_hero/rsl_rl/train.py", line 185, in <module>
    main()
  File "/data/xufeng/workspace/zero_hero/rsl_rl/train.py", line 170, in main
    dump_pickle(os.path.join(log_dir, "params", "env.pkl"), env_cfg)
  File "/data/xufeng/.local/share/ov/pkg/isaac_sim-2023.1.1/orbit/source/extensions/omni.isaac.orbit/omni/isaac/orbit/utils/io/pkl.py", line 50, in dump_pickle
    pickle.dump(data, f)
_pickle.PicklingError: Can't pickle <function SuccessCfg.move_cube_a_to_target at 0x7f1a2d0d5a20>: attribute lookup SuccessCfg.move_cube_a_to_target on envs_gpt.franka_table.Rb99e6a85.success failed

[24.598s] Simulation is stopped. Shutting down the app.
2024-01-27 21:04:12 [24,612ms] [Warning] [omni.core.ITypeFactory] Module /data/xufeng/.local/share/ov/pkg/isaac_sim-2023.1.1/kit/exts/omni.activity.core/bin/libomni.activity.core.plugin.so remained loaded after unload request.


'''
if __name__ == "__main__":
    print(get_freest_gpu())
    x = filter_traceback(s)
    print(x)