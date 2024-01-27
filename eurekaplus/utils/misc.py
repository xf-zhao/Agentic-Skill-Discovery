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
    lines = s.split('\n')
    for i, line in enumerate(lines):
        if 'Learning iteration 0/' in line:
            return ''
    pattern = r'(Traceback.*?Error:\s.*?\n)'
    traceback_msgs = re.findall(pattern, s, re.DOTALL)
    if len(traceback_msgs) > 0:
        logging.warning(f'Some env contains errors to import, may hinder this run.')
        if idx is not None:
            for msg in traceback_msgs:
                if idx in msg:
                    return msg
    return ''  # Return an empty string if no Traceback is found


if __name__ == "__main__":
    print(get_freest_gpu())