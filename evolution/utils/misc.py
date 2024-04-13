import subprocess
import re
import os
import json
import logging

# from .extract_task_code import file_to_string

def set_freest_gpu():
    freest_gpu, gpu_avi = get_freest_gpu()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(freest_gpu)
    return gpu_avi

def get_freest_gpu(key='util'):
    if key == 'util':
        return get_freest_util_gpu()
    return get_freest_mem_gpu()

def get_freest_util_gpu():
    sp = subprocess.Popen(['gpustat', '--json'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out_str, _ = sp.communicate()
    gpustats = json.loads(out_str.decode('utf-8'))
    # Find GPU with most free memory
    freest_gpu = max(gpustats['gpus'], key=lambda x: 100 - x['utilization.gpu'])
    gpu_util_avi = (100 - freest_gpu['utilization.gpu']) # * 100 %
    return freest_gpu['index'], gpu_util_avi

def get_freest_mem_gpu():
    sp = subprocess.Popen(['gpustat', '--json'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out_str, _ = sp.communicate()
    gpustats = json.loads(out_str.decode('utf-8'))
    # Find GPU with most free memory
    freest_gpu = max(gpustats['gpus'], key=lambda x: x['memory.total'] - x['memory.used'])
    gpu_mem_avi = (freest_gpu['memory.total'] - freest_gpu['memory.used'])/1024 #GB
    return freest_gpu['index'], gpu_mem_avi

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
