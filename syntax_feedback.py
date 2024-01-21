import subprocess
import sys
import os
from eurekaplus.utils.misc import filter_traceback

try:
    p = subprocess.check_output(['python', 'test.py'], stderr=subprocess.STDOUT)
except subprocess.CalledProcessError as e:
    traceback_msg = e.output.decode()

traceback_msg = '''
Traceback (most recent call last):
  File "/data/xufeng/workspace/zero_hero/test.py", line 1, in <module>
    a =err
       ^^^
NameError: name 'err' is not defined
'''

y = filter_traceback(traceback_msg)
print(y)