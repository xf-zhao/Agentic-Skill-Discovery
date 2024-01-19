import os
import traceback
import logging


# List and import all sub-modules (in directories)

dirname = os.path.dirname(__file__)
files = os.listdir(dirname)
for f in files:
    if f.startswith('_'):
        continue
    if os.path.isdir(f'{dirname}/{f}'):
        try:
            exec(f'from . import {f}')
        except Exception as e:
            err_msg = traceback.format_exc()
            logging.error(err_msg)