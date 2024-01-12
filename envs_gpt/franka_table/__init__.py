import os
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
            logging.info(e)

