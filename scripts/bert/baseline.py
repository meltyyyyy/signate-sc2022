#!/usr/bin/env python
# coding: utf-8

# ### Basic configuration

# In[1]:


class Config:
    notebook = "BERT/Baseline"
    script = "bert/baseline"

    n_splits = 5
    seed = 42
    target = "target"

    # Colab Env
    api_path = "/content/drive/MyDrive/workspace/kaggle.json"
    drive_path = "/content/drive/MyDrive/workspace/kaggle-amex"

    # Kaggle Env
    kaggle_dataset_path = None

    # Reka Env
    dir_path = "/Users/takeru.abe/Development/signate-sc2022"

    def is_notebook():
        if 'get_ipython' not in globals():
            return False
        env_name = get_ipython().__class__.__name__  # type: ignore
        if env_name == 'TerminalInteractiveShell':
            return False
        return True


# ### Import basic libraries

# In[2]:


from tqdm.auto import tqdm
import seaborn as sns
import os
import json
import warnings
import shutil
import logging
import joblib
import random
import datetime
import sys
import gc
import multiprocessing
import pickle
import subprocess
from subprocess import PIPE
import ntpath

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('seaborn-pastel')
sns.set_palette("winter_r")

tqdm.pandas()
warnings.filterwarnings('ignore')


# ### Seeding

# In[ ]:


def seed_everything():
    random.seed(Config.seed)
    np.random.seed(Config.seed)
    os.environ['PYTHONHASHSEED'] = str(Config.seed)

seed_everything()


# ### Path configuration

# In[ ]:


def path_setup():
    INPUT = os.path.join(Config.dir_path, 'input')
    OUTPUT = os.path.join(Config.dir_path, 'output')
    SUBMISSION = os.path.join(Config.dir_path, 'submissions')
    OUTPUT_EXP = os.path.join(OUTPUT, Config.script)
    EXP_MODEL = os.path.join(OUTPUT_EXP, "model")
    EXP_FIG = os.path.join(OUTPUT_EXP, "fig")
    NOTEBOOK = os.path.join(Config.dir_path, "Notebooks")
    SCRIPT = os.path.join(Config.dir_path, "scripts")

    # make dir
    for dir in [
            INPUT,
            OUTPUT,
            SUBMISSION,
            OUTPUT_EXP,
            EXP_MODEL,
            EXP_FIG,
            NOTEBOOK,
            SCRIPT]:
        os.makedirs(dir, exist_ok=True)

    if Config.is_notebook():
        notebook_path = os.path.join(NOTEBOOK, Config.notebook + ".ipynb")
        script_path = os.path.join(SCRIPT, Config.script + ".py")
        dir, _ = ntpath.split(script_path)
        subprocess.run(f"mkdir -p {dir}; touch {script_path}",
                       shell=True,
                       stdout=PIPE,
                       stderr=PIPE,
                       text=True)
        subprocess.run(
            f"jupyter nbconvert --to python {notebook_path} --output {script_path}",
            shell=True,
            stdout=PIPE,
            stderr=PIPE,
            text=True)

path_setup()


# # Load data

# In[ ]:


print('aaaaa')

