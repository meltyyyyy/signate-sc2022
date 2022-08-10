#!/usr/bin/env python
# coding: utf-8

# ### Basic configuration



class Config:
    notebook = "Linear/Baseline"
    script = "linear/baseline"

    n_splits = 5
    seed = 42

    # Reka Env
    dir_path = "/home/abe/kaggle/signate-sc2022"

    def is_notebook():
        if 'get_ipython' not in globals():
            return False
        env_name = get_ipython().__class__.__name__  # type: ignore
        if env_name == 'TerminalInteractiveShell':
            return False
        return True


# ### Import basic libraries



from tqdm.auto import tqdm
import seaborn as sns
import os
import gc
import warnings
import random
from glob import glob
import subprocess
from subprocess import PIPE
import ntpath
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('seaborn-pastel')
sns.set_palette("winter_r")
warnings.filterwarnings('ignore')
tqdm.pandas()


# ### Seeding



def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

seed_everything(Config.seed)


# ### Path configuration



def path_setup(cfg):
    cfg.INPUT = os.path.join(Config.dir_path, 'input')
    cfg.OUTPUT = os.path.join(Config.dir_path, 'output')
    cfg.SUBMISSION = os.path.join(Config.dir_path, 'submissions')
    cfg.OUTPUT_EXP = os.path.join(cfg.OUTPUT, Config.script)
    cfg.EXP_MODEL = os.path.join(cfg.OUTPUT_EXP, "model")
    cfg.EXP_PREDS = os.path.join(cfg.OUTPUT_EXP, "preds")
    cfg.EXP_FIG = os.path.join(cfg.OUTPUT_EXP, "fig")
    cfg.NOTEBOOK = os.path.join(Config.dir_path, "Notebooks")
    cfg.SCRIPT = os.path.join(Config.dir_path, "scripts")

    # make dir
    for dir in [
            cfg.INPUT,
            cfg.OUTPUT,
            cfg.SUBMISSION,
            cfg.OUTPUT_EXP,
            cfg.EXP_MODEL,
            cfg.EXP_PREDS,
            cfg.EXP_FIG,
            cfg.NOTEBOOK,
            cfg.SCRIPT]:
        os.makedirs(dir, exist_ok=True)

    if Config.is_notebook():
        notebook_path = os.path.join(cfg.NOTEBOOK, Config.notebook + ".ipynb")
        script_path = os.path.join(cfg.SCRIPT, Config.script + ".py")
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
        subprocess.run(
            shell=True,
            stdout=PIPE,
            stderr=PIPE,
            text=True)
    
    return cfg

cfg = path_setup(Config)


# # Load data



# load data
train = pd.read_csv(os.path.join(cfg.INPUT, 'train.csv'))
test = pd.read_csv(os.path.join(cfg.INPUT, 'test.csv'))
sub = pd.read_csv(os.path.join(cfg.INPUT, 'submit_sample.csv'), header=None)

# preprocess target
train['jobflag'] -= 1




train.head(5)


# ## TFIDF-Vectorizer



import texthero as hero

def vectorize(train : pd.DataFrame, test : pd.DataFrame):
    train['tfidf'] = hero.tfidf(train['description'])
    test['tfidf'] = hero.tfidf(test['description'])
    return train, test
    
train, test = vectorize(train, test)

