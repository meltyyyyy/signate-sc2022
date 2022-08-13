#!/usr/bin/env python
# coding: utf-8
"""summery
fold0 : 0.9044676174778603
fold1 : 0.9477128376584898
fold2 : 0.8800030946196042
fold3 : 0.9457377702734393
fold4 : 0.6911483906164757
oof score: 0.8738139421291737
This is leakage!!
"""


from tqdm import tqdm
import re
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from sklearn.svm import LinearSVC
import torch
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import os
import warnings
tqdm.pandas()
plt.style.use('seaborn-pastel')
warnings.filterwarnings('ignore')


class Config:
    script = "roberta/linear/baseline"

    n_splits = 5
    seed = 42

    n_classes = 4

    # Reka Env
    dir_path = "/home/abe/kaggle/signate-sc2022"


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

    return cfg


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def fit_lsvb(X, y):
    models = []
    scores = []

    skf = StratifiedKFold(
        n_splits=cfg.n_splits,
        shuffle=True,
        random_state=cfg.seed)

    for fold, (trn_index, val_index) in enumerate(skf.split(X, y)):
        X_train, y_train = X.iloc[trn_index], y.iloc[trn_index]
        X_valid, y_valid = X.iloc[val_index], y.iloc[val_index]

        model = LinearSVC(
            penalty='l2',
            loss='squared_hinge',
            multi_class='ovr',
            random_state=cfg.seed,
            verbose=0)

        model.fit(X_train, y_train)
        # --------- prediction ---------
        pred = model.predict(X_valid)
        score = f1_score(y_valid, y_pred=pred, average='macro')
        print(f"fold{fold} : {score}")

        # --------- save ---------
        models.append(model)
        scores.append(score)

    print("oof score: {}".format(np.mean(scores)))
    return models


def inference_lsvc(models, X):
    pred = np.array([model.predict(X) for model in models])
    pred = np.mean(pred, axis=0)
    return pred

# setup
cfg = path_setup(Config)
seed_everything(cfg.seed)

# load data
train = pd.read_csv(os.path.join(cfg.INPUT, "train_cleaned.csv"))
train_embeded = pd.read_csv(
    os.path.join(
        cfg.INPUT,
        "train_roberta_embeded.csv"))
print("train features : {}".format(train_embeded.shape))

# preprocess target
train['jobflag'] -= 1

# training
models = fit_lsvb(train_embeded, train['jobflag'])

# preprocess target
test_embeded = pd.read_csv(os.path.join(cfg.INPUT, "test_roberta_embeded.csv"))
print("train features : {}".format(test_embeded.shape))

# inferring
pred = inference_lsvc(models, test_embeded)
print(pred)

sub = pd.read_csv(os.path.join(cfg.INPUT, 'submit_sample.csv'), header=None)
sub[1] = pred.astype(int) + 1
sub.to_csv(os.path.join(cfg.SUBMISSION, 'roberta_linear.csv'), index=False, header=False)
