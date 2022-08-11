#!/usr/bin/env python
# coding: utf-8
"""summery
fold0 : 0.5572658102598222
fold1 : 0.5541388869891285
fold2 : 0.5943871785119128
fold3 : 0.5680619869791236
fold4 : 0.5652159570724842
oof score: 0.5678139639624943
"""

from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
import texthero as hero
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from glob import glob
import random
import warnings
import gc
import os
from tqdm.auto import tqdm
plt.style.use('seaborn-pastel')
warnings.filterwarnings('ignore')
tqdm.pandas()


class Config:
    script = "linear/naive_bayes/tfidf"

    n_splits = 5
    seed = 42

    # Reka Env
    dir_path = "/home/abe/kaggle/signate-sc2022"


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


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


def vectorize(train: pd.DataFrame, test: pd.DataFrame):

    vectorizer = TfidfVectorizer()

    train = vectorizer.fit_transform(train['description'].pipe(hero.clean)).toarray()
    test = vectorizer.fit_transform(test['description'].pipe(hero.clean)).toarray()
    return pd.DataFrame(train), pd.DataFrame(test)


def fit_nbayes(X, y):
    models = []
    scores = []

    skf = StratifiedKFold(
        n_splits=cfg.n_splits,
        shuffle=True,
        random_state=cfg.seed)

    for fold, (trn_index, val_index) in enumerate(skf.split(X, y)):
        X_train, y_train = X.iloc[trn_index], y.iloc[trn_index]
        X_valid, y_valid = X.iloc[val_index], y.iloc[val_index]

        model = MultinomialNB(alpha=1.0)

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


cfg = path_setup(Config)
seed_everything(Config.seed)

# load data
train = pd.read_csv(os.path.join(cfg.INPUT, 'train.csv'))
test = pd.read_csv(os.path.join(cfg.INPUT, 'test.csv'))
sub = pd.read_csv(os.path.join(cfg.INPUT, 'submit_sample.csv'), header=None)

# preprocess target
train['jobflag'] -= 1

train_feat, test_feat = vectorize(train, test)
models = fit_nbayes(train_feat, train['jobflag'])
