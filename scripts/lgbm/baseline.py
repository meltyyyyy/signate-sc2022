#!/usr/bin/env python
# coding: utf-8


from tabnanny import verbose
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import optuna.integration.lightgbm as lgb
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
    notebook = "Linear/Baseline"
    script = "lgbm/baseline"

    n_splits = 4
    n_classes = 4
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
    tfidf_svd = Pipeline(steps=[
        ("TfidfVectorizer", TfidfVectorizer()),
        ("TruncatedSVD", TruncatedSVD(n_components=3000, random_state=42))
    ])
    train = tfidf_svd.fit_transform(train['description'])
    test = tfidf_svd.fit_transform(test['description'])
    return pd.DataFrame(train), pd.DataFrame(test)


def plot_lgbm(fold, loss_train, loss_valid):
    plt.xlabel("Iteration")
    plt.ylabel("fscore")

    plt.plot(loss_train, label="train loss")
    plt.plot(loss_valid, label="valid loss")
    plt.legend()
    plt.savefig(os.path.join(cfg.EXP_FIG, f"fold{fold}.png"))


def fit_lgbm(X, y, params):
    models = []
    scores = []
    best_params, tuning_history = dict(), list()

    skf = StratifiedKFold(
        n_splits=cfg.n_splits,
        shuffle=True,
        random_state=cfg.seed)

    for fold, (trn_index, val_index) in enumerate(skf.split(X, y)):
        X_train, y_train = X.iloc[trn_index], y.iloc[trn_index]
        X_valid, y_valid = X.iloc[val_index], y.iloc[val_index]

        dtrain = lgb.Dataset(X_train, label=y_train)
        dvalid = lgb.Dataset(X_valid, label=y_valid)

        model = lgb.train(params,
                          dtrain,
                          valid_sets=[dtrain, dvalid],
                          valid_names=['train', 'valid'],
                          num_boost_round=100,
                          early_stopping_rounds=50,
                          verbosity=-1,
                          verbose_eval=-1,)

        print(model.params)
        # --------- prediction ---------
        pred = model.predict(X_valid, num_iteration=model.best_iteration)
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
train = pd.read_csv(os.path.join(cfg.INPUT, 'train_cleaned.csv'))
test = pd.read_csv(os.path.join(cfg.INPUT, 'test_cleaned.csv'))
sub = pd.read_csv(os.path.join(cfg.INPUT, 'submit_sample.csv'), header=None)

# preprocess target
train['jobflag'] -= 1

train_feat, test_feat = vectorize(train, test)

params = {
    "objective": "multiclass",
    "num_class": cfg.n_classes,
    "metric": "multi_logloss",
    "boosting": "gbdt",
    "learning_rate": 0.05,
    "num_leaves": 31,
    "force_col_wise": True,
    "seed": cfg.seed,
}

models = fit_lgbm(train_feat, train['jobflag'], params)
