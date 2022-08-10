#!/usr/bin/env python
# coding: utf-8

# ### Basic configuration


from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import texthero as hero
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import ntpath
from subprocess import PIPE
import subprocess
from glob import glob
import random
import warnings
import gc
import os
import seaborn as sns
from tqdm.auto import tqdm


class Config:
    notebook = "Linear/Baseline"
    script = "linear/baseline"

    n_splits = 5
    seed = 42

    # Reka Env
    dir_path = "/home/abe/kaggle/signate-sc2022"


# ### Import basic libraries


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


def vectorize(train: pd.DataFrame, test: pd.DataFrame):

    train['tfidf'] = train['description'].pipe(hero.clean).pipe(hero.tfidf)
    test['tfidf'] = test['description'].pipe(hero.clean).pipe(hero.tfidf)
    return train, test


train, test = vectorize(train, test)


assert "tfidf" in train.columns, "tfidf does not exist in train."
assert "tfidf" in test.columns, "tfidf does not exist in test."
# TODO : try GPLVM
train['pca'] = train['tfidf'].pipe(hero.pca)
test['pca'] = test['tfidf'].pipe(hero.pca)

hero.scatterplot(train, 'pca', color='jobflag', title="PCA Description")


train['labels'] = train['tfidf'].pipe(hero.kmeans, n_clusters=4).astype(str)
hero.scatterplot(train, 'pca', color='labels', title="K-means Description")


def decompose(train: pd.DataFrame, test: pd.DataFrame):
    tfidf_svd = Pipeline(steps=[
        ("TfidfVectorizer", TfidfVectorizer()),
        ("TruncatedSVD", TruncatedSVD(n_components=50, random_state=42))
    ])
    train = tfidf_svd.fit_transform(train['description'].pipe(hero.clean))
    test = tfidf_svd.fit_transform(test['description'].pipe(hero.clean))
    return pd.DataFrame(train), pd.DataFrame(test)


train_feat, test_feat = decompose(train, test)


# ## LinearSVM


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
            verbose=True)

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


models = fit_lsvb(train_feat, train['jobflag'])
