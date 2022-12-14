#!/usr/bin/env python
# coding: utf-8
"""summery
------ Truncated SVD ------
fold0 : 0.675652501803397
fold1 : 0.6578448085654202
fold2 : 0.6974607020105883
fold3 : 0.6976881894873459
fold4 : 0.6908341304280391
oof score: 0.6838960664589581
------ LatentDirichletAllocation ------
fold0 : 0.12469135802469135
fold1 : 0.125
fold2 : 0.125
fold3 : 0.125
fold4 : 0.125
oof score: 0.12493827160493827
------ NMF ------
fold0 : 0.48054909451451755
fold1 : 0.43096159021387254
fold2 : 0.4531030806950219
fold3 : 0.49056385769035277
fold4 : 0.476780905752754
oof score: 0.46639170577330374
"""

from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD, NMF, LatentDirichletAllocation
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
    script = "linear/svc/decompositions"

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


def vectorize_svd(train: pd.DataFrame, test: pd.DataFrame):
    tfidf_svd = Pipeline(
        steps=[
            ("TfidfVectorizer",
             TfidfVectorizer()),
            ("TruncatedSVD",
             TruncatedSVD(
                 n_components=3000,
                 random_state=cfg.seed))])
    train = tfidf_svd.fit_transform(train['description'].pipe(hero.clean))
    test = tfidf_svd.fit_transform(test['description'].pipe(hero.clean))
    return pd.DataFrame(train), pd.DataFrame(test)


def vectorize_nmf(train: pd.DataFrame, test: pd.DataFrame):
    tfidf_svd = Pipeline(steps=[
        ("TfidfVectorizer", TfidfVectorizer()),
        ("NMF", NMF(n_components=3000, random_state=cfg.seed))
    ])
    train = tfidf_svd.fit_transform(train['description'].pipe(hero.clean))
    test = tfidf_svd.fit_transform(test['description'].pipe(hero.clean))
    return pd.DataFrame(train), pd.DataFrame(test)


def vectorize_lda(train: pd.DataFrame, test: pd.DataFrame):
    tfidf_svd = Pipeline(
        steps=[
            ("TfidfVectorizer",
             TfidfVectorizer()),
            ("LatentDirichletAllocation",
             LatentDirichletAllocation(
                 n_components=3000,
                 random_state=cfg.seed))])
    train = tfidf_svd.fit_transform(train['description'].pipe(hero.clean))
    test = tfidf_svd.fit_transform(test['description'].pipe(hero.clean))
    return pd.DataFrame(train), pd.DataFrame(test)


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


cfg = path_setup(Config)
seed_everything(Config.seed)

# load data
train = pd.read_csv(os.path.join(cfg.INPUT, 'train.csv'))
test = pd.read_csv(os.path.join(cfg.INPUT, 'test.csv'))
sub = pd.read_csv(os.path.join(cfg.INPUT, 'submit_sample.csv'), header=None)

# preprocess target
train['jobflag'] -= 1

print("------ Truncated SVD ------")
train_feat, test_feat = vectorize_svd(train, test)
models = fit_lsvb(train_feat, train['jobflag'])

print("------ LatentDirichletAllocation ------")
train_feat, test_feat = vectorize_lda(train, test)
models = fit_lsvb(train_feat, train['jobflag'])

print("------ NMF ------")
train_feat, test_feat = vectorize_nmf(train, test)
models = fit_lsvb(train_feat, train['jobflag'])
