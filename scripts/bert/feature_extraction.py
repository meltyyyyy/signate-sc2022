#!/usr/bin/env python
# coding: utf-8

# Reference
# https://www.guruguru.science/competitions/16/discussions/fb792c87-6bad-445d-aa34-b4118fc378c1/


from tqdm import tqdm
from transformers import BertTokenizer
import transformers
from nltk.stem.porter import PorterStemmer
import re
from transformers import AdamW, AutoModel, AutoTokenizer
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
import torch
import texthero as hero
import seaborn as sns
import pandas as pd
import numpy as np
import nlp
import matplotlib.pyplot as plt
import random
import os
import warnings
tqdm.pandas()
warnings.filterwarnings('ignore')


class Config:
    script = "bert/bert/feature_extraction"

    n_splits = 5
    seed = 42

    trn_batch = 32
    val_batch = 128
    n_classes = 4
    epochs = 5

    # bert
    model = "bert-base-uncased"

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


def cleaning(df: pd.DataFrame):
    assert "description" in df.columns
    df['description'] = df['description'].pipe(remove_tag).pipe(hero.clean)
    return df


def remove_tag(x):
    p = re.compile(r"<[^>]*?>")
    return x.apply(lambda x: p.sub("", x))


class BertSequenceVectorizer:
    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model_name = cfg.model
        self.tokenizer = BertTokenizer.from_pretrained(self.model_name)
        self.bert_model = transformers.BertModel.from_pretrained(
            self.model_name)
        self.bert_model = self.bert_model.to(self.device)
        self.max_len = 128

    def vectorize(self, sentence: str) -> np.array:
        inp = self.tokenizer.encode(sentence)
        len_inp = len(inp)

        if len_inp >= self.max_len:
            inputs = inp[:self.max_len]
            masks = [1] * self.max_len
        else:
            inputs = inp + [0] * (self.max_len - len_inp)
            masks = [1] * len_inp + [0] * (self.max_len - len_inp)

        inputs_tensor = torch.tensor(
            [inputs], dtype=torch.long).to(
            self.device)
        masks_tensor = torch.tensor([masks], dtype=torch.long).to(self.device)

        bert_out = self.bert_model(inputs_tensor, masks_tensor)
        seq_out, pooled_out = bert_out['last_hidden_state'], bert_out['pooler_output']

        if torch.cuda.is_available():
            return seq_out.cpu().detach().numpy()
        else:
            return seq_out[0][0].detach().numpy()


def vectorize(df: pd.DataFrame):
    assert "description" in df.columns
    BSV = BertSequenceVectorizer()
    df['feature'] = df['description'].progress_apply(
        lambda x: BSV.vectorize(x))
    return pd.DataFrame(np.stack(df['feature']))


cfg = path_setup(Config)
seed_everything(cfg.seed)

if torch.cuda.is_available():
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    current_device = torch.cuda.current_device()
    print("Device:", torch.cuda.get_device_name(current_device))


train = pd.read_csv(os.path.join(cfg.INPUT, "train.csv"))
train = cleaning(train)
print(train['description'].head(10))


feat_train = vectorize(train)
print(feat_train.head())
print(feat_train.shape)
