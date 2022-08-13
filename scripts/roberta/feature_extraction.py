#!/usr/bin/env python
# coding: utf-8


from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from torch.cuda.amp import autocast, GradScaler
from transformers import AdamW, get_linear_schedule_with_warmup
from transformers import AutoConfig, AutoModel, AutoTokenizer
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from glob import glob
import random
import warnings
import gc
import os
import seaborn as sns
from tqdm.auto import tqdm
import torch.nn as nn
import torch
plt.style.use('seaborn-pastel')
sns.set_palette("winter_r")
warnings.filterwarnings('ignore')
tqdm.pandas()


class Config:
    script = "roberta/feature_extraction"

    n_splits = 4
    seed = 42

    batch_size = 16
    n_classes = 4
    n_epochs = 10

    # bert
    model_name = "roberta-base"
    weight_decay = 2e-5
    beta = (0.9, 0.98)
    max_len = 128
    lr = 2e-5
    num_warmup_steps_rate = 0.01
    clip_grad_norm = None
    gradient_accumulation_steps = 1
    num_eval = 1

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Reka Env
    dir_path = "/home/abe/kaggle/signate-sc2022"


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def path_setup(cfg):
    cfg.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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


class BERTModel(nn.Module):
    def __init__(self, model_name="roberta-base", criterion=None):
        super().__init__()
        self.criterion = criterion
        self.config = AutoConfig.from_pretrained(
            model_name,
            output_hidden_states=True
        )
        self.backbone = AutoModel.from_pretrained(
            model_name,
            config=self.config
        )
        self.fc = nn.Sequential(
            nn.Linear(self.config.hidden_size, 4),
        )

    def forward(self, inputs, labels=None):
        outputs = self.backbone(**inputs)

        if labels is None:

            logits = self.fc(outputs["last_hidden_state"][:, 0, :])
            return logits, outputs

        outputs = outputs["last_hidden_state"][:, 0, :]
        logits = self.fc(outputs)
        loss = self.criterion(logits, labels)
        return logits, loss


class BertSequenceVectorizer:
    def __init__(self, model, tokenizer, max_len):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.tokenizer = tokenizer
        self.bert_model = model
        self.bert_model = self.bert_model.to(self.device)
        self.max_len = max_len

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
        _, bert_out = self.bert_model({"input_ids": inputs_tensor, "attention_mask": masks_tensor})
        seq_out, pooled_out = bert_out['last_hidden_state'], bert_out['pooler_output']

        if torch.cuda.is_available():
            return seq_out[0][0].cpu().detach().numpy()
        else:
            return seq_out[0][0].detach().numpy()


def vectorize(df: pd.DataFrame, tokenizer, model_paths) -> pd.DataFrame:
    assert "description" in df.columns
    print('\n'.join(model_paths))
    df['embeded'] = 0
    for _, model_weight in enumerate(model_paths):
        model = BERTModel()
        model.load_state_dict(torch.load(model_weight))
        model = model.to(cfg.device)

        BSV = BertSequenceVectorizer(model, tokenizer, cfg.max_len)
        df['embeded'] = df['description'].progress_apply(lambda x: BSV.vectorize(x))
        del model
        gc.collect()
    df['embeded'] = df['embeded'] / len(model_paths)
    return pd.DataFrame(np.stack(df['embeded']))


seed_everything(Config.seed)
cfg = path_setup(Config)

# load tokenizer
tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)

# trained RoBERTa model path
model_paths = [p for p in sorted(glob(os.path.join(cfg.dir_path + "/output/roberta/baseline/model/", "fold*.pth")))]

# process train data
train = pd.read_csv(os.path.join(cfg.INPUT, "train_cleaned.csv"))
print(train['description'].head(10))

feat_train = vectorize(train, tokenizer, model_paths)
print(feat_train.head())
print(feat_train.shape)

# process test data
test = pd.read_csv(os.path.join(cfg.INPUT, "test_cleaned.csv"))
print(test['description'].head(10))

feat_test = vectorize(test, tokenizer, model_paths)
print(feat_test.head())
print(feat_test.shape)

# save
feat_train.to_csv(os.path.join(cfg.INPUT, "train_roberta_embeded.csv"), index=False)
feat_test.to_csv(os.path.join(cfg.INPUT, "test_roberta_embeded.csv"), index=False)
