#!/usr/bin/env python
# coding: utf-8

# ### Basic configuration


from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from torch.cuda.amp import autocast, GradScaler
from transformers import AdamW, get_linear_schedule_with_warmup
from transformers import AutoConfig, AutoModel, AutoTokenizer
from torch.utils.data import Dataset, DataLoader
import transformers
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
import torch.nn as nn
import torch


class Config:
    notebook = "RoBERTa/Baseline"
    script = "roberta/baseline"
    model = "roberta-base"

    n_splits = 4
    batch_size = 16
    trn_fold = [0, 1, 2, 3]
    # max length of token
    max_len = 128
    lr = 2e-5

    # optimizer settings
    weight_decay = 2e-5
    beta = (0.9, 0.98)
    num_warmup_steps_rate = 0.01
    clip_grad_norm = None
    n_epochs = 10
    gradient_accumulation_steps = 1
    num_eval = 1

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


plt.style.use('seaborn-pastel')
sns.set_palette("winter_r")
warnings.filterwarnings('ignore')
tqdm.pandas()


# ### Seeding


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


seed_everything(Config.seed)


# ### Path configuration


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

    return cfg


cfg = path_setup(Config)


# # Define dataset


class BERTDataset(Dataset):
    def __init__(self, cfg, texts, labels=None):
        self.cfg = cfg
        self.texts = texts
        self.labels = labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):
        inputs = self.prepare_input(self.cfg, self.texts[index])
        if self.labels is not None:
            label = torch.tensor(self.labels[index], dtype=torch.int64)
            return inputs, label
        else:
            return inputs

    @staticmethod
    def prepare_input(cfg, text):
        inputs = cfg.tokenizer(
            text,
            add_special_tokens=True,
            max_length=cfg.max_len,
            padding="max_length",
            truncation=True,
            return_offsets_mapping=False,
        )
        for k, v in inputs.items():
            inputs[k] = torch.tensor(v, dtype=torch.long)
        return inputs


# ## Define model


class BERTModel(nn.Module):
    def __init__(self, cfg, criterion=None):
        super().__init__()
        self.cfg = cfg
        self.criterion = criterion
        self.config = AutoConfig.from_pretrained(
            cfg.model,
            output_hidden_states=True
        )
        self.backbone = AutoModel.from_pretrained(
            cfg.model,
            config=self.config
        )
        self.fc = nn.Sequential(
            nn.Linear(self.config.hidden_size, 4),
        )

    def forward(self, inputs, labels=None):
        outputs = self.backbone(**inputs)["last_hidden_state"]
        outputs = outputs[:, 0, :]
        if labels is not None:
            logits = self.fc(outputs)
            loss = self.criterion(logits, labels)
            return logits, loss
        else:
            logits = self.fc(outputs)
            return logits


# ## Training


# KFold
def get_stratifiedkfold(train, target_col, n_splits, seed):
    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    generator = kf.split(train, train[target_col])
    fold_series = []
    for fold, (idx_train, idx_valid) in enumerate(generator):
        fold_series.append(pd.Series(fold, index=idx_valid))
    fold_series = pd.concat(fold_series).sort_index()
    return fold_series

# collatte


def collatte(inputs, labels=None):
    mask_len = int(inputs["attention_mask"].sum(axis=1).max())
    if labels is not None:
        inputs = {
            "input_ids": inputs['input_ids'][:, :mask_len],
            "attention_mask": inputs['attention_mask'][:, :mask_len],
        }
        labels = labels[:, :mask_len]
        return inputs, labels, mask_len

    else:
        inputs = {
            "input_ids": inputs['input_ids'][:, :mask_len],
            "attention_mask": inputs['attention_mask'][:, :mask_len],
        }
        return inputs, mask_len


def training(cfg, train):
    # =====================
    # Training
    # =====================
    oof_pred = np.zeros((len(train), 4), dtype=np.float32)

    # 損失関数
    criterion = nn.CrossEntropyLoss()

    for fold in cfg.trn_fold:
        # Dataset,Dataloaderの設定
        train_df = train.loc[cfg.folds != fold]
        valid_df = train.loc[cfg.folds == fold]
        train_idx = list(train_df.index)
        valid_idx = list(valid_df.index)

        train_dataset = BERTDataset(
            cfg,
            train_df['description'].to_numpy(),
            train_df['jobflag'].to_numpy(),
        )
        valid_dataset = BERTDataset(
            cfg,
            valid_df['description'].to_numpy(),
            valid_df['jobflag'].to_numpy()
        )
        train_loader = DataLoader(
            dataset=train_dataset,
            batch_size=cfg.batch_size,
            shuffle=True,
            pin_memory=True,
            drop_last=True
        )
        valid_loader = DataLoader(
            dataset=valid_dataset,
            batch_size=cfg.batch_size,
            shuffle=False,
            pin_memory=True,
            drop_last=False
        )

        # 初期化
        best_val_preds = None
        best_val_score = -1

        # modelの読み込み
        model = BERTModel(cfg, criterion)
        model = model.to(cfg.device)

        # optimizer，schedulerの設定
        param_optimizer = list(model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = []
        optimizer_grouped_parameters.append({
            'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
            'weight_decay': cfg.weight_decay
        })
        optimizer_grouped_parameters.append({
            'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
            'weight_decay': 0.0
        })
        optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=cfg.lr,
            betas=cfg.beta,
            weight_decay=cfg.weight_decay,
        )
        num_train_optimization_steps = int(
            len(train_loader) * cfg.n_epochs // cfg.gradient_accumulation_steps
        )
        num_warmup_steps = int(
            num_train_optimization_steps *
            cfg.num_warmup_steps_rate)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_train_optimization_steps
        )
        num_eval_step = len(train_loader) // cfg.num_eval + cfg.num_eval

        for epoch in range(cfg.n_epochs):
            # training
            print(f"# ============ start epoch:{epoch} ============== #")
            model.train()
            val_losses_batch = []
            scaler = GradScaler()
            with tqdm(train_loader, total=len(train_loader)) as pbar:
                for step, (inputs, labels) in enumerate(pbar):
                    inputs, max_len = collatte(inputs)
                    for k, v in inputs.items():
                        inputs[k] = v.to(cfg.device)
                    labels = labels.to(cfg.device)

                    optimizer.zero_grad()
                    with autocast():
                        output, loss = model(inputs, labels)
                    pbar.set_postfix({
                        'loss': loss.item(),
                        'lr': scheduler.get_lr()[0]
                    })

                    if cfg.gradient_accumulation_steps > 1:
                        loss = loss / cfg.gradient_accumulation_steps
                    scaler.scale(loss).backward()
                    if cfg.clip_grad_norm is not None:
                        torch.nn.utils.clip_grad_norm_(
                            model.parameters(),
                            cfg.clip_grad_norm
                        )
                    if (step + 1) % cfg.gradient_accumulation_steps == 0:
                        scaler.step(optimizer)
                        scaler.update()
                        scheduler.step()

            # evaluating
            val_preds = []
            val_losses = []
            val_nums = []
            model.eval()
            with torch.no_grad():
                with tqdm(valid_loader, total=len(valid_loader)) as pbar:
                    for (inputs, labels) in pbar:
                        inputs, max_len = collatte(inputs)
                        for k, v in inputs.items():
                            inputs[k] = v.to(cfg.device)
                        labels = labels.to(cfg.device)
                        with autocast():
                            output, loss = model(inputs, labels)
                        output = output.sigmoid().detach().cpu().numpy()
                        val_preds.append(output)
                        val_losses.append(loss.item() * len(labels))
                        val_nums.append(len(labels))
                        pbar.set_postfix({
                            'val_loss': loss.item()
                        })

            val_preds = np.concatenate(val_preds)
            val_loss = sum(val_losses) / sum(val_nums)
            score = f1_score(
                np.argmax(
                    val_preds,
                    axis=1),
                valid_df['jobflag'],
                average='macro')
            val_log = {
                'val_loss': val_loss,
                'score': score,
            }
            print(val_log)
            if best_val_score < score:
                print("save model weight")
                best_val_preds = val_preds
                best_val_score = score
                torch.save(
                    model.state_dict(),
                    os.path.join(cfg.EXP_MODEL, f"fold{fold}.pth")
                )

        oof_pred[valid_idx] = best_val_preds.astype(np.float32)
        np.save(
            os.path.join(
                cfg.EXP_PREDS,
                f'oof_pred_fold{fold}.npy'),
            best_val_preds)
        del model
        gc.collect()

    # scoring
    np.save(os.path.join(cfg.EXP_PREDS, 'oof_pred.npy'), oof_pred)
    score = f1_score(
        np.argmax(
            oof_pred,
            axis=1),
        train['jobflag'],
        average='macro')
    print('CV:', round(score, 5))
    return score


def inferring(cfg, test):
    print('\n'.join(cfg.model_weights))
    sub_pred = np.zeros((len(test), 4), dtype=np.float32)
    for fold, model_weight in enumerate(cfg.model_weights):
        # dataset, dataloader
        test_dataset = BERTDataset(
            cfg,
            test['description'].to_numpy()
        )
        test_loader = DataLoader(
            dataset=test_dataset,
            batch_size=cfg.batch_size,
            shuffle=False,
            pin_memory=True
        )
        model = BERTModel(cfg)
        model.load_state_dict(torch.load(model_weight))
        model = model.to(cfg.device)

        model.eval()
        fold_pred = []
        with torch.no_grad():
            for inputs in tqdm(test_loader, total=len(test_loader)):
                inputs, max_len = collatte(inputs)
                for k, v in inputs.items():
                    inputs[k] = v.to(cfg.device)
                with autocast():
                    output = model(inputs)
                output = output.softmax(axis=1).detach().cpu().numpy()
                fold_pred.append(output)
        fold_pred = np.concatenate(fold_pred)
        np.save(
            os.path.join(
                cfg.EXP_PREDS,
                f'sub_pred_fold{fold}.npy'),
            fold_pred)
        sub_pred += fold_pred / len(cfg.model_weights)
        del model
        gc.collect()
    np.save(os.path.join(cfg.EXP_PREDS, f'sub_pred.npy'), sub_pred)
    return sub_pred


# load data
train = pd.read_csv(os.path.join(cfg.INPUT, 'train.csv'))

# preprocess target
train['jobflag'] -= 1

# load tokenizer
cfg.tokenizer = AutoTokenizer.from_pretrained(cfg.model)
# create folds
cfg.folds = get_stratifiedkfold(train, 'jobflag', cfg.n_splits, cfg.seed)
cfg.folds.to_csv(os.path.join(cfg.EXP_PREDS, 'folds.csv'))
# train BERT
score = training(cfg, train)


test = pd.read_csv(os.path.join(cfg.INPUT, 'test.csv'))
sub = pd.read_csv(os.path.join(cfg.INPUT, 'submit_sample.csv'), header=None)
# BERTの推論
cfg.model_weights = [
    p for p in sorted(
        glob(
            os.path.join(
                cfg.EXP_MODEL,
                'fold*.pth')))]
sub_pred = inferring(cfg, test)
sub[1] = np.argmax(sub_pred, axis=1)
sub[1] = sub[1].astype(int) + 1

sub.to_csv(
    os.path.join(
        cfg.SUBMISSION,
        'submission.csv'),
    index=False,
    header=False)
