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
    script = "roberta/baseline"

    n_splits = 5
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

    train = False
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


class BERTDataset(Dataset):
    def __init__(self, tokenizer, texts, labels=None):
        self.tokenizer = tokenizer
        self.texts = texts
        self.labels = labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):
        inputs = self.prepare_input(self.tokenizer, self.texts[index])
        if self.labels is not None:
            label = torch.tensor(self.labels[index], dtype=torch.int64)
            return inputs, label
        else:
            return inputs

    @staticmethod
    def prepare_input(tokenizer, text):
        inputs = tokenizer(
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
        outputs = self.backbone(**inputs)["last_hidden_state"]
        outputs = outputs[:, 0, :]
        if labels is not None:
            logits = self.fc(outputs)
            loss = self.criterion(logits, labels)
            return logits, loss
        else:
            logits = self.fc(outputs)
            return logits


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


def training(X, y, tokenizer, batch_size):
    # =====================
    # Training
    # =====================
    oof_pred = np.zeros((len(X), cfg.n_classes), dtype=np.float32)
    criterion = nn.CrossEntropyLoss()

    skf = StratifiedKFold(
        n_splits=cfg.n_splits,
        shuffle=True,
        random_state=cfg.seed)

    for fold, (trn_index, val_index) in enumerate(skf.split(X, y)):
        print("#" * 25)
        print(f"# fold : {fold}")
        print("#" * 25)

        X_train, y_train = X.iloc[trn_index], y.iloc[trn_index]
        X_valid, y_valid = X.iloc[val_index], y.iloc[val_index]
        train_idx = list(X_train.index)
        valid_idx = list(X_valid.index)

        train_dataset = BERTDataset(
            tokenizer,
            X_train.to_numpy(),
            y_train.to_numpy(),
        )
        valid_dataset = BERTDataset(
            tokenizer,
            X_valid.to_numpy(),
            y_valid.to_numpy()
        )
        train_loader = DataLoader(
            dataset=train_dataset,
            batch_size=batch_size,
            shuffle=True,
            pin_memory=True,
            drop_last=True
        )
        valid_loader = DataLoader(
            dataset=valid_dataset,
            batch_size=batch_size,
            shuffle=False,
            pin_memory=True,
            drop_last=False
        )

        # initialize
        best_val_preds = None
        best_val_score = -1

        # model
        model = BERTModel(cfg.model_name, criterion)
        model = model.to(cfg.device)

        # settings for optimizer， scheduler
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
                y_valid,
                average='macro')

            print(f"val_loss : {val_loss}, score : {score}")

            if best_val_score < score:
                best_val_preds = val_preds
                best_val_score = score
                torch.save(
                    model.state_dict(),
                    os.path.join(cfg.EXP_MODEL, f"fold{fold}.pth")
                )

        oof_pred[valid_idx] = best_val_preds.astype(np.float32)
        del model
        gc.collect()

    score = f1_score(np.argmax(oof_pred, axis=1), y, average='macro')
    print('CV:', round(score, 5))
    return score


def inferring(X, tokenizer):
    # =====================
    # Inferring
    # =====================
    print('\n'.join(cfg.model_weights))
    sub_pred = np.zeros((len(X), cfg.n_classes), dtype=np.float32)
    for fold, model_weight in enumerate(cfg.model_weights):
        # dataset, dataloader
        test_dataset = BERTDataset(
            tokenizer,
            X.to_numpy()
        )
        test_loader = DataLoader(
            dataset=test_dataset,
            batch_size=cfg.batch_size,
            shuffle=False,
            pin_memory=True
        )
        model = BERTModel()
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
        sub_pred += fold_pred / len(cfg.model_weights)
        del model
        gc.collect()
    return sub_pred


seed_everything(Config.seed)
cfg = path_setup(Config)

# load tokenizer
tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)

if cfg.train:
    print("#" * 25)
    print("# Training")
    print("#" * 25)

    # load data
    train = pd.read_csv(os.path.join(cfg.INPUT, 'train_cleaned.csv'))
    print(train['description'].head(10))
    # preprocess target
    train['jobflag'] -= 1
    # train BERT
    score = training(train['description'],
                     train['jobflag'],
                     tokenizer=tokenizer,
                     batch_size=cfg.batch_size)
else:
    print("#" * 25)
    print("# Inferring")
    print("#" * 25)

    test = pd.read_csv(os.path.join(cfg.INPUT, 'test_cleaned.csv'))
    sub = pd.read_csv(os.path.join(cfg.INPUT, 'submit_sample.csv'), header=None)
    print(test['description'].head(10))

    # BERTの推論
    cfg.model_weights = [
        p for p in sorted(
            glob(
                os.path.join(
                    cfg.EXP_MODEL,
                    'fold*.pth')))]
    preds = inferring(test['description'], tokenizer)

    sub[1] = np.argmax(preds, axis=1)
    sub[1] = sub[1].astype(int) + 1
    sub.to_csv(
        os.path.join(
            cfg.SUBMISSION,
            'submission.csv'),
        index=False,
        header=False)
