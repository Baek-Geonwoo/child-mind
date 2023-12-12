import numpy as np
import gc
import time
import json
import matplotlib.pyplot as plt
import os
import joblib
import math
from tqdm.auto import tqdm 
from math import pi, sqrt, exp
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, SubsetRandomSampler
from timm.scheduler import CosineLRScheduler
plt.style.use("ggplot")
import logging

def normalize(y):
    mean = y[:,0].mean().item()
    std = y[:,0].std().item()
    y[:,0] = (y[:,0]-mean)/(std+1e-16)
    mean = y[:,1].mean().item()
    std = y[:,1].std().item()
    y[:,1] = (y[:,1]-mean)/(std+1e-16)
    return y

def plot_history(history, model_path=".", show=True):
    epochs = range(1, len(history["train_loss"]) + 1)

    plt.figure()
    plt.plot(epochs, history["train_loss"], label="Training Loss")
    plt.plot(epochs, history["valid_loss"], label="Validation Loss")
    plt.title("Loss evolution")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(model_path, "loss_evo.png"))
    if show:
        plt.show()
    plt.close()

    plt.figure()
    plt.plot(epochs, history["lr"])
    plt.title("Learning Rate evolution")
    plt.xlabel("Epochs")
    plt.ylabel("LR")
    plt.savefig(os.path.join(model_path, "lr_evo.png"))
    if show:
        plt.show()
    plt.close()

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.3, max_len=24*60):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class Transformer(nn.Module):
    def __init__(self, input_size, hidden_size, out_size, n_layers):
        super(Transformer, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.out_size = out_size
        self.n_layers = n_layers

        self.fc_in = nn.Linear(input_size, hidden_size)
        self.ln = nn.LayerNorm(hidden_size)
        
        encoder_layers = nn.TransformerEncoderLayer(hidden_size, 8)
        encoder_layers.self_attn.batch_first = True
        
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, n_layers)
        
        self.fc_out = nn.Linear(hidden_size, out_size)
        self.pos_encoder = PositionalEncoding(hidden_size)

    def forward(self, x):
        x = self.fc_in(x)
        x = self.ln(x)
        x = nn.functional.relu(x)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        x = self.fc_out(x)
        return x.float()

SIGMA = 720
SAMPLE_FREQ = 12
class SleepDataset(Dataset):
    def __init__(
        self,
        file
    ):
        self.targets,self.data,self.ids = joblib.load(file)
        self.X = []
        self.y = []
        
        for index in tqdm(range(len(self.targets))):
            X = self.data[index][['anglez', 'enmo']]
            y = self.targets[index]

            target_guassian = np.zeros((len(X), 2))
            for s, e in y:
                st1, st2 = max(0, s - SIGMA // 2), s + SIGMA // 2 + 1
                ed1, ed2 = e - SIGMA // 2, min(len(X), e + SIGMA // 2 + 1)
                target_guassian[st1:st2, 0] = self.gauss()[st1 - (s - SIGMA // 2):]
                target_guassian[ed1:ed2, 1] = self.gauss()[:SIGMA + 1 - ((e + SIGMA // 2 + 1) - ed2)]
            X = np.concatenate([self.downsample_seq_generate_features(X.values[:, i], SAMPLE_FREQ) for i in range(X.shape[1])], -1)
            gc.collect()
            y = np.dstack([self.downsample_seq(target_guassian[:, i], SAMPLE_FREQ) for i in range(target_guassian.shape[1])])[0]
            gc.collect()

            self.X.append(X)
            self.y.append(y)
            gc.collect()
            
    def downsample_seq_generate_features(self,feat, downsample_factor = SAMPLE_FREQ):
        if len(feat)%SAMPLE_FREQ==0:
            feat = np.concatenate([feat,np.zeros(SAMPLE_FREQ-((len(feat))%SAMPLE_FREQ))+feat[-1]])
        feat = np.reshape(feat, (-1,SAMPLE_FREQ))
        feat_mean = np.mean(feat,1)
        feat_std = np.std(feat,1)
        feat_median = np.median(feat,1)
        feat_max = np.max(feat,1)
        feat_min = np.min(feat,1)

        return np.dstack([feat_mean,feat_std,feat_median,feat_max,feat_min])[0]
    def downsample_seq(self,feat, downsample_factor = SAMPLE_FREQ):
        if len(feat)%SAMPLE_FREQ==0:
            feat = np.concatenate([feat,np.zeros(SAMPLE_FREQ-((len(feat))%SAMPLE_FREQ))+feat[-1]])
        feat = np.reshape(feat, (-1,SAMPLE_FREQ))
        feat_mean = np.mean(feat,1)
        return feat_mean
    
    def gauss(self,n=SIGMA,sigma=SIGMA*0.15):
        r = range(-int(n/2),int(n/2)+1)
        return [1 / (sigma * sqrt(2*pi)) * exp(-float(x)**2/(2*sigma**2)) for x in r]
    
    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        return self.X[index], self.y[index]

if __name__ == "__main__":
    logging.basicConfig(filename='./model/transformer.log', level=logging.INFO)
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    EPOCHS = 5
    WARMUP_PROP = 0.2
    BS = 1
    NUM_WORKERS = 1
    TRAIN_PROP = 0.9
    train_ds = joblib.load('../data/train_ds.pkl')
    train_size = int(TRAIN_PROP * len(train_ds))
    valid_size = len(train_ds) - train_size
    indices = torch.randperm(len(train_ds))
    train_sampler = SubsetRandomSampler(indices[:train_size])
    valid_sampler = SubsetRandomSampler(
        indices[train_size : train_size + valid_size]
    )
    steps = train_size*EPOCHS
    warmup_steps = int(steps*WARMUP_PROP)
    model = Transformer(input_size=10,hidden_size=32,out_size=2,n_layers=1).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3,weight_decay = 0)
    scheduler = CosineLRScheduler(optimizer,t_initial= steps,warmup_t=warmup_steps, warmup_lr_init=1e-6,lr_min=2e-8,)
    dt = time.time()
    model_path = os.path.join('.', 'model')
    history = {
        "train_loss": [],
        "valid_loss": [],
        "valid_mAP": [],
        "lr": [],
    }
    best_valid_loss = np.inf
    criterion = torch.nn.MSELoss()
    train_loader = DataLoader(
        train_ds,
        batch_size=BS,
        sampler=train_sampler,
        pin_memory=True,
        num_workers=1,
    )

    valid_loader = DataLoader(
        train_ds,
        batch_size=BS,
        sampler=valid_sampler,
        pin_memory=True,
        num_workers=1,
    )
    for epoch in range(1, EPOCHS + 1):
        train_loss = 0.0
        n_tot_chunks = 0
        pbar = tqdm(
            train_loader, desc="Training", unit="batch"
        )
        model = model.to(device)
        model.train()
        for step, (X_batch, y_batch) in enumerate(pbar):
            y_batch = y_batch.to(device, dtype=torch.float, non_blocking=True)

            optimizer.zero_grad()
            scheduler.step(step+train_size*epoch)
            y_pred = model(X_batch.to(device, dtype=torch.float, non_blocking=True))
            loss = criterion(normalize(y_pred), y_batch)
            loss.backward()
            train_loss += loss.item()
            n_tot_chunks += 1
            pbar.set_description(f'Training: loss = {(train_loss/n_tot_chunks):.2f}')

            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1e-1)
            optimizer.step()

            del y_pred, loss, y_batch
            gc.collect()
        train_loss /= len(train_loader)
        del pbar
        gc.collect()
        
        if epoch % 1 == 0:
            model.eval()
            valid_loss = 0.0
            for X_batch, y_batch in tqdm(valid_loader, desc="Eval", unit="batch"):
                X_batch = X_batch.to(device, dtype=torch.float, non_blocking=True)
                y_batch = y_batch.to(dtype=torch.float, non_blocking=True)

                y_pred = model(X_batch).detach().cpu()
                loss = criterion(y_pred.float(), y_batch.float())
                valid_loss += loss.item()

                del y_pred, loss, y_batch, X_batch
                gc.collect()

            valid_loss /= len(valid_loader)

            history["train_loss"].append(train_loss)
            history["valid_loss"].append(valid_loss)
            history["lr"].append(optimizer.param_groups[0]["lr"])

            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                torch.save(
                    model.state_dict(),
                    os.path.join(model_path, f'model_best.pth')
                )

            dt = time.time() - dt
            logging.info(f'{epoch}/{EPOCHS} -- '
                + f'Train Loss: {train_loss:.6f}, '
                + f'Valid Loss: {valid_loss:.6f}, '
                + f'Time: {dt:.6f}s'
            )
            dt = time.time()
    plot_history(history, model_path=model_path)
    history_path = os.path.join(model_path, "history.json")
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=4)