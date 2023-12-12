import pandas as pd
import numpy as np
import gc
import time
import json
import matplotlib.pyplot as plt
import os
import joblib
from tqdm.auto import tqdm
from sklearn import preprocessing
from functools import wraps
from math import pi, sqrt, exp
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, SubsetRandomSampler
from timm.scheduler import CosineLRScheduler
plt.style.use("ggplot")
import logging

def track_time(f):
    @wraps(f)
    def wrap(*args, **kw):
        ts = time.time()
        result = f(*args, **kw)
        te = time.time()
        print('func:%r took: %2.4f sec' % \
          (f.__name__, te-ts))
        return result
    return wrap

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

class ResidualBiGRU(nn.Module):
    def __init__(self, hidden_size, n_layers=1, bidir=True):
        super(ResidualBiGRU, self).__init__()

        self.hidden_size = hidden_size
        self.n_layers = n_layers

        self.gru = nn.GRU(
            hidden_size,
            hidden_size,
            n_layers,
            batch_first=True,
            bidirectional=bidir,
        )
        dir_factor = 2 if bidir else 1
        self.fc1 = nn.Linear(
            hidden_size * dir_factor, hidden_size * dir_factor * 2
        )
        self.ln1 = nn.LayerNorm(hidden_size * dir_factor * 2)
        self.fc2 = nn.Linear(hidden_size * dir_factor * 2, hidden_size)
        self.ln2 = nn.LayerNorm(hidden_size)

    def forward(self, x, h=None):
        res, new_h = self.gru(x, h)

        res = self.fc1(res)
        res = self.ln1(res)
        res = nn.functional.relu(res)
        
        res = self.fc2(res)
        res = self.ln2(res)
        res = nn.functional.relu(res)

        res = res + x

        return res, new_h

class MultiResidualBiGRU(nn.Module):
    def __init__(self, input_size, hidden_size, out_size, n_layers, kernel_size=3, bidir=True):
        super(MultiResidualBiGRU, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.out_size = out_size
        self.n_layers = n_layers

        self.conv1d = nn.Conv1d(in_channels=input_size, out_channels=(hidden_size - input_size) // 2, kernel_size=kernel_size, padding=kernel_size // 2)
        self.conv_gn = nn.GroupNorm(1, (hidden_size - input_size) // 2)

        self.conv1d_5 = nn.Conv1d(in_channels=input_size, out_channels=(hidden_size - input_size) // 2, kernel_size=5, padding=2)
        self.conv1d_gn_5 = nn.GroupNorm(1, (hidden_size - input_size) // 2)

        self.res_bigrus = nn.ModuleList([
            ResidualBiGRU(hidden_size, n_layers=1, bidir=bidir) for _ in range(n_layers)
        ])
        
        self.fc_out = nn.Linear(hidden_size, out_size)

    def forward(self, x, h=None):
        x = x.transpose(1, 2)

        x1 = self.conv1d(x)
        x1 = self.conv_gn(x1)
        x1 = nn.functional.relu(x1)

        x2 = self.conv1d_5(x)
        x2 = self.conv1d_gn_5(x2)
        x2 = nn.functional.relu(x2)
        
        x = torch.cat((x, x1, x2), dim=1)
        x = x.transpose(1, 2)
        
        if h is None:
            h = [None for _ in range(self.n_layers)]

        new_h = []
        for i, res_bigru in enumerate(self.res_bigrus):
            x, new_hi = res_bigru(x, h[i])
            new_h.append(new_hi)

        x = self.fc_out(x)
        return x, new_h

class DataParser:
    def __init__(self, data_dir: str = "../data/", columns_to_scale = ['anglez', 'enmo']) -> None:
        self.data_dir = data_dir
        self.scaler = preprocessing.MinMaxScaler()
        self.columns_to_scale = columns_to_scale
    @track_time
    def _clean(self, df: pd.DataFrame) -> pd.DataFrame:
        if "step" in df.columns and "timestamp" in df.columns:
            df = df.dropna(subset=["step", "timestamp"])
            return df
        else:
            raise KeyError("Missing columns: either `step` or `timestamp` not exist.")

    @track_time
    def _transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if "night" in df.columns:
            df["night"] = df["night"].astype(np.int16)

        if "step" in df.columns and "timestamp" in df.columns:
            df["step"] = df["step"].astype(np.int32)
            df["timestamp"] = pd.to_datetime(df["timestamp"], format="%Y-%m-%dT%H:%M:%S%z", utc=True)

        if "anglez" and "enmo" in df.columns:
            normalized_data = self.scaler.fit_transform(df[self.columns_to_scale])
            df['anglez_norm'] = normalized_data[:, 0]
            df['enmo_norm'] = normalized_data[:, 1]
            
        df['hour'] = df['timestamp'].dt.hour

        return df

    def load_data(self, file_name: str, file_type: str) -> pd.DataFrame:
        if file_type == "parquet":
            df = pd.read_parquet(os.path.join(self.data_dir, file_name))
            df = self._clean(df)
            df = self._transform(df)
        else:
            df = pd.read_csv(os.path.join(self.data_dir, file_name))
            df = self._clean(df)
            
        return df

SIGMA = 720
SAMPLE_FREQ = 12

class SleepDataset(Dataset):
    def __init__(
        self,
        series_ids,
        series
    ):
        series_ids = series_ids
        series = series.reset_index()
        self.X = []
        self.y = joblib.load('./y.pkl')

        for index, viz_id in tqdm(enumerate(series_ids)):
            X = series.loc[(series.series_id==viz_id)].copy().reset_index()[["anglez_norm", "enmo_norm", "hour"]]
            X_anglez = self.downsample_seq_generate_features(X.values[:, 0], SAMPLE_FREQ, std_only=True)
            X_enmo   = self.downsample_seq_generate_features(X.values[:, 1], SAMPLE_FREQ)
            X_hour   = self.downsample_seq_generate_features(X.values[:, 2], SAMPLE_FREQ, is_hour=True)
            X = np.concatenate([X_anglez, X_enmo, X_hour], -1)
            X = torch.from_numpy(X)
            
            self.X.append(X)
            del X
            gc.collect()


    def downsample_seq_generate_features(self, feat, downsample_factor=SAMPLE_FREQ, std_only=False, is_hour=False):
        if len(feat) % downsample_factor != 0:
            feat = np.concatenate([feat, np.zeros(downsample_factor-((len(feat))%downsample_factor))+feat[-1]])

        feat = np.reshape(feat, (-1, downsample_factor))
        
        if is_hour:
            feat_hour = np.max(feat, 1)
            hour_sin = np.sin(feat_hour * (2 * np.pi / 24))
            hour_cos = np.cos(feat_hour * (2 * np.pi / 24))
            return np.dstack([hour_sin, hour_cos])[0]

        feat_mean   = np.mean(feat,1)
        feat_std    = np.std(feat,1)
        feat_median = np.median(feat,1)
        feat_max    = np.max(feat,1)
        feat_min    = np.min(feat,1)

        if std_only:
            return np.dstack([feat_std])[0]
    
        return np.dstack([feat_mean, feat_std, feat_median, feat_max, feat_min])[0]
    
    def downsample_seq(self,feat, downsample_factor = SAMPLE_FREQ):
        if len(feat)%SAMPLE_FREQ!=0:
            feat = np.concatenate([feat,np.zeros(SAMPLE_FREQ-((len(feat))%SAMPLE_FREQ))+feat[-1]])
        feat = np.reshape(feat, (-1,SAMPLE_FREQ))
        feat_mean = np.mean(feat,1)
        return feat_mean

    def gauss(self,n=SIGMA,sigma=SIGMA*0.15):
        r = range(-int(n/2),int(n/2)+1)
        return [1 / (sigma * sqrt(2*pi)) * exp(-float(x)**2/(2*sigma**2)) for x in r]

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        return self.X[index], self.y[index]

if __name__ == "__main__":
    model_path = os.path.join('.', 'model')
    os.makedirs(model_path, exist_ok=True)
    logging.basicConfig(filename='./model/bigru.log', level=logging.INFO)
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    train_ds = joblib.load("./train_ds.pkl")
    EPOCHS = 10
    WARMUP_PROP = 0.2
    BS = 1
    NUM_WORKERS = 1
    TRAIN_PROP = 0.9    
    max_chunk_size = 150000
    train_ds = joblib.load('./train_ds.pkl')
    train_size = int(TRAIN_PROP * len(train_ds))
    valid_size = len(train_ds) - train_size
    indices = torch.randperm(len(train_ds))
    train_sampler = SubsetRandomSampler(indices[:train_size])
    valid_sampler = SubsetRandomSampler(
        indices[train_size : train_size + valid_size]
    )
    steps = train_size*EPOCHS
    warmup_steps = int(steps*WARMUP_PROP)
    model = MultiResidualBiGRU(input_size=8, hidden_size=64, out_size=2, n_layers=5).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5,weight_decay = 0)
    scheduler = CosineLRScheduler(optimizer,t_initial= steps,warmup_t=warmup_steps, warmup_lr_init=1e-6,lr_min=2e-8,)
    dt = time.time()
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
        num_workers=NUM_WORKERS,
    )
    valid_loader = DataLoader(
        train_ds,
        batch_size=BS,
        sampler=valid_sampler,
        pin_memory=True,
        num_workers=NUM_WORKERS,
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
            pred = torch.zeros(y_batch.shape).to(device, non_blocking=True)
            optimizer.zero_grad()
            scheduler.step(step+train_size*epoch)
            h = None
            seq_len = X_batch.shape[1]

            for i in range(0, seq_len, max_chunk_size):
                X_chunk = X_batch[:, i : min(i + max_chunk_size, seq_len)].float()
                X_chunk = X_chunk.to(device, non_blocking=True)
                y_pred, h = model(X_chunk, h)
                h = [hi.detach() for hi in h]
                pred[:, i : min(i + max_chunk_size, seq_len)] = y_pred
                del X_chunk, y_pred

            loss = criterion(
                pred.float(),
                y_batch.float(),
            )
            loss.backward()
            train_loss += loss.item()
            n_tot_chunks += 1
            pbar.set_description(f'Training: loss = {(train_loss/n_tot_chunks):.2f}')

            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1e-1)
            optimizer.step()

            del pred, loss, X_batch, y_batch, h
            gc.collect()
        train_loss /= len(train_loader)
        del pbar
        gc.collect()

        if epoch % 1 == 0:
            model.eval()
            valid_loss = 0.0
            
            for X_batch, y_batch in tqdm(valid_loader, desc="Eval", unit="batch"):
                X_batch = X_batch.to(device, dtype=torch.float, non_blocking=True)
                y_batch = y_batch.to(device, dtype=torch.float, non_blocking=True)
                pred = torch.zeros(y_batch.shape).to(device, dtype=torch.float, non_blocking=True)
                h = None

                for i in range(0, seq_len, max_chunk_size):
                    X_chunk = X_batch[:, i : min(i + max_chunk_size, seq_len)].float().to(device, non_blocking=True)
                    y_pred, h = model(X_chunk, h)
                    h = [hi.detach() for hi in h]
                    pred[:, i : min(i + max_chunk_size, seq_len)] = y_pred
                    del X_chunk
                loss = criterion(
                    pred.float(),
                    y_batch.float(),
                )
                valid_loss += loss.item()
                del pred, loss, X_batch, y_batch
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