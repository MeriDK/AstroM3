import joblib
import torch
import numpy as np
from torch.utils.data import Dataset


class MachoDataset(Dataset):
    def __init__(self, data_root, prediction_length, window_length=None, mode='train', use_errors=False, seed=42):
        data = joblib.load(data_root + f'{mode}.pkl')

        self.prediction_length = prediction_length
        self.use_errors = use_errors

        if use_errors and data[0].shape[1] != 3:
            raise Exception("use_errors was True but dataset does not contain errors. "
                            "Try running preprocess_data.py with the flag --use-error")

        if window_length and window_length > data[0].shape[2]:
            raise Exception("Cannot get a longer sequence that we have in our dataset "
                            f"Try to reduce `window_length` <= {data[0].shape[2]}")

        idx = list(np.arange(data[0].shape[2]))

        if window_length:
            rng = np.random.default_rng(seed)
            rng.shuffle(idx)
            idx = sorted(idx[:window_length])

        self.times = data[0][:, 0, idx]

        if use_errors:
            self.values = data[0][:, 1:, idx]
        else:
            self.values = data[0][:, 1, idx]

        self.aux = data[1]
        self.labels = data[2]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        past_times = torch.tensor(self.times[idx, :-self.prediction_length], dtype=torch.float)
        future_times = torch.tensor(self.times[idx, -self.prediction_length:], dtype=torch.float)

        if self.use_errors:
            past_values = torch.tensor(self.values[idx, :, :-self.prediction_length], dtype=torch.float).T
            future_values = torch.tensor(self.values[idx, :, -self.prediction_length:], dtype=torch.float).T
        else:
            past_values = torch.tensor(self.values[idx, :-self.prediction_length], dtype=torch.float)
            future_values = torch.tensor(self.values[idx, -self.prediction_length:], dtype=torch.float)

        past_mask = torch.ones(past_times.shape, dtype=torch.float)
        future_mask = torch.ones(future_times.shape, dtype=torch.float)
        aux = torch.tensor(self.aux[idx], dtype=torch.long)
        labels = torch.tensor(self.labels[idx], dtype=torch.long)

        past_times = past_times.unsqueeze(-1)
        future_times = future_times.unsqueeze(-1)

        return past_times, future_times, past_values, future_values, past_mask, future_mask, aux, labels
