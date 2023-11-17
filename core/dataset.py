import joblib
import torch
from torch.utils.data import Dataset, DataLoader


class MachoDataset(Dataset):
    def __init__(self, data_root, prediction_length, mode='train'):
        data = joblib.load(data_root + f'{mode}.pkl')
        self.prediction_length = prediction_length

        self.times = data[0][:, 0, :]
        self.values = data[0][:, 1, :]
        self.aux = data[1]
        self.labels = data[2]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        past_times = torch.tensor(self.times[idx, :-self.prediction_length], dtype=torch.float)
        future_times = torch.tensor(self.times[idx, -self.prediction_length:], dtype=torch.float)
        past_values = torch.tensor(self.values[idx, :-self.prediction_length], dtype=torch.float)
        future_values = torch.tensor(self.values[idx, -self.prediction_length:], dtype=torch.float)
        past_mask = torch.ones(past_times.shape, dtype=torch.float)
        future_mask = torch.ones(future_times.shape, dtype=torch.float)
        labels = torch.tensor(self.labels[idx], dtype=torch.long)

        past_times = past_times.unsqueeze(-1)
        future_times = future_times.unsqueeze(-1)

        return past_times, future_times, past_values, future_values, past_mask, future_mask, labels
