import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import TimeSeriesTransformerConfig, TimeSeriesTransformerForPrediction

from dataset import MachoDataset
from trainer import PredictionTrainer, ClassificationTrainer
from model import ClassificationModel


def main():
    data_root = '/home/mrizhko/AstroML/contra_periodic/data/macho/'
    window_length = 200
    prediction_length = 1

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_dataset = MachoDataset(data_root, prediction_length, mode='train')
    val_dataset = MachoDataset(data_root, prediction_length, mode='val')
    test_dataset = MachoDataset(data_root, prediction_length, mode='test')

    train_dataloader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=256, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=256, shuffle=False)

    config = TimeSeriesTransformerConfig(
        prediction_length=prediction_length,
        context_length=window_length - prediction_length - 7,  # 7 is max(lags) for default lags
        num_time_features=1,
        encoder_layers=2,
        decoder_layers=2,
        d_model=64,
    )

    model = TimeSeriesTransformerForPrediction(config)
    model = model.to(device)

    optimizer = AdamW(model.parameters(), lr=6e-4, betas=(0.9, 0.95), weight_decay=1e-1)

