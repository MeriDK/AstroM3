import wandb
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import TimeSeriesTransformerConfig, TimeSeriesTransformerForPrediction

from dataset import MachoDataset
from trainer import PredictionTrainer, ClassificationTrainer
from model import ClassificationModel


def main():
    config = {
        'data_root': '/home/mrizhko/AML/contra_periodic/data/macho/',
        'weights_path': '/home/mrizhko/AML/AstroML/weights/model.ckpt',
        'window_length': 200,
        'prediction_length': 10,        # 1, 5, 10, 25, 50
        'lags': None,                   # ???
        'batch_size': 124,              # 32, 64, 128, 256, 512
        'num_time_features': 1,
        'num_static_real_features': 3,
        'encoder_layers': 2,            # 2, 4, 8
        'decoder_layers': 2,            # 2, 4, 8
        'd_model': 64,                  # 32, 64, 128, 256
        'lr': 1e-3,                     # 0.001 0.003 0.0001 0.0003 0.00001 0.00003
        'weight_decay': 0.01,           # 0 0.1 0.001 0.0001
        'epochs': 5
    }
    wandb.init(project='AstroML', config=config)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using', device)

    train_dataset = MachoDataset(config['data_root'], config['prediction_length'], mode='train')
    val_dataset = MachoDataset(config['data_root'], config['prediction_length'], mode='val')
    test_dataset = MachoDataset(config['data_root'], config['prediction_length'], mode='test')

    train_dataloader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False)

    transformer_config = TimeSeriesTransformerConfig(
        prediction_length=config['prediction_length'],
        context_length=config['window_length'] - config['prediction_length'] - 7,  # 7 is max(lags) for default lags
        num_time_features=config['num_time_features'],
        num_static_real_features=config['num_static_real_features'],
        encoder_layers=config['encoder_layers'],
        decoder_layers=config['decoder_layers'],
        d_model=config['d_model']
    )

    model = TimeSeriesTransformerForPrediction(transformer_config)
    model = model.to(device)
    optimizer = AdamW(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])

    prediction_trainer = PredictionTrainer(model=model, optimizer=optimizer, device=device)
    prediction_trainer.train(train_dataloader, val_dataloader, epochs=config['epochs'])
    prediction_trainer.evaluate(val_dataloader, val_dataset)

    model.save_pretrained(config['weights_path'])
    artifact = wandb.Artifact('TimeSeriesTransformer', type='model')
    artifact.add_file(config['weights_path'])
    wandb.log_artifact(artifact)

    wandb.finish()


if __name__ == '__main__':
    main()
