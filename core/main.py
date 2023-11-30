import os
import wandb
import random
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from transformers import TimeSeriesTransformerConfig, TimeSeriesTransformerForPrediction

from dataset import MachoDataset
from trainer import PredictionTrainer, ClassificationTrainer
from model import ClassificationModel


def main():
    random_seed = 42
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    random.seed(random_seed)
    torch.backends.cudnn.deterministic = True

    config = {
        'random_seed': random_seed,
        'data_root': '/home/mrizhko/AML/contra_periodic/data/macho/',
        'balanced_data_root': '/home/mrizhko/AML/AstroML/data/macho-balanced/',
        'weights_path': '/home/mrizhko/AML/AstroML/weights/no-train-prediction190',

        # Time Series Transformer
        'lags': None,                   # ???
        'num_static_real_features': 3,
        'num_time_features': 1,
        'd_model': 128,
        'decoder_layers': 2,
        'encoder_layers': 2,

        # Data
        'window_length': 200,
        'prediction_length': 10,        # 1 5 10 25 50

        # Training
        'batch_size': 256,
        'lr': 0.001,
        'weight_decay': 0.1,
        'epochs': 100,

        # Learning Rate Scheduler
        'factor': 0.1,
        'patience': 10,

        'mode': 'fine-tuning',          # 'pre-training' 'fine-tuning'
        'save_weights': False,
        'config_from_run': None,        # 'MeriDK/AstroML/656hx6o2'
    }

    if config['config_from_run']:
        run_id = config['config_from_run']
        api = wandb.Api()
        old_run = api.run(run_id)
        old_config = old_run.config

        for el in old_config:
            config[el] = old_config[el]

    run = wandb.init(project='AstroML', config=config)
    print(run.name)
    print(run.config)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using', device)

    if run.config['mode'] == 'pre-training':

        train_dataset = MachoDataset(run.config['data_root'], run.config['prediction_length'], mode='train')
        val_dataset = MachoDataset(run.config['data_root'], run.config['prediction_length'], mode='val')
        test_dataset = MachoDataset(run.config['data_root'], run.config['prediction_length'], mode='test')

        train_dataloader = DataLoader(train_dataset, batch_size=run.config['batch_size'], shuffle=True, num_workers=8)
        val_dataloader = DataLoader(val_dataset, batch_size=4, shuffle=False)
        test_dataloader = DataLoader(test_dataset, batch_size=4, shuffle=False)

        transformer_config = TimeSeriesTransformerConfig(
            prediction_length=run.config['prediction_length'],
            context_length=run.config['window_length'] - run.config['prediction_length'] - 7,  # 7 is max(lags) for default lags
            num_time_features=run.config['num_time_features'],
            num_static_real_features=run.config['num_static_real_features'],
            encoder_layers=run.config['encoder_layers'],
            decoder_layers=run.config['decoder_layers'],
            d_model=run.config['d_model']
        )

        model = TimeSeriesTransformerForPrediction(transformer_config)
        model = model.to(device)
        optimizer = AdamW(model.parameters(), lr=run.config['lr'], weight_decay=run.config['weight_decay'])

        prediction_trainer = PredictionTrainer(model=model, optimizer=optimizer, device=device, use_wandb=True)
        prediction_trainer.train(train_dataloader, val_dataloader, epochs=run.config['epochs'])
        prediction_trainer.evaluate(val_dataloader, val_dataset)

        if run.config['save_weights']:
            model_path = os.path.join(config['weights_path'], run.id)
            model.save_pretrained(model_path)

            artifact = wandb.Artifact('TimeSeriesTransformer', type='model')
            artifact.add_file(os.path.join(model_path, 'config.json'))
            artifact.add_file(os.path.join(model_path, 'generation_config.json'))
            artifact.add_file(os.path.join(model_path, 'pytorch_model.bin'))
            wandb.log_artifact(artifact)

    elif run.config['mode'] == 'fine-tuning':
        model = ClassificationModel(pretrained_model_path=run.config['weights_path'], device=device)
        optimizer = AdamW(model.parameters(), lr=run.config['lr'], weight_decay=run.config['weight_decay'])
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=run.config['factor'],
                                      patience=run.config['patience'], verbose=True)
        criterion = nn.CrossEntropyLoss()

        classification_trainer = ClassificationTrainer(model=model, optimizer=optimizer, scheduler=scheduler,
                                                       criterion=criterion, device=device, use_wandb=True)

        train_dataset = MachoDataset(run.config['balanced_data_root'], run.config['prediction_length'],
                                     window_length=run.config['window_length'], mode='train')
        val_dataset = MachoDataset(run.config['balanced_data_root'], run.config['prediction_length'],
                                   window_length=run.config['window_length'], mode='val')
        test_dataset = MachoDataset(run.config['balanced_data_root'], run.config['prediction_length'],
                                    window_length=run.config['window_length'], mode='test')

        train_dataloader = DataLoader(train_dataset, batch_size=run.config['batch_size'], shuffle=True, num_workers=8)
        val_dataloader = DataLoader(val_dataset, batch_size=run.config['batch_size'], shuffle=False)
        test_dataloader = DataLoader(test_dataset, batch_size=run.config['batch_size'], shuffle=False)

        classification_trainer.train(train_dataloader, val_dataloader, epochs=run.config['epochs'])
        classification_trainer.evaluate(val_dataloader)
    else:
        raise NotImplementedError(f"Incorrect mode {run.config['mode']}")

    wandb.finish()


if __name__ == '__main__':
    main()
