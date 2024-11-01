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

from dev.periodic.dataset import MachoDataset
from trainer import PredictionTrainer, ClassificationTrainer
from model import ClassificationModel


def pre_train(config):
    train_dataset = MachoDataset(config['data_root'], config['prediction_length'], mode='train')
    val_dataset = MachoDataset(config['data_root'], config['prediction_length'], mode='val')
    test_dataset = MachoDataset(config['data_root'], config['prediction_length'], mode='test')

    train_dataloader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=8)
    val_dataloader = DataLoader(val_dataset, batch_size=4, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=4, shuffle=False)

    transformer_config = TimeSeriesTransformerConfig(
        prediction_length=config['prediction_length'],
        context_length=config['window_length'] - config['prediction_length'] - 7,  # 7 is max(lags) for default lags
        num_time_features=config['num_time_features'],
        num_static_real_features=config['num_static_real_features'],
        encoder_layers=config['encoder_layers'],
        decoder_layers=config['decoder_layers'],
        d_model=config['d_model'],
        distribution_output='normal',
        scaling=None,
        dropout=config['dropout'],
        encoder_layerdrop=config['encoder_layerdrop'],
        decoder_layerdrop=config['decoder_layerdrop'],
        attention_dropout=config['attention_dropout'],
        activation_dropout=config['activation_dropout']
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using', device)

    model = TimeSeriesTransformerForPrediction(transformer_config)
    model = model.to(device)
    optimizer = AdamW(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=config['factor'],
                                  patience=config['patience'], verbose=True)

    prediction_trainer = PredictionTrainer(model=model, optimizer=optimizer, scheduler=scheduler,
                                           device=device, use_wandb=True)
    prediction_trainer.train(train_dataloader, val_dataloader, epochs=config['epochs_pre_training'])
    # prediction_trainer.evaluate(val_dataloader, val_dataset)

    if config['save_weights']:
        model_path = os.path.join(config['weights_path'], config['run_id'])
        model.save_pretrained(model_path)

        artifact = wandb.Artifact('TimeSeriesTransformer', type='model')
        artifact.add_file(os.path.join(model_path, 'config.json'))
        artifact.add_file(os.path.join(model_path, 'generation_config.json'))
        artifact.add_file(os.path.join(model_path, 'pytorch_model.bin'))
        wandb.log_artifact(artifact)

    return model


def fine_tuning(config, pretrained_model=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using', device)

    if pretrained_model:
        model = ClassificationModel(device=device, pretrained_model=pretrained_model)
    else:
        model = ClassificationModel(device=device, pretrained_model_path=config['weights_path'])

    optimizer = AdamW(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=config['factor'],
                                  patience=config['patience'], verbose=True)
    criterion = nn.CrossEntropyLoss()

    classification_trainer = ClassificationTrainer(model=model, optimizer=optimizer, scheduler=scheduler,
                                                   criterion=criterion, device=device, use_wandb=True)

    train_dataset = MachoDataset(config['balanced_data_root'], config['prediction_length'],
                                 window_length=config['window_length'], mode='train')
    val_dataset = MachoDataset(config['balanced_data_root'], config['prediction_length'],
                               window_length=config['window_length'], mode='val')
    test_dataset = MachoDataset(config['balanced_data_root'], config['prediction_length'],
                                window_length=config['window_length'], mode='test')

    train_dataloader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=8)
    val_dataloader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False)

    classification_trainer.train(train_dataloader, val_dataloader, epochs=config['epochs_fine_tuning'])
    classification_trainer.evaluate(val_dataloader)


def get_config(random_seed):
    config = {
        'random_seed': random_seed,
        'data_root': '/home/mrizhko/AML/contra_periodic/data/macho/',
        'balanced_data_root': '/home/mrizhko/AML/AstroML/data/macho-balanced/',
        'weights_path': '/home/mrizhko/AML/AstroML/weights/',

        # Time Series Transformer
        'lags': None,  # ?
        'num_static_real_features': 0,  # if 0 we don't use real features
        'num_time_features': 1,
        'd_model': 256,
        'decoder_layers': 4,
        'encoder_layers': 8,
        'dropout': 0,
        'encoder_layerdrop': 0,
        'decoder_layerdrop': 0,
        'attention_dropout': 0,
        'activation_dropout': 0,

        # Data
        'window_length': 200,
        'prediction_length': 10,  # 1 5 10 25 50

        # Training
        'batch_size': 32,
        'lr': 0.0001,
        'weight_decay': 0.001,
        'epochs_pre_training': 1000,
        'epochs_fine_tuning': 100,

        # Learning Rate Scheduler
        'factor': 0.3,
        'patience': 10,

        'mode': 'pre-training',  # 'pre-training' 'fine-tuning' 'both'
        'save_weights': False,
        'config_from_run': None,  # 'MeriDK/AstroML/qtun67bq'
    }

    if config['config_from_run']:
        print(f"Copying params from the {config['config_from_run']} run")

        run_id = config['config_from_run']
        api = wandb.Api()
        old_run = api.run(run_id)
        old_config = old_run.config

        for el in old_config:
            if el not in ['run_id', 'save_weights', 'config_from_run']:
                config[el] = old_config[el]

    return config


def set_random_seeds(random_seed):
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    random.seed(random_seed)
    torch.backends.cudnn.deterministic = True


def main():
    random_seed = 42
    set_random_seeds(random_seed)
    config = get_config(random_seed)

    run = wandb.init(project='AstroML', config=config)
    run.config['run_id'] = run.id
    print(run.name, run.config)

    if run.config['mode'] == 'pre-training':
        print('Pre training...')
        pre_train(run.config)

    elif run.config['mode'] == 'fine-tuning':
        print('Fine tuning...')
        fine_tuning(run.config)

    elif run.config['mode'] == 'both':
        print('Pre training...')
        pretrained_model = pre_train(run.config)

        print('Fine tuning...')
        fine_tuning(run.config, pretrained_model=pretrained_model)
    else:
        raise NotImplementedError(f"Incorrect mode {run.config['mode']}")

    wandb.finish()


if __name__ == '__main__':
    main()
