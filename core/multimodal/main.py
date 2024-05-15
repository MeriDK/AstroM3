import os
import wandb
import random
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from transformers import TimeSeriesTransformerConfig
from transformers.models.time_series_transformer.modeling_time_series_transformer import TimeSeriesTransformerEncoder
from pathlib import Path
from functools import partial
from datetime import datetime

from trainer import ClassificationTrainer
from model import ClassificationModel
from dataset import collate_fn, ASASSNVarStarDataset
from models.Informer import Informer


def classification(config):

    datapath = Path(config['datapath'])
    train_dataset = ASASSNVarStarDataset(
        datapath, mode='train', verbose=True, only_periodic=config['only_periodic'],
        recalc_period=config['recalc_period'], prime=config['prime'], use_bands=config['use_bands'],
        max_samples=config['max_samples'], only_sources_with_spectra=config['only_sources_with_spectra'],
        return_phased=config['return_phased'], fill_value=config['fill_value']
    )
    val_dataset = ASASSNVarStarDataset(
        datapath, mode='val', verbose=True, only_periodic=config['only_periodic'],
        recalc_period=config['recalc_period'], prime=config['prime'], use_bands=config['use_bands'],
        max_samples=config['max_samples'], only_sources_with_spectra=config['only_sources_with_spectra'],
        return_phased=config['return_phased'], fill_value=config['fill_value']
    )

    no_spectra_data_keys = config['data_keys']
    no_spectra_collate_fn = partial(collate_fn, data_keys=no_spectra_data_keys, fill_value=0)

    train_dataloader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, pin_memory=False,
                                  num_workers=4, collate_fn=no_spectra_collate_fn)
    val_dataloader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False, pin_memory=False,
                                num_workers=0, collate_fn=no_spectra_collate_fn)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using', device)

    if config['model'] == 'vanilla':
        encoder_config = get_encoder_config(config)
        encoder = TimeSeriesTransformerEncoder(encoder_config)
        model = ClassificationModel(encoder, num_classes=len(train_dataset.target_lookup.keys()))
    elif config['model'] == 'informer':
        model = Informer(enc_in=2, d_model=config['d_model'], dropout=config['dropout'], factor=1,
                         output_attention=False, n_heads=config['n_heads'], d_ff=config['d_ff'],
                         activation='gelu', e_layers=config['encoder_layers'], seq_len=config['context_length'],
                         num_class=len(train_dataset.target_lookup))
    else:
        raise ValueError(f'Model {config["model"]} not recognized')

    model = model.to(device)

    if config['use_pretrain']:
        model.load_state_dict(torch.load(config['use_pretrain']))

    optimizer = AdamW(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=config['factor'],
                                  patience=config['patience'], verbose=True)
    criterion = nn.CrossEntropyLoss()

    classification_trainer = ClassificationTrainer(model=model, optimizer=optimizer, scheduler=scheduler,
                                                   criterion=criterion, device=device, config=config,
                                                   use_wandb=config['use_wandb'])

    classification_trainer.train(train_dataloader, val_dataloader, epochs=config['epochs'])
    classification_trainer.evaluate(val_dataloader)


def get_config(random_seed):

    config = {
        'project': 'vband-classification',
        'random_seed': random_seed,
        'use_wandb': True,
        'save_weights': True,
        'weights_path': f'/home/mariia/AstroML/weights/{datetime.now().strftime("%Y-%m-%d-%H-%M")}',
        'use_pretrain': None,
        
        # Data
        'datapath': '/home/mariia/AstroML/data/asassn',
        'scales_file': 'scales.json',
        'only_periodic': True,
        'recalc_period': False,
        'prime': True,
        'use_bands': ['v'],
        'max_samples': 20000,
        'only_sources_with_spectra': False,
        'return_phased': True,
        'fill_value': 0,
        'data_keys': ['lcs', 'classes'],

        # Time Series Transformer
        'model': 'informer',    # 'informer' or 'vanilla'
        'prediction_length': 20,    # doesn't matter for classification, but it's required by hf
        'context_length': 200,
        'num_time_features': 1,
        'num_static_real_features': 0,  # if 0 we don't use real features
        'encoder_layers': 2,
        'd_model': 128,
        'distribution_output': 'normal',
        'scaling': None,
        'dropout': 0,
        'encoder_layerdrop': 0,
        'attention_dropout': 0,
        'activation_dropout': 0,
        'feature_size': 2,

        # Informer
        'n_heads': 4,
        'd_ff': 512,

        # Training
        'batch_size': 512,
        'lr': 1e-3,
        'weight_decay': 0,
        'epochs': 50,

        # Learning Rate Scheduler
        'factor': 0.3,
        'patience': 3,
    }

    return config


def get_encoder_config(config):
    encoder_config = TimeSeriesTransformerConfig(
        prediction_length=config['prediction_length'],  # doesn't matter but it's required by hf
        context_length=config['context_length'],
        num_time_features=config['num_time_features'],
        num_static_real_features=config['num_static_real_features'],
        encoder_layers=config['encoder_layers'],
        d_model=config['d_model'],
        distribution_output=config['distribution_output'],
        scaling=config['scaling'],
        dropout=config['dropout'],
        encoder_layerdrop=config['encoder_layerdrop'],
        attention_dropout=config['attention_dropout'],
        activation_dropout=config['activation_dropout']
    )
    encoder_config.feature_size = config['feature_size']

    return encoder_config


def set_random_seeds(random_seed):
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    random.seed(random_seed)
    torch.backends.cudnn.deterministic = True


def main():
    random_seed = 42
    set_random_seeds(random_seed)
    config = get_config(random_seed)

    if config['use_wandb']:
        run = wandb.init(project=config['project'], config=config)
        config['run_id'] = run.id
        config['weights_path'] += f'-{run.id}'
        print(run.name, config)

    if config['save_weights']:
        os.makedirs(config['weights_path'], exist_ok=True)

    classification(config)
    wandb.finish()


if __name__ == '__main__':
    main()
