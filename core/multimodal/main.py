import os
import wandb
import random
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from datetime import datetime

from dataset import VPSMDataset
from dataset2 import VPSMDatasetV2
from trainer import ClassificationTrainer
from model import ModelV0, ModelV1

CLASSES = ['EW', 'SR', 'EA', 'RRAB', 'EB', 'ROT', 'RRC', 'HADS', 'M', 'DSCT']


def get_datasets(config):
    if config['dataset'] == 'VPSMDataset':
        train_dataset = VPSMDataset(
            data_root=config['data_root'], file=config['file'], split='train', min_samples=config['min_samples'],
            max_samples=config['max_samples'], classes=config['classes'], random_seed=config['random_seed'], verbose=True,
            v_zip=config['v_zip'], v_prefix=config['v_prefix'], seq_len=config['seq_len'], phased=config['phased'],
            clip=config['clip'], aux=config['aux'], lamost_spec_dir=config['lamost_spec_dir'],
            spectra_v_file=config['spectra_v_file'], z_corr=config['z_corr']
        )
        val_dataset = VPSMDataset(
            data_root=config['data_root'], file=config['file'], split='val', min_samples=config['min_samples'],
            max_samples=config['max_samples'], classes=config['classes'], random_seed=config['random_seed'], verbose=True,
            v_zip=config['v_zip'], v_prefix=config['v_prefix'], seq_len=config['seq_len'], phased=config['phased'],
            clip=config['clip'], aux=config['aux'], lamost_spec_dir=config['lamost_spec_dir'],
            spectra_v_file=config['spectra_v_file'], z_corr=config['z_corr']
        )
    elif config['dataset'] == 'VPSMDatasetV2':
        train_dataset = VPSMDatasetV2(
            split='train', data_root=config['data_root'], file=config['file'], v_zip=config['v_zip'],
            v_prefix=config['v_prefix'], lamost_spec_dir=config['lamost_spec_dir'], min_samples=config['min_samples'],
            max_samples=config['max_samples'], classes=config['classes'], seq_len=config['seq_len'],
            phased=config['phased'], clip=config['clip'], aux=config['aux'], z_corr=config['z_corr'],
            random_seed=config['random_seed']
        )
        val_dataset = VPSMDatasetV2(
            split='val', data_root=config['data_root'], file=config['file'], v_zip=config['v_zip'],
            v_prefix=config['v_prefix'], lamost_spec_dir=config['lamost_spec_dir'], min_samples=config['min_samples'],
            max_samples=config['max_samples'], classes=config['classes'], seq_len=config['seq_len'],
            phased=config['phased'], clip=config['clip'], aux=config['aux'], z_corr=config['z_corr'],
            random_seed=config['random_seed']
        )
    else:
        raise ValueError(f'Invalid dataset: {config["dataset"]}')

    return val_dataset, train_dataset


def get_dataloaders(train_dataset, val_dataset, config):
    train_dataloader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=4)
    val_dataloader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False)

    return val_dataloader, train_dataloader


def get_model(config):
    if config['model'] == 'ModelV0':
        model = ModelV0(config)
    elif config['model'] == 'ModelV1':
        model = ModelV1(config)
    else:
        raise ValueError(f'Invalid model: {config["model"]}')

    print(model)

    if config['use_pretrain']:
        model.load_state_dict(torch.load(config['use_pretrain']))

    return model


def get_optimizer(config, model):
    optimizer = AdamW(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    return optimizer


def classification(config):
    val_dataset, train_dataset = get_datasets(config)
    val_dataloader, train_dataloader = get_dataloaders(train_dataset, val_dataset, config)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using', device)

    model = get_model(config)
    model = model.to(device)

    optimizer = get_optimizer(config, model)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=config['factor'], patience=config['patience'])

    classification_trainer = ClassificationTrainer(model=model, optimizer=optimizer, scheduler=scheduler,
                                                   device=device, config=config, use_wandb=config['use_wandb'])
    classification_trainer.train(train_dataloader, val_dataloader, epochs=config['epochs'])


def set_random_seeds(random_seed):
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    random.seed(random_seed)
    torch.backends.cudnn.deterministic = True


def get_config(random_seed):
    config = {
        'project': 'AstroCLIP',
        'random_seed': random_seed,
        'use_wandb': True,
        'save_weights': True,
        'weights_path': f'/home/mariia/AstroML/weights/{datetime.now().strftime("%Y-%m-%d-%H-%M")}',
        'use_pretrain': None,

        # Data General
        'dataset': 'VPSMDatasetV2',     # 'VPSMDataset' or 'VPSMDatasetV2'
        'data_root': '/home/mariia/AstroML/data/asassn/',
        'file': 'preprocessed_data/full/spectra_and_v',
        'classes': CLASSES,
        'min_samples': None,
        'max_samples': None,

        # Photometry
        'v_zip': 'asassnvarlc_vband_complete.zip',
        'v_prefix': 'vardb_files',
        'seq_len': 200,
        'phased': True,
        'clip': False,
        'aux': True,

        # Spectra
        'lamost_spec_dir': 'Spectra/v2',
        'spectra_v_file': 'spectra_v_merged.csv',
        'z_corr': False,

        # Photometry Model
        'p_encoder_layers': 8,
        'p_d_model': 128,
        'p_dropout': 0,
        'p_feature_size': 3,
        'p_n_heads': 4,
        'p_d_ff': 512,

        # Spectra Model
        's_hidden_dim': 512,
        's_dropout': 0,

        # Metadata Model
        'm_hidden_dim': 512,
        'm_dropout': 0,

        # MultiModal Model
        'model': 'ModelV1',     # 'ModelV0' or 'ModelV1'
        'hidden_dim': 256,

        # Training
        'batch_size': 32,
        'lr': 1e-3,
        'weight_decay': 0,
        'epochs': 50,
        'optimizer': 'AdamW',
        'early_stopping_patience': 100,

        # Learning Rate Scheduler
        'factor': 0.3,
        'patience': 50,
    }

    config['p_feature_size'] += 4

    return config


def main():
    random_seed = 66
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

