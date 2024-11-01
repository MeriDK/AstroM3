import os
import wandb
import random
import numpy as np
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from datetime import datetime

from dataset import VPSMDataset
from dataset2 import VPSMDatasetV2
from trainer import CLIPTrainer
from trainer2 import ClassificationTrainer
from model import ModelV0, ModelV1, ModelV1Classification

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
            random_seed=config['random_seed'], noise=config['noise'], noise_coef=config['noise_coef'],
        )
        val_dataset = VPSMDatasetV2(
            split='val', data_root=config['data_root'], file=config['file'], v_zip=config['v_zip'],
            v_prefix=config['v_prefix'], lamost_spec_dir=config['lamost_spec_dir'], min_samples=config['min_samples'],
            max_samples=config['max_samples'], classes=config['classes'], seq_len=config['seq_len'],
            phased=config['phased'], clip=config['clip'], aux=config['aux'], z_corr=config['z_corr'],
            random_seed=config['random_seed'], noise=False,
        )
    else:
        raise ValueError(f'Invalid dataset: {config["dataset"]}')

    return val_dataset, train_dataset


def get_dataloaders(train_dataset, val_dataset, config):
    train_dataloader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=4)
    val_dataloader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False)

    return val_dataloader, train_dataloader


def get_model(config, num_classes, device):
    if config['model'] == 'ModelV1Classification':
        model = ModelV1Classification(config, num_classes, device)
    elif config['model'] in ('ModelV0', 'ModelV1'):
        model = ModelV0(config) if config['model'] == 'ModelV0' else ModelV1(config)

        if config['use_pretrain']:
            model.load_state_dict(torch.load(config['use_pretrain']))
    else:
        raise ValueError(f'Invalid model: {config["model"]}')

    print(model)

    return model


def get_optimizer(config, model):
    optimizer = AdamW(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    return optimizer


def classification(config):
    val_dataset, train_dataset = get_datasets(config)
    val_dataloader, train_dataloader = get_dataloaders(train_dataset, val_dataset, config)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using', device)

    model = get_model(config, train_dataset.num_classes, device)
    model = model.to(device)

    optimizer = get_optimizer(config, model)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=config['factor'], patience=config['patience'])
    criterion = torch.nn.CrossEntropyLoss()

    if config['training_mode'] == 'clip':
        clip_trainer = CLIPTrainer(model=model, optimizer=optimizer, scheduler=scheduler,
                                   device=device, config=config, use_wandb=config['use_wandb'])
        clip_trainer.train(train_dataloader, val_dataloader, epochs=config['epochs'])
    elif config['training_mode'] == 'classification':
        classification_trainer = ClassificationTrainer(model=model, optimizer=optimizer, scheduler=scheduler,
                                                       criterion=criterion, device=device, config=config,
                                                       use_wandb=config['use_wandb'])
        classification_trainer.train(train_dataloader, val_dataloader, epochs=config['epochs'])
        classification_trainer.evaluate(val_dataloader, id2target=train_dataset.id2target)
    else:
        raise ValueError(f'Invalid training mode: {config["training_mode"]}')


def set_random_seeds(random_seed):
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    random.seed(random_seed)
    torch.backends.cudnn.deterministic = True


def get_config(random_seed):
    config = {
        'project': 'AstroCLIP',     # 'AstroCLIP', 'AstroCLIPResults'
        'training_mode': 'clip',      # 'clip', 'classification'
        'model': 'ModelV1',  # 'ModelV0' or 'ModelV1'
        'random_seed': random_seed,
        'use_wandb': True,
        'save_weights': True,
        'weights_path': f'/home/mariia/AstroML/weights/{datetime.now().strftime("%Y-%m-%d-%H-%M")}',
        # 'use_pretrain': '/home/mariia/AstroML/weights/2024-07-25-14-18-es6hl0nb/weights-41.pth',
        'use_pretrain': None,
        'freeze': False,
        'fusion': 'avg',    # 'avg', 'concat'

        # Data General
        'dataset': 'VPSMDatasetV2',     # 'VPSMDataset' or 'VPSMDatasetV2'
        'data_root': '/home/mariia/AstroML/data/asassn/',
        'file': 'preprocessed_data/full/spectra_and_v',
        'classes': CLASSES,
        'min_samples': None,
        'max_samples': None,
        'noise': False,  # for train data only
        'noise_coef': 2,

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
        'p_dropout': 0.2,
        'p_feature_size': 3,
        'p_n_heads': 4,
        'p_d_ff': 512,

        # Spectra Model
        's_hidden_dim': 512,
        's_dropout': 0.2,

        # Metadata Model
        'm_hidden_dim': 512,
        'm_dropout': 0.2,

        # MultiModal Model
        'hidden_dim': 1024,
        'ps_coef': 1,
        'mp_coef': 1,
        'sm_coef': 1,

        # Training
        'batch_size': 64,
        'lr': 1e-3,
        'weight_decay': 1e-3,
        'epochs': 100,
        'optimizer': 'AdamW',
        'early_stopping_patience': 10,

        # Learning Rate Scheduler
        'factor': 0.3,
        'patience': 5,
    }

    if config['aux']:
        config['p_feature_size'] += 4

    return config


def main():
    random_seed = 123    # 42, 66, 0, 12, 123
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

