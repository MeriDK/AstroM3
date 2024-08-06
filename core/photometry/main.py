import os
import wandb
import random
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW, RAdam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from transformers import TimeSeriesTransformerConfig
from transformers.models.time_series_transformer.modeling_time_series_transformer import TimeSeriesTransformerEncoder
from datetime import datetime
from pathlib import Path
from functools import partial

from trainer import ClassificationTrainer
from model import ClassificationModel
from dataset import ASASSNVarStarDataset, collate_fn
from dataset2 import VGDataset, VPSMDatasetV2Photo
from models.Informer import Informer


# Classes from ASAS-SN paper. Except for L, GCAS, YSO, GCAS: and VAR
# CLASSES = ['CWA', 'CWB', 'DCEP', 'DCEPS', 'DSCT', 'EA', 'EB', 'EW',
#            'HADS', 'M', 'ROT', 'RRAB', 'RRC', 'RRD', 'RVA', 'SR']
CLASSES = ['EW', 'SR', 'EA', 'RRAB', 'EB', 'ROT', 'RRC', 'HADS', 'M', 'DSCT']


def get_datasets(config):
    if config['dataset_class'] == 'VGDataset':
        train_dataset = VGDataset(
            config['data_root'], config['vg_file'], split='train', seq_len=config['seq_len'],
            min_samples=config['min_samples'], max_samples=config['max_samples'], phased=config['phased'],
            periodic=config['periodic'], classes=config['classes'], clip=config['clip_outliers'],
            random_seed=config['random_seed'], scales=config['scales'], aux=config['aux']
        )
        val_dataset = VGDataset(
            config['data_root'], config['vg_file'], split='val', seq_len=config['seq_len'],
            min_samples=config['min_samples'], max_samples=config['max_samples'], phased=config['phased'],
            periodic=config['periodic'], classes=config['classes'], clip=config['clip_outliers'],
            random_seed=config['random_seed'], scales=config['scales'], aux=config['aux']
        )
    elif config['dataset_class'] == 'ASASSNVarStarDataset':
        datapath = Path(config['data_root'])
        rng = np.random.default_rng(config['random_seed'])

        train_dataset = ASASSNVarStarDataset(
            datapath, mode='train', verbose=True, only_periodic=config['periodic'], recalc_period=False, prime=True,
            use_bands=['v'], max_samples=config['max_samples'], only_sources_with_spectra=False,
            return_phased=config['phased'], fill_value=0, rng=rng
        )
        val_dataset = ASASSNVarStarDataset(
            datapath, mode='val', verbose=True, only_periodic=config['periodic'], recalc_period=False, prime=True,
            use_bands=['v'], max_samples=config['max_samples'], only_sources_with_spectra=False,
            return_phased=config['phased'], fill_value=0, rng=rng
        )
    elif config['dataset_class'] == 'VPSMDatasetV2Photo':
        train_dataset = VPSMDatasetV2Photo(
            split='train', data_root=config['data_root'], file=config['file'], v_zip=config['v_zip'],
            v_prefix=config['v_prefix'], min_samples=config['min_samples'],
            max_samples=config['max_samples'], classes=config['classes'], seq_len=config['seq_len'],
            phased=config['phased'], clip=config['clip'], aux=config['aux'], random_seed=config['random_seed']
        )
        val_dataset = VPSMDatasetV2Photo(
            split='val', data_root=config['data_root'], file=config['file'], v_zip=config['v_zip'],
            v_prefix=config['v_prefix'], min_samples=config['min_samples'],
            max_samples=config['max_samples'], classes=config['classes'], seq_len=config['seq_len'],
            phased=config['phased'], clip=config['clip'], aux=config['aux'], random_seed=config['random_seed']
        )
    else:
        raise ValueError(f"Dataset class {config['dataset_class']} not supported")

    return val_dataset, train_dataset


def get_dataloaders(train_dataset, val_dataset, config):
    if config['dataset_class'] == 'ASASSNVarStarDataset':
        no_spectra_data_keys = ['lcs', 'classes']
        no_spectra_collate_fn = partial(collate_fn, data_keys=no_spectra_data_keys, fill_value=0)

        train_dataloader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, pin_memory=False,
                                      num_workers=4, collate_fn=no_spectra_collate_fn)
        val_dataloader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False, pin_memory=False,
                                    num_workers=0, collate_fn=no_spectra_collate_fn)
    else:
        train_dataloader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=4)
        val_dataloader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False)

    return val_dataloader, train_dataloader


def get_encoder_config(config):
    encoder_config = TimeSeriesTransformerConfig(
        prediction_length=config['prediction_length'],  # doesn't matter but it's required by hf
        context_length=config['seq_len'],
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


def get_model(num_classes, config):
    if config['model'] == 'vanilla':
        encoder_config = get_encoder_config(config)
        encoder = TimeSeriesTransformerEncoder(encoder_config)
        model = ClassificationModel(encoder, num_classes=num_classes)
    elif config['model'] == 'informer':
        model = Informer(enc_in=config['feature_size'], d_model=config['d_model'], dropout=config['dropout'], factor=1,
                         output_attention=False, n_heads=config['n_heads'], d_ff=config['d_ff'],
                         activation='gelu', e_layers=config['encoder_layers'], seq_len=config['seq_len'],
                         num_class=num_classes)
    else:
        raise ValueError(f'Model {config["model"]} not recognized')

    if config['use_pretrain']:
        model.load_state_dict(torch.load(config['use_pretrain']))

    return model


def get_optimizer(config, model):
    if config['optimizer'] == 'AdamW':
        optimizer = AdamW(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    elif config['optimizer'] == 'RAdam':
        optimizer = RAdam(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    else:
        raise NotImplementedError(f'Optimizer {config["optimizer"]} not implemented')

    return optimizer


def classification(config):

    val_dataset, train_dataset = get_datasets(config)
    val_dataloader, train_dataloader = get_dataloaders(train_dataset, val_dataset, config)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using', device)

    model = get_model(train_dataset.num_classes, config)
    model = model.to(device)

    optimizer = get_optimizer(config, model)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=config['factor'], patience=config['patience'])
    criterion = nn.CrossEntropyLoss()

    classification_trainer = ClassificationTrainer(model=model, optimizer=optimizer, scheduler=scheduler,
                                                   criterion=criterion, device=device, config=config,
                                                   use_wandb=config['use_wandb'])
    classification_trainer.train(train_dataloader, val_dataloader, epochs=config['epochs'])
    classification_trainer.evaluate(val_dataloader, val_dataset.id2target)


def set_random_seeds(random_seed):
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    random.seed(random_seed)
    torch.backends.cudnn.deterministic = True


def get_config(random_seed):

    config = {
        'project': 'AstroCLIPResults',
        'random_seed': random_seed,
        'use_wandb': True,
        'save_weights': True,
        'weights_path': f'/home/mariia/AstroML/weights/{datetime.now().strftime("%Y-%m-%d-%H-%M")}',
        'use_pretrain': None,
        
        # Data
        'dataset_class': 'VPSMDatasetV2Photo',   # 'VGDataset' or 'ASASSNVarStarDataset' 'VPSMDatasetV2Photo'
        # 'vg_file': 'v.csv',     # 'vg_combined.csv', 'v.csv', 'g.csv'
        'scales': 'mean-mad',    # 'scales.json', 'mean-std', 'mean-mad'

        'periodic': False,
        'clip_outliers': False,

        'dataset': 'VPSMDatasetV2Photo',  # 'VPSMDataset' or 'VPSMDatasetV2'
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

        # Model
        'model': 'informer',  # 'informer' or 'vanilla'
        'encoder_layers': 8,
        'd_model': 128,
        'dropout': 0,
        'feature_size': 3,

        # Informer
        'n_heads': 4,
        'd_ff': 512,

        # Time Series Transformer
        'prediction_length': 20,    # doesn't matter for classification, but it's required by hf
        'num_time_features': 1,
        'num_static_real_features': 0,  # if 0 we don't use real features
        'distribution_output': 'normal',
        'scaling': None,
        'encoder_layerdrop': 0,
        'attention_dropout': 0,
        'activation_dropout': 0,

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
        # config['feature_size'] += 5     # + (min, max, mean, std, period)
        config['feature_size'] += 4

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
