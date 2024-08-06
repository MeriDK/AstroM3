import os
import wandb
import random
import numpy as np
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR, StepLR, SequentialLR
from torch.utils.data import DataLoader
from datetime import datetime

from dataset import PSMDataset
from model import Informer, GalSpecNet, MetaModel, AstroModel
from loss import CLIPLoss
from trainer import Trainer


CLASSES = ['EW', 'SR', 'EA', 'RRAB', 'EB', 'ROT', 'RRC', 'HADS', 'M', 'DSCT']
METADATA_COLS = [
    'mean_vmag', 'amplitude', 'period', 'phot_g_mean_mag', 'e_phot_g_mean_mag', 'lksl_statistic',
    'rfr_score', 'phot_bp_mean_mag', 'e_phot_bp_mean_mag', 'phot_rp_mean_mag', 'e_phot_rp_mean_mag',
    'bp_rp', 'parallax', 'parallax_error', 'parallax_over_error', 'pmra', 'pmra_error', 'pmdec',
    'pmdec_error', 'j_mag', 'e_j_mag', 'h_mag', 'e_h_mag', 'k_mag', 'e_k_mag', 'w1_mag', 'e_w1_mag',
    'w2_mag', 'e_w2_mag', 'w3_mag', 'w4_mag', 'j_k', 'w1_w2', 'w3_w4', 'pm', 'ruwe'
]


def get_model(config):
    if config['mode'] == 'photo':
        model = Informer(config)
    elif config['mode'] == 'spectra':
        model = GalSpecNet(config)
    elif config['mode'] == 'meta':
        model = MetaModel(config)
    else:
        model = AstroModel(config)

    return model


def run(config):
    train_dataset = PSMDataset(config, split='train')
    val_dataset = PSMDataset(config, split='val')

    config['num_classes'] = train_dataset.num_classes

    train_dataloader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=4)
    val_dataloader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using', device)

    model = get_model(config)
    model = model.to(device)

    optimizer = Adam(model.parameters(), lr=config['lr'], betas=(config['beta1'], config['beta2']),
                     weight_decay=config['weight_decay'])
    scheduler = ExponentialLR(optimizer, gamma=config['gamma'])
    criterion = CLIPLoss() if config['mode'] == 'clip' else torch.nn.CrossEntropyLoss()

    trainer = Trainer(model=model, optimizer=optimizer, scheduler=scheduler, criterion=criterion, device=device,
                      config=config)
    trainer.train(train_dataloader, val_dataloader, epochs=config['epochs'])

    if config['mode'] != 'clip':
        trainer.evaluate(val_dataloader, id2target=train_dataset.id2target)


def set_random_seeds(random_seed):
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    random.seed(random_seed)
    torch.backends.cudnn.deterministic = True


def get_config(random_seed):
    config = {
        'project': 'AstroCLIPFinetune',
        'mode': 'clip',      # 'clip' 'photo' 'spectra' 'meta' 'all'
        'random_seed': random_seed,
        'use_wandb': True,
        'save_weights': False,
        'weights_path': f'/home/mariia/AstroML/weights/{datetime.now().strftime("%Y-%m-%d-%H-%M")}',
        # 'use_pretrain': '/home/mariia/AstroML/weights/2024-07-25-14-18-es6hl0nb/weights-41.pth',
        'use_pretrain': None,
        'freeze': False,

        # Data General
        'data_root': '/home/mariia/AstroML/data/asassn/',
        'file': 'preprocessed_data/full/spectra_and_v',
        'classes': CLASSES,
        'meta_cols': METADATA_COLS,
        'min_samples': None,
        'max_samples': None,

        # Photometry
        'v_zip': 'asassnvarlc_vband_complete.zip',
        'v_prefix': 'vardb_files',
        'seq_len': 200,
        'phased': True,
        'aux': True,

        # Spectra
        'lamost_spec_dir': 'Spectra/v2',
        'spectra_v_file': 'spectra_v_merged.csv',

        # Photometry Model
        'p_enc_in': 3,
        'p_d_model': 128,
        'p_dropout': 0.2,
        'p_feature_size': 3,
        'p_factor': 1,
        'p_output_attention': False,
        'p_n_heads': 4,
        'p_d_ff': 512,
        'p_activation': 'gelu',
        'p_e_layers': 8,

        # Spectra Model
        's_dropout': 0.2,
        's_conv_channels': [1, 64, 64, 32, 32],
        's_kernel_size': 3,
        's_mp_kernel_size': 4,

        # Metadata Model
        'm_hidden_dim': 512,
        'm_dropout': 0.2,

        # MultiModal Model
        'hidden_dim': 1024,
        'fusion': 'avg',  # 'avg', 'concat'

        # Training
        'batch_size': 512,
        'lr': 1e-3,
        'beta1': 0.9,
        'beta2': 0.999,
        'weight_decay': 1e-3,
        'epochs': 100,
        'early_stopping_patience': 10,
        'gamma': 0.9    # for scheduler
    }

    if config['aux']:
        config['p_enc_in'] += 4

    return config


def main():
    random_seed = 42    # 42, 66, 0, 12, 123
    set_random_seeds(random_seed)
    config = get_config(random_seed)

    if config['use_wandb']:
        wandb_run = wandb.init(project=config['project'], config=config)
        config['run_id'] = wandb_run.id
        config['weights_path'] += f'-{wandb_run.id}'
        print(wandb_run.name, config)

    if config['save_weights']:
        os.makedirs(config['weights_path'], exist_ok=True)

    run(config)
    wandb.finish()


if __name__ == '__main__':
    main()

