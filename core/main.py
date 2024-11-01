import os
import wandb
import random
import numpy as np
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR, ReduceLROnPlateau, LinearLR
from torch.utils.data import DataLoader
from datetime import datetime

from dataset import PSMDataset
from model import Informer, GalSpecNet, MetaModel, AstroModel
from loss import CLIPLoss
from trainer import Trainer

CLASSES = ['EW', 'SR', 'EA', 'RRAB', 'EB', 'ROT', 'RRC', 'HADS', 'M', 'DSCT']
METADATA_COLS = [
    'mean_vmag',  'phot_g_mean_mag', 'e_phot_g_mean_mag', 'phot_bp_mean_mag', 'e_phot_bp_mean_mag', 'phot_rp_mean_mag',
    'e_phot_rp_mean_mag', 'bp_rp', 'parallax', 'parallax_error', 'parallax_over_error', 'pmra', 'pmra_error', 'pmdec',
    'pmdec_error', 'j_mag', 'e_j_mag', 'h_mag', 'e_h_mag', 'k_mag', 'e_k_mag', 'w1_mag', 'e_w1_mag',
    'w2_mag', 'e_w2_mag', 'w3_mag', 'w4_mag', 'j_k', 'w1_w2', 'w3_w4', 'pm', 'ruwe', 'l', 'b'
]
PHOTO_COLS = ['amplitude', 'period', 'lksl_statistic', 'rfr_score']


def get_model(config):
    if config['mode'] == 'photo':
        model = Informer(config)
    elif config['mode'] == 'spectra':
        model = GalSpecNet(config)
    elif config['mode'] == 'meta':
        model = MetaModel(config)
    else:
        model = AstroModel(config)

    if config['use_pretrain'] and config['use_pretrain'].startswith('CLIP'):
        weights = torch.load(config['use_pretrain'][4:], weights_only=True)

        if config['mode'] == 'photo':
            weights_prefix = 'photometry_encoder'
        elif config['mode'] == 'spectra':
            weights_prefix = 'spectra_encoder'
        elif config['mode'] == 'meta':
            weights_prefix = 'metadata_encoder'
        else:
            weights_prefix = None

        if weights_prefix:
            weights = {k[len(weights_prefix) + 1:]: v for k, v in weights.items() if k.startswith(weights_prefix)}

        model.load_state_dict(weights, strict=False)
        print('Loaded weights from {}'.format(config['use_pretrain']))

    return model


def get_schedulers(config, optimizer):
    if config['scheduler'] == 'ExponentialLR':
        scheduler = ExponentialLR(optimizer, gamma=config['gamma'])
    elif config['scheduler'] == 'ReduceLROnPlateau':
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=config['factor'], patience=config['patience'])
    else:
        raise NotImplementedError(f"Scheduler {config['scheduler']} not implemented")

    if config['warmup']:
        warmup_scheduler = LinearLR(optimizer, start_factor=1e-5, end_factor=1, total_iters=config['warmup_epochs'])
    else:
        warmup_scheduler = None

    return scheduler, warmup_scheduler


def run(config):
    train_dataset = PSMDataset(config, split='train')
    val_dataset = PSMDataset(config, split='val')

    train_dataloader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, drop_last=True,
                                  num_workers=4)
    val_dataloader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using', device)

    model = get_model(config)
    model = model.to(device)

    optimizer = Adam(model.parameters(), lr=config['lr'], betas=(config['beta1'], config['beta2']),
                     weight_decay=config['weight_decay'])
    scheduler, warmup_scheduler = get_schedulers(config, optimizer)
    criterion = CLIPLoss() if config['mode'] == 'clip' else torch.nn.CrossEntropyLoss()

    trainer = Trainer(model=model, optimizer=optimizer, scheduler=scheduler, warmup_scheduler=warmup_scheduler,
                      criterion=criterion, device=device, config=config)
    trainer.train(train_dataloader, val_dataloader, epochs=config['epochs'])

    if config['mode'] != 'clip':
        trainer.evaluate(val_dataloader, id2target=train_dataset.id2target)


def set_random_seeds(random_seed):
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    np.random.seed(random_seed)
    random.seed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_config():
    config = {
        'project': 'AstroCLIPResults3',
        'mode': 'spectra',    # 'clip' 'photo' 'spectra' 'meta' 'all'
        'config_from': 'meridk/AstroCLIPOptuna3/atdo5ybp',    # 'meridk/AstroCLIPResults/d2u52yml',
        'random_seed': 123,  # 42, 66, 0, 12, 123
        'use_wandb': True,
        'save_weights': True,
        'weights_path': f'/home/mariia/AstroML/weights/{datetime.now().strftime("%Y-%m-%d-%H-%M")}',
        # 'use_pretrain': 'CLIP/home/mariia/AstroML/weights/2024-08-14-14-05-zmjau1cu/weights-51.pth',
        'use_pretrain': None,
        'freeze': False,

        # Data General
        'data_root': '/home/mariia/AstroML/data/asassn/',
        'file': 'preprocessed_data/full_lb/spectra_and_v',
        'classes': CLASSES,
        'num_classes': len(CLASSES),
        'meta_cols': METADATA_COLS,
        'photo_cols': PHOTO_COLS,
        'min_samples': None,
        'max_samples': None,

        # Photometry
        'v_zip': 'asassnvarlc_vband_complete.zip',
        'v_prefix': 'vardb_files',
        'seq_len': 200,
        'phased': False,
        'p_aux': True,

        # Spectra
        'lamost_spec_dir': 'Spectra/v2',
        's_mad': True,     # if True use mad for norm else std
        's_aux': True,
        's_err': True,
        's_err_norm': True,

        # Photometry Model
        'p_enc_in': 3,
        'p_d_model': 128,
        'p_dropout': 0.2,
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
        'hidden_dim': 512,
        'fusion': 'avg',  # 'avg', 'concat'

        # Training
        'batch_size': 512,
        'lr': 0.001,
        'beta1': 0.9,
        'beta2': 0.999,
        'weight_decay': 0.01,
        'epochs': 100,
        'early_stopping_patience': 6,
        'scheduler': 'ReduceLROnPlateau',  # 'ExponentialLR', 'ReduceLROnPlateau'
        'gamma': 0.9,  # for ExponentialLR scheduler
        'factor': 0.3,  # for ReduceLROnPlateau scheduler
        'patience': 3,  # for ReduceLROnPlateau scheduler
        'warmup': True,
        'warmup_epochs': 10,
        'clip_grad': True,
        'clip_value': 5
    }

    if config['p_aux']:
        config['p_enc_in'] += len(config['photo_cols']) + 2     # +2 for mad and delta t

    if config['s_aux']:
        config['s_conv_channels'][0] += 1

    if config['s_err']:
        config['s_conv_channels'][0] += 1

    if config['config_from']:
        print(f"Copying params from the {config['config_from']} run")
        old_config = wandb.Api().run(config['config_from']).config

        for el in old_config:
            if el in [
                'p_dropout', 's_dropout', 'm_dropout', 'lr', 'beta1', 'weight_decay', 'epochs',
                'early_stopping_patience', 'factor', 'patience', 'warmup', 'warmup_epochs', 'clip_grad', 'clip_value',
                'use_pretrain', 'freeze', 'phased', 'p_aux', 's_aux', 's_err', 'file'
            ]:
                config[el] = old_config[el]

    if config['random_seed'] != 42:
        file = config['file'].split('/')
        # data splits saved in different folders depending on random seed
        config['file'] = '/'.join(file[:-1]) + str(config['random_seed']) + '/' + file[-1]

    return config


def main():
    config = get_config()
    set_random_seeds(config['random_seed'])

    if config['use_wandb']:
        wandb_run = wandb.init(project=config['project'], config=config)
        config.update(wandb.config)

        config['run_id'] = wandb_run.id
        config['weights_path'] += f'-{wandb_run.id}'
        print(wandb_run.name, config)

    if config['save_weights']:
        os.makedirs(config['weights_path'], exist_ok=True)

    run(config)
    wandb.finish()


if __name__ == '__main__':
    main()
