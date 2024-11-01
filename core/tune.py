import os
import wandb
import random
import numpy as np
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR, ReduceLROnPlateau, LinearLR
from torch.utils.data import DataLoader
from datetime import datetime
import optuna
from optuna.exceptions import DuplicatedStudyError

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


def set_random_seeds(random_seed):
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    np.random.seed(random_seed)
    random.seed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_config(trial):
    config = {
        'project': 'AstroCLIPOptuna3',
        'random_seed': 42,  # 42, 66, 0, 12, 123
        'use_wandb': True,
        'use_optuna': True,
        'save_weights': False,
        'weights_path': f'/home/mariia/AstroML/weights/{datetime.now().strftime("%Y-%m-%d-%H-%M")}',
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
        's_mad': True,  # if True use mad for norm else std
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
        'lr': 0.0003,
        'beta1': 0.9,
        'beta2': 0.999,
        'weight_decay': 0.01,
        'early_stopping_patience': 6,
        'scheduler': 'ReduceLROnPlateau',  # 'ExponentialLR', 'ReduceLROnPlateau'
        'gamma': 0.9,  # for ExponentialLR scheduler
        'factor': 0.3,  # for ReduceLROnPlateau scheduler
        'patience': 3,  # for ReduceLROnPlateau scheduler
        'warmup': True,
        'warmup_epochs': 10,
        'clip_grad': False,
        'clip_value': 45
    }

    if STUDY_NAME.startswith('clip'):
        config['mode'] = 'clip'
        config['epochs'] = 100
        config['clip_grad'] = True
        config['clip_value'] = 45
    elif STUDY_NAME.startswith('photo'):
        config['mode'] = 'photo'
        config['epochs'] = 50
        config['clip_grad'] = True
        config['clip_value'] = 5
    elif STUDY_NAME.startswith('spectra'):
        config['mode'] = 'spectra'
        config['epochs'] = 50
        config['clip_grad'] = True
        config['clip_value'] = 5
    elif STUDY_NAME.startswith('meta'):
        config['mode'] = 'meta'
        config['epochs'] = 50
        config['clip_grad'] = False
    elif STUDY_NAME.startswith('psm'):
        config['mode'] = 'all'
        config['epochs'] = 50
        config['clip_grad'] = False
    else:
        raise NotImplementedError(f"Unknown study name {STUDY_NAME}")

    if 'clip' in STUDY_NAME and STUDY_NAME != 'clip':    # ('metaclip', 'photoclip', 'spectraclip', 'psmclip')
        config['use_pretrain'] = 'CLIP/home/mariia/AstroML/weights/2024-09-12-13-21-03ai5zsz/weights-best.pth'
    else:
        config['use_pretrain'] = None

    if STUDY_NAME.endswith('10'):
        config['file'] = 'preprocessed_data/sub10_lb/spectra_and_v'
    elif STUDY_NAME.endswith('25'):
        config['file'] = 'preprocessed_data/sub25_lb/spectra_and_v'
    elif STUDY_NAME.endswith('50'):
        config['file'] = 'preprocessed_data/sub50_lb/spectra_and_v'

    if config['p_aux']:
        config['p_enc_in'] += len(config['photo_cols']) + 2     # +2 for mad and delta t

    if config['s_aux']:
        config['s_conv_channels'][0] += 1

    if config['s_err']:
        config['s_conv_channels'][0] += 1

    if config['use_optuna']:
        if config['mode'] in ('photo', 'all', 'clip'):
            config['p_dropout'] = trial.suggest_float('p_dropout', 0.0, 0.4)

        if config['mode'] in ('spectra', 'all', 'clip'):
            config['s_dropout'] = trial.suggest_float('s_dropout', 0.0, 0.4)

        if config['mode'] in ('meta', 'all', 'clip'):
            config['m_dropout'] = trial.suggest_float('m_dropout', 0.0, 0.4)

        config['lr'] = trial.suggest_float('lr', 1e-5, 1e-2, log=True)
        config['factor'] = trial.suggest_float('factor', 0.1, 1.0)
        config['beta1'] = trial.suggest_float('beta1', 0.7, 0.99)
        config['weight_decay'] = trial.suggest_float('weight_decay', 1e-5, 1e-1, log=True)

    config['study_name'] = STUDY_NAME

    return config


def run(config, trial):
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
                      criterion=criterion, device=device, config=config, trial=trial)
    best_val_loss = trainer.train(train_dataloader, val_dataloader, epochs=config['epochs'])

    if config['mode'] != 'clip':
        trainer.evaluate(val_dataloader, id2target=train_dataset.id2target)

    return best_val_loss


def objective(trial):
    config = get_config(trial)
    set_random_seeds(config['random_seed'])

    if config['use_wandb']:
        wandb_run = wandb.init(project=config['project'], config=config, group=STUDY_NAME, reinit=True)
        config.update(wandb.config)

        config['run_id'] = wandb_run.id
        config['weights_path'] += f'-{wandb_run.id}'
        print(wandb_run.name, config)

    if config['save_weights']:
        os.makedirs(config['weights_path'], exist_ok=True)

    best_val_loss = run(config, trial)
    wandb.finish()

    return best_val_loss


if __name__ == '__main__':
    STUDY_NAME = 'spectra50'
    storage = f'mysql://root@localhost/{STUDY_NAME}'

    try:
        study = optuna.create_study(study_name=STUDY_NAME, storage=storage, direction='minimize',
                                    pruner=optuna.pruners.NopPruner())
        print(f"Study '{STUDY_NAME}' created.")
    except DuplicatedStudyError:
        study = optuna.load_study(study_name=STUDY_NAME, storage=storage, pruner=optuna.pruners.NopPruner())
        print(f"Study '{STUDY_NAME}' loaded.")

    study.optimize(objective, n_trials=100)
