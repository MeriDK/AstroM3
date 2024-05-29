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

from dataset import SpectraVDataset
from trainer import ClassificationTrainer
from model import GalSpecNet, ResNet1

CLASSES = ['EW', 'SR', 'EA', 'RRAB', 'L', 'EB', 'ROT']


def get_datasets(config):
    train_dataset = SpectraVDataset(
        split='train', data_root=config['data_root'], lamost_spec_dir=config['lamost_spec_dir'],
        spectra_v_file=config['spectra_v_file'], classes=config['classes'], z_corr=config['z_corr']
    )
    val_dataset = SpectraVDataset(
        split='val', data_root=config['data_root'], lamost_spec_dir=config['lamost_spec_dir'],
        spectra_v_file=config['spectra_v_file'], classes=config['classes'], z_corr=config['z_corr']
    )

    return val_dataset, train_dataset


def get_dataloaders(train_dataset, val_dataset, config):
    train_dataloader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=4)
    val_dataloader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False)

    return val_dataloader, train_dataloader


def get_model(num_classes, config):
    if config['model'] == 'GalSpecNet':
        model = GalSpecNet(num_classes=num_classes)
    elif config['model'] == 'ResNet1':
        model = ResNet1(num_classes=num_classes, dropout=config['dropout'])
    else:
        raise ValueError(f'Model {config["model"]} not supported')

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
        'project': 'spectra-classification',
        'random_seed': random_seed,
        'use_wandb': True,
        'save_weights': True,
        'weights_path': f'/home/mariia/AstroML/weights/{datetime.now().strftime("%Y-%m-%d-%H-%M")}',
        'use_pretrain': None,

        # Data
        'data_root': '/home/mariia/AstroML/data/asassn',
        'lamost_spec_dir': 'Spectra/v2',
        'spectra_v_file': 'spectra_v_merged.csv',
        'classes': CLASSES,
        'z_corr': False,

        # Model
        'model': 'GalSpecNet',

        # Training
        'batch_size': 128,
        'lr': 1e-3,
        'weight_decay': 0.01,
        'epochs': 10,
        'optimizer': 'AdamW',
        'early_stopping_patience': 10,

        # Learning Rate Scheduler
        'factor': 0.3,
        'patience': 5,
    }

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
