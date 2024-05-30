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

from dataset import MetaVDataset
from trainer import ClassificationTrainer
from model import MetaClassifier

CLASSES = ['EW', 'SR', 'EA', 'RRAB', 'L', 'EB', 'ROT', 'RRC', 'VAR', 'ROT:', 'M', 'HADS', 'DSCT']


def get_datasets(config):
    train_dataset = MetaVDataset(
        config['file'], split='train', classes=config['classes'], min_samples=config['min_samples'],
        max_samples=config['max_samples'], random_seed=config['random_seed'], verbose=True
    )
    val_dataset = MetaVDataset(
        config['file'], split='val', classes=config['classes'], min_samples=config['min_samples'],
        max_samples=config['max_samples'], random_seed=config['random_seed'], verbose=True
    )

    return val_dataset, train_dataset


def get_dataloaders(train_dataset, val_dataset, config):
    train_dataloader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=4)
    val_dataloader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False)

    return val_dataloader, train_dataloader


def get_model(num_classes, config):
    if config['model'] == 'MetaClassifier':
        model = MetaClassifier(hidden_dim=config['hidden_dim'], num_classes=num_classes, dropout=config['dropout'])
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
        'project': 'meta-classification',
        'random_seed': random_seed,
        'use_wandb': True,
        'save_weights': True,
        'weights_path': f'/home/mariia/AstroML/weights/{datetime.now().strftime("%Y-%m-%d-%H-%M")}',
        'use_pretrain': None,

        # Data
        'file': '/home/mariia/AstroML/data/asassn/asassn_catalog_full.csv',
        'classes': CLASSES,
        'min_samples': None,
        'max_samples': 20000,

        # Model
        'model': 'MetaClassifier',
        'hidden_dim': 512,
        'dropout': 0.5,

        # Training
        'batch_size': 128,
        'lr': 1e-3,
        'weight_decay': 0.01,
        'epochs': 50,
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
