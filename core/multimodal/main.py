import wandb
import random
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from transformers.models.time_series_transformer.modeling_time_series_transformer import TimeSeriesTransformerEncoder

from trainer import ClassificationTrainer
from model import ClassificationModel
from core.multimodal.dataset import collate_fn, ASASSNVarStarDataset


def classification(config):

    datapath = Path(config['datapath'])
    train_dataset = ASASSNVarStarDataset(
        datapath, mode='train', verbose=False, only_periodic=config['only_periodic'],
        recalc_period=config['recalc_period'], prime=config['prime'], use_bands=config['use_bands'],
        only_sources_with_spectra=config['only_sources_with_spectra'], return_phased=config['return_phased'],
        fill_value=config['fill_value']
    )
    val_dataset = ASASSNVarStarDataset(
        datapath, mode='val', verbose=False, only_periodic=config['only_periodic'],
        recalc_period=config['recalc_period'], prime=config['prime'], use_bands=config['use_bands'],
        only_sources_with_spectra=config['only_sources_with_spectra'], return_phased=config['return_phased'],
        fill_value=config['fill_value']
    )

    no_spectra_data_keys = ['lcs', 'classes']
    no_spectra_collate_fn = partial(collate_fn, data_keys=no_spectra_data_keys, fill_value=0)

    train_dataloader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, pin_memory=True,
                                  num_workers=8, collate_fn=no_spectra_collate_fn)
    val_dataloader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False, pin_memory=True,
                                num_workers=8, collate_fn=no_spectra_collate_fn)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using', device)

    encoder = TimeSeriesTransformerEncoder(config)
    model = ClassificationModel(encoder, num_classes=len(train_dataset.target_lookup.keys()))
    model = model.to(device)

    optimizer = AdamW(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=config['factor'],
                                  patience=config['patience'], verbose=True)
    criterion = nn.CrossEntropyLoss()

    classification_trainer = ClassificationTrainer(model=model, optimizer=optimizer, scheduler=scheduler,
                                                   criterion=criterion, device=device, use_wandb=False)

    classification_trainer.train(train_dataloader, val_dataloader, epochs=config['epochs'])
    classification_trainer.evaluate(val_dataloader)


def get_config(random_seed):
    config = {
        'random_seed': random_seed,

        # data
        'datapath': '../data/asaasn',
        'only_periodic': True,
        'recalc_period': False,
        'prime': True,
        'use_bands': ['v'],
        'only_sources_with_spectra': True,
        'return_phased': True,
        'fill_value': 0,

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
