import yaml
import random
import numpy as np
import torch
from torch.optim.lr_scheduler import ExponentialLR, ReduceLROnPlateau, LinearLR


def get_schedulers(config, optimizer):
    """
    Initializes and returns the learning rate scheduler.

    Args:
        config (dict): Configuration containing scheduler type and hyperparameters.
        optimizer (torch.optim.Optimizer): Optimizer for training.

    Returns:
        tuple: (scheduler, warmup_scheduler)
    """
    if config['scheduler'] == 'ExponentialLR':
        scheduler = ExponentialLR(optimizer, gamma=config['scheduler_gamma'])
    elif config['scheduler'] == 'ReduceLROnPlateau':
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=config['scheduler_factor'],
                                      patience=config['scheduler_patience'])
    else:
        raise NotImplementedError(f"Scheduler {config['scheduler']} not implemented")

    # Warmup scheduler (Optional)
    if config['warmup_epochs'] > 0:
        warmup_scheduler = LinearLR(optimizer, start_factor=1e-5, end_factor=1, total_iters=config['warmup_epochs'])
    else:
        warmup_scheduler = None

    return scheduler, warmup_scheduler


def set_random_seeds(random_seed):
    """
    Sets random seeds for reproducibility.

    Args:
        random_seed (int): The seed value.
    """
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    np.random.seed(random_seed)
    random.seed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_config(config_path):
    """
    Load model and training configurations from a YAML file.

    Args:
        config_path (str): Path to the YAML configuration file.

    Returns:
        dict: Parsed configuration dictionary.
    """
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config


class EarlyStopping:
    """
    Implements early stopping to stop training when validation loss stops improving.

    Args:
        patience (int, optional): Number of epochs to wait before stopping. Default is 15.
    """
    def __init__(self, patience=15):
        self.patience = patience
        self.counter = 0
        self.best_score = None

    def step(self, metric):
        """
        Checks if validation loss has improved. Stops training if not.

        Args:
            metric (float): The current validation loss.

        Returns:
            bool: True if training should stop, False otherwise.
        """
        if self.best_score is None:
            self.best_score = metric
            self.counter = 1
        else:
            if metric < self.best_score:
                self.best_score = metric
                self.counter = 0
            else:
                self.counter += 1
        return self.counter >= self.patience
