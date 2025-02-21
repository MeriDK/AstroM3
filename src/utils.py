import random
import numpy as np
import torch
from torch.optim.lr_scheduler import ExponentialLR, ReduceLROnPlateau, LinearLR
from model import Informer, GalSpecNet, MetaModel, AstroM3

CLASSES = ['EW', 'SR', 'EA', 'RRAB', 'EB', 'ROT', 'RRC', 'HADS', 'M', 'DSCT']


def get_model(config):
    """
    Initializes and returns the model based on the mode in the configuration.

    Args:
        config (dict): Dictionary containing model parameters and settings.

    Returns:
        torch.nn.Module: Initialized model.
    """
    if config['mode'] == 'photo':
        # Initialize the photometry model (Informer)
        model = Informer(
            classification=True if config['mode'] == 'photo' else False,
            num_classes=config['num_classes'],
            seq_len=config['seq_len'],
            enc_in=config['p_enc_in'],
            d_model=config['p_d_model'],
            dropout=config['p_dropout'],
            factor=config['p_factor'],
            output_attention=config['p_output_attention'],
            n_heads=config['p_n_heads'],
            d_ff=config['p_d_ff'],
            activation=config['p_activation'],
            e_layers=config['p_e_layers']
        )
    elif config['mode'] == 'spectra':
        # Initialize the spectra model (GalSpecNet)
        model = GalSpecNet(
            classification=True if config['mode'] == 'spectra' else False,
            num_classes=config['num_classes'],
            dropout_rate=config['s_dropout'],
            conv_channels=config['s_conv_channels'],
            kernel_size=config['s_kernel_size'],
            mp_kernel_size=config['s_mp_kernel_size']
        )
    elif config['mode'] == 'meta':
        # Initialize the metadata model (MetaModel)
        model = MetaModel(
            classification=True if config['mode'] == 'meta' else False,
            num_classes=config['num_classes'],
            input_dim=config['m_input_dim'],
            hidden_dim=config['m_hidden_dim'],
            dropout=config['m_dropout']
        )
    else:
        # Initialize the AstroM3 multimodal model
        model = AstroM3(
            classification=True if config['mode'] == 'all' else False,
            num_classes=config['num_classes'],
            hidden_dim=config['hidden_dim'],
            fusion=config['fusion'],

            # Photometry model params
            seq_len=config['seq_len'],
            p_enc_in=config['p_enc_in'],
            p_d_model=config['p_d_model'],
            p_dropout=config['p_dropout'],
            p_factor=config['p_factor'],
            p_output_attention=config['p_output_attention'],
            p_n_heads=config['p_n_heads'],
            p_d_ff=config['p_d_ff'],
            p_activation=config['p_activation'],
            p_e_layers=config['p_e_layers'],

            # Spectra model params
            s_dropout=config['s_dropout'],
            s_conv_channels=config['s_conv_channels'],
            s_kernel_size=config['s_kernel_size'],
            s_mp_kernel_size=config['s_mp_kernel_size'],

            # Metadata model params
            m_input_dim=config['m_input_dim'],
            m_hidden_dim=config['m_hidden_dim'],
            m_dropout=config['m_dropout']
        )

    # Load CLIP pretrained weights if specified
    if config['use_pretrain']:
        weights = torch.load(config['use_pretrain'], weights_only=True)

        # Determine the weights prefix based on the selected mode
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

        if len(weights) == 0:
            raise ValueError('Can load pretrained weights only from CLIP model')

        model.load_state_dict(weights, strict=False)
        print('Loaded weights from {}'.format(config['use_pretrain']))

    return model


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
