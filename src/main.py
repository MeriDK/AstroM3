import os
import wandb
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from datetime import datetime
from datasets import load_dataset
import yaml
import argparse

from dataset import HFPSMDataset
from loss import CLIPLoss
from trainer import Trainer
from utils import get_model, get_schedulers, set_random_seeds


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


def run(config):
    """
    Prepares and runs the training pipeline: loads the dataset, initializes the model, optimizer,
    scheduler, loss function, and trainer, then trains and evaluates the model.

    Args:
        config (dict): Dictionary containing model, dataset, and training configurations.
    """
    # Load the processed dataset from Hugging Face Hub
    ds = load_dataset('MeriDK/AstroM3Processed', name=f'{config["data_sub"]}_{config["random_seed"]}_norm')

    # Initialize train and validation datasets
    train_dataset = HFPSMDataset(ds, classes=config['classes'], seq_len=config['seq_len'], split='train')
    val_dataset = HFPSMDataset(ds, classes=config['classes'], seq_len=config['seq_len'], split='validation')

    # DataLoader for batch processing
    train_dataloader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True,
                                  drop_last=True, num_workers=4)
    val_dataloader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False)

    # Set device (GPU if available, otherwise CPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using', device)

    # Initialize the model based on the config
    model = get_model(config)
    model = model.to(device)

    # Define optimizer (Adam with betas and weight decay)
    optimizer = Adam(model.parameters(), lr=config['lr'], betas=(config['beta1'], config['beta2']),
                     weight_decay=config['weight_decay'])

    # Get learning rate scheduler (supports warmup if enabled)
    scheduler, warmup_scheduler = get_schedulers(config, optimizer)

    # Define loss function based on training mode
    criterion = CLIPLoss() if config['mode'] == 'clip' else torch.nn.CrossEntropyLoss()

    # Initialize trainer class for model training and evaluation
    trainer = Trainer(model=model, optimizer=optimizer, scheduler=scheduler, warmup_scheduler=warmup_scheduler,
                      criterion=criterion, device=device, config=config)

    # Train model
    trainer.train(train_dataloader, val_dataloader, epochs=config['epochs'])

    # Evaluate the model on validation data if not in CLIP mode (for classification)
    if config['mode'] != 'clip':
        trainer.evaluate(val_dataloader, id2target=train_dataset.id2label)


def main():
    """
    Parses command-line arguments, loads configurations, initializes training, and handles logging.
    """
    parser = argparse.ArgumentParser(description="AstroM3")
    parser.add_argument("--config", type=str, required=True, help="Path to the YAML config file")
    parser.add_argument("--random-seed", type=int, default=42, help="Random seed: 42, 0, 66, 12 or 123")
    args = parser.parse_args()

    # Load YAML config file
    config = load_config(args.config)

    # Append timestamp to the weights path to avoid overwriting previous runs
    config['weights_path'] = os.path.join(config['weights_path'], datetime.now().strftime("%Y-%m-%d-%H-%M"))

    # Set random seed for reproducibility
    set_random_seeds(args.random_seed)
    config['random_seed'] = args.random_seed

    # Initialize Weights & Biases (wandb) for logging if enabled
    if config['use_wandb']:
        wandb_run = wandb.init(project=config['project'], config=config)
        config.update(wandb.config)

        config['run_id'] = wandb_run.id
        config['weights_path'] += f'-{wandb_run.id}'
        print(wandb_run.name, config)

    # Create directory for saving model weights if enabled
    if config['save_weights']:
        os.makedirs(config['weights_path'], exist_ok=True)

    # Run training
    run(config)

    # Finish wandb logging session
    if config['use_wandb']:
        wandb.finish()


if __name__ == "__main__":
    main()
