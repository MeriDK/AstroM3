import os
import wandb
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from datetime import datetime
import argparse

from data import get_datasets
from loss import CLIPLoss
from trainer import Trainer
from utils import get_schedulers, set_random_seeds, load_config
from model import get_model, load_clip_weights

# Dictionary for pre-trained CLIP weights paths
CLIP_WEIGHTS = {
    42: './weights/2025-02-27-19-55-6jb7me31/weights-best.pt',
    66: './weights/2025-02-27-20-52-6vu1qioj/weights-best.pt',
    0: './weights/2025-02-27-20-52-5ee7tjcc/weights-best.pt',
    12: './weights/2025-02-27-19-55-et132353/weights-best.pt',
    123: './weights/2025-02-27-19-55-ylaxu03p/weights-best.pt'
}


def run(config):
    """
    Prepares and runs the training pipeline: loads the dataset, initializes the model, optimizer,
    scheduler, loss function, and trainer, then trains and evaluates the model.

    Args:
        config (dict): Dictionary containing model, dataset, and training configurations.
    """
    # Load the processed dataset from Hugging Face Hub
    train, val, _ = get_datasets(config['dataset'],
                                 name=f'{config["data_sub"]}_{config["random_seed"]}',
                                 seq_len=config['seq_len'])

    # DataLoader for batch processing
    train_dataloader = DataLoader(train, batch_size=config['batch_size'], shuffle=True,
                                  drop_last=True, num_workers=4)
    val_dataloader = DataLoader(val, batch_size=config['batch_size'], shuffle=False)

    # Initialize the model based on the config
    model = get_model(config)

    # Load CLIP pretrained weights if specified
    if config['use_pretrain']:
        model = load_clip_weights(model, config['pretrain_path'], config['mode'])

    # Set device (GPU if available, otherwise CPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    print('Using', device)

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
        trainer.evaluate(val_dataloader, id2label=train.features['label'].names)


def main():
    """
    Parses command-line arguments, loads configurations, initializes training.
    """
    parser = argparse.ArgumentParser(description="AstroM3")
    parser.add_argument("--config", type=str, required=True, help="Path to the YAML config file")
    parser.add_argument("--random-seed", type=int, default=42, help="Random seed: 42, 0, 66, 12 or 123")
    args = parser.parse_args()

    # Load YAML config file
    config = load_config(args.config)

    # Set random seed for reproducibility
    set_random_seeds(args.random_seed)
    config['random_seed'] = args.random_seed

    # Selects the pre-trained CLIP weights based on the random seed.
    if config['use_pretrain']:
        # If weights exist locally for the given random seed, use them
        if os.path.exists(CLIP_WEIGHTS[config['random_seed']]):
            config['pretrain_path'] = CLIP_WEIGHTS[config['random_seed']]
        else:
            # Otherwise, use the Hugging Face Hub model with the corresponding seed
            config['pretrain_path'] = f"MeriDK/AstroM3-CLIP-{config['random_seed']}"
            print(f"CLIP weights not found at {CLIP_WEIGHTS[config['random_seed']]}.\n"
                  f"Using Hugging Face pretrained weights from {config['pretrain_path']}")

    # Append timestamp to the weights path to avoid overwriting previous runs
    config['weights_path'] = os.path.join(config['weights_path'], datetime.now().strftime("%Y-%m-%d-%H-%M"))

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
