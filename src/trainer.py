import numpy as np
import torch
import torch.nn.functional as F
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
from tqdm import tqdm
import wandb
import os

from utils import EarlyStopping


class Trainer:
    def __init__(self, model, optimizer, scheduler, warmup_scheduler, criterion, device, config):
        """
        Trainer class for handling model training, validation, and evaluation.

        Args:
            model (torch.nn.Module): Neural network model.
            optimizer (torch.optim.Optimizer): Optimizer for training.
            scheduler (torch.optim.lr_scheduler): Learning rate scheduler.
            warmup_scheduler (torch.optim.lr_scheduler): Learning rate scheduler for warmup.
            criterion (torch.nn.Module): Loss function.
            device (torch.device): Device (CPU/GPU).
            config (dict): Training configuration dictionary.
        """
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.warmup_scheduler = warmup_scheduler
        self.criterion = criterion
        self.device = device

        # Training configurations
        self.mode = config['mode']
        self.save_weights = config['save_weights']
        self.weights_path = config['weights_path']
        self.use_wandb = config['use_wandb']
        self.early_stopping = EarlyStopping(patience=config['early_stopping_patience'])
        self.warmup_epochs = config['warmup_epochs']
        self.clip_grad = config['clip_grad']
        self.clip_value = config.get('clip_value', 10)

        # Tracking training statistics
        self.total_loss = []
        self.total_correct_predictions = 0
        self.total_predictions = 0

    def store_weights(self, epoch):
        """
        Saves model weights for a given epoch.

        Args:
            epoch (int): Current epoch.
        """
        torch.save(self.model.state_dict(), os.path.join(self.weights_path, f'weights-{epoch}.pth'))
        torch.save(self.model.state_dict(), os.path.join(self.weights_path, f'weights-best.pth'))

    def zero_stats(self):
        """Resets training statistics before each epoch."""

        self.total_loss = []
        self.total_correct_predictions = 0
        self.total_predictions = 0

    def update_stats_clip(self, loss, logits_ps, logits_sm, logits_mp):
        """
        Updates training statistics for CLIP-style pretraining.

        Computes accuracy based on averaged softmax probabilities across modalities.

        Args:
            loss (torch.Tensor): Total CLIP loss.
            logits_ps (torch.Tensor): Logits for Photometry-Spectra alignment.
            logits_sm (torch.Tensor): Logits for Spectra-Metadata alignment.
            logits_mp (torch.Tensor): Logits for Metadata-Photometry alignment.
        """

        labels = torch.arange(logits_ps.shape[0], dtype=torch.int64, device=self.device)

        # Compute probability distributions
        prob_ps = (F.softmax(logits_ps, dim=1) + F.softmax(logits_ps.transpose(-1, -2), dim=1)) / 2
        prob_sm = (F.softmax(logits_sm, dim=1) + F.softmax(logits_sm.transpose(-1, -2), dim=1)) / 2
        prob_mp = (F.softmax(logits_mp, dim=1) + F.softmax(logits_mp.transpose(-1, -2), dim=1)) / 2

        # Compute average probability across modalities
        prob = (prob_ps + prob_sm + prob_mp) / 3

        # Compute accuracy
        _, pred_labels = torch.max(prob, dim=1)
        correct_predictions = (pred_labels == labels).sum().item()

        self.total_correct_predictions += correct_predictions
        self.total_predictions += labels.size(0)
        self.total_loss.append(loss.item())

    def update_stats(self, loss, logits, labels):
        """
        Updates training statistics for classification.

        Args:
            loss (torch.Tensor): Loss value.
            logits (torch.Tensor): Model predictions.
            labels (torch.Tensor): Ground-truth labels.
        """
        probabilities = F.softmax(logits, dim=1)
        _, predicted_labels = torch.max(probabilities, dim=1)
        correct_predictions = (predicted_labels == labels).sum().item()

        self.total_correct_predictions += correct_predictions
        self.total_predictions += labels.size(0)
        self.total_loss.append(loss.item())

    def calculate_stats(self):
        """
        Computes average loss and accuracy.

        Returns:
            Tuple[float, float]: Average loss and accuracy.
        """
        return sum(self.total_loss) / len(self.total_loss), self.total_correct_predictions / self.total_predictions

    def get_logits(self, photometry, photometry_mask, spectra, metadata):
        """
        Runs model forward pass based on selected modality.

        Args:
            photometry, photometry_mask, spectra, metadata: Model inputs.

        Returns:
            torch.Tensor: Logits from the model.
        """
        if self.mode == 'photo':
            logits = self.model(photometry, photometry_mask)
        elif self.mode == 'spectra':
            logits = self.model(spectra)
        elif self.mode == 'meta':
            logits = self.model(metadata)
        else:  # all 3 modalities
            logits = self.model(photometry, photometry_mask, spectra, metadata)

        return logits

    def step_clip(self, photometry, photometry_mask, spectra, metadata):
        """
        Performs a CLIP-style training step.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: Total loss and individual losses.
        """
        logits_ps, logits_sm, logits_mp = self.model(photometry, photometry_mask, spectra, metadata)
        loss_ps, loss_sm, loss_mp = self.criterion(logits_ps, logits_sm, logits_mp)
        loss = loss_ps + loss_sm + loss_mp

        self.update_stats_clip(loss, logits_ps, logits_sm, logits_mp)

        return loss, loss_ps, loss_sm, loss_mp

    def step(self, photometry, photometry_mask, spectra, metadata, labels):
        """
        Performs a classification training step.

        Returns:
            torch.Tensor: Classification loss.
        """
        logits = self.get_logits(photometry, photometry_mask, spectra, metadata)
        loss = self.criterion(logits, labels)

        self.update_stats(loss, logits, labels)

        return loss

    def get_gradient_norm(self):
        """
        Computes total gradient norm.

        Returns:
            float: Gradient norm.
        """
        total_norm = 0.0

        for param in self.model.parameters():
            if param.grad is not None:
                param_norm = param.grad.data.norm(2)
                total_norm += param_norm.item() ** 2

        return total_norm ** 0.5

    def train_epoch(self, train_dataloader):
        """
        Runs one epoch of training.

        Returns:
            Tuple[float, float]: Training loss and accuracy.
        """
        self.model.train()
        self.zero_stats()

        for photometry, photometry_mask, spectra, metadata, labels in tqdm(train_dataloader):
            photometry, photometry_mask = photometry.to(self.device), photometry_mask.to(self.device)
            spectra, metadata, labels = spectra.to(self.device), metadata.to(self.device), labels.to(self.device)

            self.optimizer.zero_grad()

            if self.mode == 'clip':
                loss, loss_ps, loss_sm, loss_mp = self.step_clip(photometry, photometry_mask, spectra, metadata)

                if self.use_wandb:
                    wandb.log({'step_loss': loss.item(), 'loss_ps': loss_ps.item(), 'loss_sm': loss_sm.item(),
                               'loss_mp': loss_mp.item()})
            else:
                loss = self.step(photometry, photometry_mask, spectra, metadata, labels)

                if self.use_wandb:
                    wandb.log({'step_loss': loss.item()})

            loss.backward()

            if self.use_wandb:
                grad_norm = self.get_gradient_norm()
                wandb.log({'grad_norm': grad_norm})

            if self.clip_grad:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_value)

                if self.use_wandb:
                    clip_grad_norm = self.get_gradient_norm()
                    wandb.log({'clip_grad_norm': clip_grad_norm})

            self.optimizer.step()

        loss, acc = self.calculate_stats()

        return loss, acc

    def val_epoch(self, val_dataloader):
        """
        Runs one epoch of validation.

        Returns:
            Tuple[float, float]: Validation loss and accuracy.
        """
        self.model.eval()
        self.zero_stats()

        with torch.no_grad():
            for photometry, photometry_mask, spectra, metadata, labels in tqdm(val_dataloader):
                photometry, photometry_mask = photometry.to(self.device), photometry_mask.to(self.device)
                spectra, metadata, labels = spectra.to(self.device), metadata.to(self.device), labels.to(self.device)

                if self.mode == 'clip':
                    self.step_clip(photometry, photometry_mask, spectra, metadata)
                else:
                    self.step(photometry, photometry_mask, spectra, metadata, labels)

        loss, acc = self.calculate_stats()

        return loss, acc

    def train(self, train_dataloader, val_dataloader, epochs):
        """
        Train the model for a specified number of epochs.

        This function iterates through multiple epochs, performing training and validation.
        It logs metrics to Weights & Biases (wandb) if enabled and applies early stopping.
        The best model weights are saved based on validation accuracy.

        Args:
            train_dataloader (torch.utils.data.DataLoader): Dataloader for the training set.
            val_dataloader (torch.utils.data.DataLoader): Dataloader for the validation set.
            epochs (int): Number of training epochs.

        Returns:
            float: Best validation loss achieved during training.
        """
        best_val_loss = np.inf
        best_val_acc = 0

        for epoch in range(epochs):
            train_loss, train_acc = self.train_epoch(train_dataloader)
            val_loss, val_acc = self.val_epoch(val_dataloader)

            best_val_loss = min(val_loss, best_val_loss)

            # Learning rate update
            if self.warmup_scheduler and epoch < self.warmup_epochs:
                self.warmup_scheduler.step()
                current_lr = self.warmup_scheduler.get_last_lr()[0]
            else:
                self.scheduler.step(val_loss)
                current_lr = self.scheduler.get_last_lr()[0]

            # Log training progress
            if self.use_wandb:
                wandb.log({'train_loss': train_loss, 'val_loss': val_loss, 'train_acc': train_acc, 'val_acc': val_acc,
                           'learning_rate': current_lr, 'epoch': epoch})

            # Save best model based on validation accuracy
            if best_val_acc < val_acc:
                best_val_acc = val_acc

                if self.use_wandb:
                    wandb.log({'best_val_acc': best_val_acc})

                if self.save_weights:
                    self.store_weights(epoch)

            print(f'Epoch {epoch}: Train Loss {round(train_loss, 4)} \t Val Loss {round(val_loss, 4)} \t \
                    Train Acc {round(train_acc, 4)} \t Val Acc {round(val_acc, 4)}')

            if self.early_stopping.step(val_loss):
                print(f'Early stopping at epoch {epoch}')
                break

        return best_val_loss

    def evaluate(self, val_dataloader, id2target):
        """
        Evaluate the model on the validation dataset.
        This function computes and logs (if wandb is enabled) absolute and percentage-based confusion matrices.

        Args:
            val_dataloader (torch.utils.data.DataLoader): Dataloader for the validation set.
            id2target (dict): Mapping from numerical class indices to class names.

        Returns:
            np.ndarray: The computed absolute confusion matrix.
        """
        self.model.eval()

        all_true_labels = []
        all_predicted_labels = []

        # Iterate over the validation set
        for photometry, photometry_mask, spectra, metadata, labels in tqdm(val_dataloader):
            with torch.no_grad():
                photometry, photometry_mask = photometry.to(self.device), photometry_mask.to(self.device)
                spectra, metadata = spectra.to(self.device), metadata.to(self.device)

                logits = self.get_logits(photometry, photometry_mask, spectra, metadata)
                probabilities = torch.nn.functional.softmax(logits, dim=1)
                _, predicted_labels = torch.max(probabilities, dim=1)

                all_true_labels.extend(labels.numpy())
                all_predicted_labels.extend(predicted_labels.cpu().numpy())

        # Compute confusion matrices
        conf_matrix = confusion_matrix(all_true_labels, all_predicted_labels)
        conf_matrix_percent = 100 * conf_matrix / conf_matrix.sum(axis=1)[:, np.newaxis]

        labels = [id2target[i] for i in range(len(conf_matrix))]

        # Create plots for absolute and percentage-based confusion matrices
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(20, 7))

        # Plot absolute values confusion matrix
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels, ax=axes[0])
        axes[0].set_xlabel('Predicted')
        axes[0].set_ylabel('True')
        axes[0].set_title('Confusion Matrix - Absolute Values')

        # Plot percentage values confusion matrix
        sns.heatmap(conf_matrix_percent, annot=True, fmt='.0f', cmap='Blues', xticklabels=labels, yticklabels=labels,
                    ax=axes[1])
        axes[1].set_xlabel('Predicted')
        axes[1].set_ylabel('True')
        axes[1].set_title('Confusion Matrix - Percentages')

        # Log to wandb
        if self.use_wandb:
            wandb.log({'conf_matrix': wandb.Image(fig)})

        return conf_matrix
