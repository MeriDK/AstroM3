import numpy as np
import torch
import torch.nn.functional as F
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
from tqdm import tqdm
import wandb
import os
import optuna

from util.early_stopping import EarlyStopping


class Trainer:
    def __init__(self, model, optimizer, scheduler, warmup_scheduler, criterion, device, config, trial=None):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.warmup_scheduler = warmup_scheduler
        self.criterion = criterion
        self.device = device
        self.trial = trial

        self.mode = config['mode']
        self.save_weights = config['save_weights']
        self.weights_path = config['weights_path']
        self.use_wandb = config['use_wandb']
        self.early_stopping = EarlyStopping(patience=config['early_stopping_patience'])
        self.warmup_epochs = config['warmup_epochs']
        self.clip_grad = config['clip_grad']
        self.clip_value = config['clip_value']

        self.total_loss = []
        self.total_correct_predictions = 0
        self.total_predictions = 0

    def store_weights(self, epoch):
        torch.save(self.model.state_dict(), os.path.join(self.weights_path, f'weights-{epoch}.pth'))
        torch.save(self.model.state_dict(), os.path.join(self.weights_path, f'weights-best.pth'))

    def zero_stats(self):
        self.total_loss = []
        self.total_correct_predictions = 0
        self.total_predictions = 0

    def update_stats_clip(self, loss, logits_ps, logits_sm, logits_mp):
        labels = torch.arange(logits_ps.shape[0], dtype=torch.int64, device=self.device)

        prob_ps = (F.softmax(logits_ps, dim=1) + F.softmax(logits_ps.transpose(-1, -2), dim=1)) / 2
        prob_sm = (F.softmax(logits_sm, dim=1) + F.softmax(logits_sm.transpose(-1, -2), dim=1)) / 2
        prob_mp = (F.softmax(logits_mp, dim=1) + F.softmax(logits_mp.transpose(-1, -2), dim=1)) / 2
        prob = (prob_ps + prob_sm + prob_mp) / 3

        _, pred_labels = torch.max(prob, dim=1)
        correct_predictions = (pred_labels == labels).sum().item()

        self.total_correct_predictions += correct_predictions
        self.total_predictions += labels.size(0)
        self.total_loss.append(loss.item())

    def update_stats(self, loss, logits, labels):
        probabilities = torch.nn.functional.softmax(logits, dim=1)
        _, predicted_labels = torch.max(probabilities, dim=1)
        correct_predictions = (predicted_labels == labels).sum().item()

        self.total_correct_predictions += correct_predictions
        self.total_predictions += labels.size(0)
        self.total_loss.append(loss.item())

    def calculate_stats(self):
        return sum(self.total_loss) / len(self.total_loss), self.total_correct_predictions / self.total_predictions

    def get_logits(self, photometry, photometry_mask, spectra, metadata):
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
        """Perform a training step for the CLIP pretraining model"""

        logits_ps, logits_sm, logits_mp = self.model(photometry, photometry_mask, spectra, metadata)
        loss_ps, loss_sm, loss_mp = self.criterion(logits_ps, logits_sm, logits_mp)
        loss = loss_ps + loss_sm + loss_mp

        self.update_stats_clip(loss, logits_ps, logits_sm, logits_mp)

        return loss, loss_ps, loss_sm, loss_mp

    def step(self, photometry, photometry_mask, spectra, metadata, labels):
        """Perform a training step for the classification model"""

        logits = self.get_logits(photometry, photometry_mask, spectra, metadata)
        loss = self.criterion(logits, labels)

        self.update_stats(loss, logits, labels)

        return loss

    def get_gradient_norm(self):
        total_norm = 0.0

        for param in self.model.parameters():
            if param.grad is not None:
                param_norm = param.grad.data.norm(2)
                total_norm += param_norm.item() ** 2

        return total_norm ** 0.5

    def train_epoch(self, train_dataloader):
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
        best_val_loss = np.inf
        best_val_acc = 0

        for epoch in range(epochs):
            train_loss, train_acc = self.train_epoch(train_dataloader)
            val_loss, val_acc = self.val_epoch(val_dataloader)

            best_val_loss = min(val_loss, best_val_loss)

            if self.trial:
                self.trial.report(val_loss, epoch)

                if self.trial.should_prune():
                    print('Prune')
                    wandb.finish()
                    raise optuna.exceptions.TrialPruned()

            if self.warmup_scheduler and epoch < self.warmup_epochs:
                self.warmup_scheduler.step()
                current_lr = self.warmup_scheduler.get_last_lr()[0]
            else:
                self.scheduler.step(val_loss)
                current_lr = self.scheduler.get_last_lr()[0]

            if self.use_wandb:
                wandb.log({'train_loss': train_loss, 'val_loss': val_loss, 'train_acc': train_acc, 'val_acc': val_acc,
                           'learning_rate': current_lr, 'epoch': epoch})

            if self.save_weights and best_val_acc < val_acc:
                self.store_weights(epoch)
                best_val_acc = val_acc

                if self.use_wandb:
                    wandb.log({'step_loss': best_val_acc})

            print(f'Epoch {epoch}: Train Loss {round(train_loss, 4)} \t Val Loss {round(val_loss, 4)} \t \
                    Train Acc {round(train_acc, 4)} \t Val Acc {round(val_acc, 4)}')

            if self.early_stopping.step(val_loss):
                print(f'Early stopping at epoch {epoch}')
                break

        return best_val_loss

    def evaluate(self, val_dataloader, id2target):
        self.model.eval()

        all_true_labels = []
        all_predicted_labels = []

        for photometry, photometry_mask, spectra, metadata, labels in tqdm(val_dataloader):
            with torch.no_grad():
                photometry, photometry_mask = photometry.to(self.device), photometry_mask.to(self.device)
                spectra, metadata = spectra.to(self.device), metadata.to(self.device)

                logits = self.get_logits(photometry, photometry_mask, spectra, metadata)
                probabilities = torch.nn.functional.softmax(logits, dim=1)
                _, predicted_labels = torch.max(probabilities, dim=1)

                all_true_labels.extend(labels.numpy())
                all_predicted_labels.extend(predicted_labels.cpu().numpy())

        conf_matrix = confusion_matrix(all_true_labels, all_predicted_labels)
        conf_matrix_percent = 100 * conf_matrix / conf_matrix.sum(axis=1)[:, np.newaxis]

        labels = [id2target[i] for i in range(len(conf_matrix))]
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

        if self.use_wandb:
            wandb.log({'conf_matrix': wandb.Image(fig)})

        return conf_matrix
