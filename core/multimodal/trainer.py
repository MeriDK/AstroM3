import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import wandb
import os
from util.early_stopping import EarlyStopping
from torch.nn.functional import sigmoid


class ClassificationTrainer:
    def __init__(self, model, optimizer, scheduler, criterion, device, config, use_wandb=False):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion
        self.device = device
        self.save_weights = config['save_weights']
        self.weights_path = config['weights_path']
        self.use_wandb = use_wandb
        self.early_stopping = EarlyStopping(patience=config['early_stopping_patience'])

    def store_weights(self, epoch):
        torch.save(self.model.state_dict(), os.path.join(self.weights_path, f'weights-{epoch}.pth'))

    def step(self, el1, el2):
        p, p_mask, s, m = el1
        p2, p_mask2, s2, m2 = el2

        p, p_mask, s, m = p.to(self.device), p_mask.to(self.device), s.to(self.device), m.to(self.device)
        p2, p_mask2, s2, m2 = p2.to(self.device), p_mask2.to(self.device), s2.to(self.device), m2.to(self.device)

        ps_sim, mp_sim, sm_sim = self.model((p, p_mask, s, m), (p2, p_mask2, s2, m2))

        return ps_sim, mp_sim, sm_sim

    def calculate_loss(self, ps_sim, mp_sim, sm_sim, y):
        return self.criterion(ps_sim, y), self.criterion(mp_sim, y), self.criterion(sm_sim, y)

    def train_epoch(self, train_dataloader):
        self.model.train()
        total_loss = []
        total_correct_predictions = 0
        total_predictions = 0

        for el1, el2, y in tqdm(train_dataloader):
            self.optimizer.zero_grad()

            y = y.to(self.device, dtype=torch.float32)
            ps_sim, mp_sim, sm_sim = self.step(el1, el2)
            ps_loss, mp_loss, sm_loss = self.calculate_loss(ps_sim, mp_sim, sm_sim, y)
            loss = ps_loss + mp_loss + sm_loss

            if self.use_wandb:
                wandb.log({'step_loss': loss.item(), 'ps_loss': ps_loss.item(), 'mp_loss': mp_loss.item(),
                           'sm_loss': sm_loss.item()})

            total_loss.append(loss.item())

            probabilities = (sigmoid(ps_sim) + sigmoid(mp_sim) + sigmoid(sm_sim)) / 3
            predicted_labels = (probabilities >= 0.5).float()
            correct_predictions = (predicted_labels == y).sum().item()

            total_correct_predictions += correct_predictions
            total_predictions += y.size(0)

            loss.backward()
            self.optimizer.step()

        return sum(total_loss) / len(total_loss), total_correct_predictions / total_predictions

    def val_epoch(self, val_dataloader):
        self.model.eval()
        total_loss = []
        total_correct_predictions = 0
        total_predictions = 0

        with torch.no_grad():
            for el1, el2, y in tqdm(val_dataloader):
                y = y.to(self.device, dtype=torch.float32)
                ps_sim, mp_sim, sm_sim = self.step(el1, el2)
                ps_loss, mp_loss, sm_loss = self.calculate_loss(ps_sim, mp_sim, sm_sim, y)
                loss = ps_loss + mp_loss + sm_loss
                total_loss.append(loss.item())

                probabilities = (sigmoid(ps_sim) + sigmoid(mp_sim) + sigmoid(sm_sim)) / 3
                predicted_labels = (probabilities >= 0.5).float()
                correct_predictions = (predicted_labels == y).sum().item()

                total_correct_predictions += correct_predictions
                total_predictions += y.size(0)

        return sum(total_loss) / len(total_loss), total_correct_predictions / total_predictions

    def train(self, train_dataloader, val_dataloader, epochs):
        for epoch in range(epochs):
            train_loss, train_acc = self.train_epoch(train_dataloader)
            # train_loss, train_acc = self.val_epoch(train_dataloader)
            val_loss, val_acc = self.val_epoch(val_dataloader)

            self.scheduler.step(val_loss)
            current_lr = self.scheduler.get_last_lr()[0]

            if self.use_wandb:
                wandb.log({'train_loss': train_loss, 'val_loss': val_loss, 'train_acc': train_acc, 'val_acc': val_acc,
                           'learning_rate': current_lr, 'epoch': epoch})

            if self.save_weights:
                self.store_weights(epoch)

            print(f'Epoch {epoch}: Train Loss {round(train_loss, 4)} \t Val Loss {round(val_loss, 4)} \t \
                    Train Acc {round(train_acc, 4)} \t Val Acc {round(val_acc, 4)}')

            if self.early_stopping.step(val_loss):
                print(f'Early stopping at epoch {epoch}')
                break

    def evaluate(self, val_dataloader, id2target):
        self.model.eval()

        all_true_labels = []
        all_predicted_labels = []

        for el1, el2, y in tqdm(val_dataloader):
            with torch.no_grad():
                y = y.to(self.device, dtype=torch.float32)
                ps_sim, mp_sim, sm_sim = self.step(el1, el2)

                probabilities = (sigmoid(ps_sim) + sigmoid(mp_sim) + sigmoid(sm_sim)) / 3
                predicted_labels = (probabilities >= 0.5).float()

                all_true_labels.extend(y.numpy())
                all_predicted_labels.extend(predicted_labels.cpu().numpy())

        # Calculate confusion matrix
        conf_matrix = confusion_matrix(all_true_labels, all_predicted_labels)

        # Calculate percentage values for confusion matrix
        conf_matrix_percent = 100 * conf_matrix / conf_matrix.sum(axis=1)[:, np.newaxis]

        # Plot both confusion matrices side by side
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(20, 7))

        # Plot absolute values confusion matrix
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', ax=axes[0])
        axes[0].set_xlabel('Predicted')
        axes[0].set_ylabel('True')
        axes[0].set_title('Confusion Matrix - Absolute Values')

        # Plot percentage values confusion matrix
        sns.heatmap(conf_matrix_percent, annot=True, fmt='.0f', cmap='Blues', ax=axes[1])
        axes[1].set_xlabel('Predicted')
        axes[1].set_ylabel('True')
        axes[1].set_title('Confusion Matrix - Percentages')

        if self.use_wandb:
            wandb.log({'conf_matrix': wandb.Image(fig)})
