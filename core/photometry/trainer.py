import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import wandb
import os
import json
from scipy import stats
from util.early_stopping import EarlyStopping


class ClassificationTrainer:
    def __init__(self, model, optimizer, scheduler, criterion, device, config, use_wandb=False):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion
        self.device = device
        self.save_weights = config['save_weights']
        self.weights_path = config['weights_path']
        self.context_length = config['seq_len']
        self.data_root = config['data_root']
        self.scales = config['scales']
        self.dataset_class = config['dataset_class']
        self.use_wandb = use_wandb
        self.early_stopping = EarlyStopping(patience=config['early_stopping_patience'])

    def preprocess_batch(self, batch, masks):
        lcs, classes = batch
        lcs_mask, classes_mask = masks

        # shape now [128, 1, 3, 759], make [128, 3, 759]
        X = lcs[:, 0, :, :]

        # change axises, shape now [128, 3, 759], make [128, 759, 3]
        X = X.transpose(1, 2)

        # since mask is the same for time flux and flux err we can make it 2D
        mask = lcs_mask[:, 0, 0, :]

        # context length 200, crop X and MASK if longer, pad if shorter
        if X.shape[1] < self.context_length:
            X_padding = (0, 0, 0, self.context_length - X.shape[1], 0, 0)
            mask_padding = (0, self.context_length - X.shape[1])
            X = F.pad(X, X_padding)
            mask = F.pad(mask, mask_padding, value=True)
        else:
            X = X[:, :self.context_length, :]
            mask = mask[:, :self.context_length]

        # the last dimension is (time, flux, flux_err), sort it based on time
        sort_indices = torch.argsort(X[:, :, 0], dim=1)
        sorted_X = torch.zeros_like(X)

        for i in range(X.shape[0]):
            sorted_X[i] = X[i, sort_indices[i]]

        # rearange indexes for masks as well
        sorted_mask = torch.zeros_like(mask)

        for i in range(mask.shape[0]):
            sorted_mask[i] = mask[i, sort_indices[i]]

        # mask should be 1 for values that are observed and 0 for values that are missing
        sorted_mask = 1 - sorted_mask.int()

        # read scales
        # TODO fix 'v' band to be dynamically set
        if self.scales.endswith('.json'):
            with open(os.path.join(self.data_root, self.scales)) as f:
                s = json.load(f)
                mean, std = s['v']['mean'], s['v']['std']
        elif self.scales == 'mean-std':
            mean, std = X[:, :, 1].mean(), X[:, :, 1].std()
        elif self.scales == 'mean-mad':
            mean = X[:, :, 1].mean()
            std = stats.median_abs_deviation(X[:, :, 1])
        else:
            raise NotImplementedError(f'Unsupported scales {self.scales}')

        # scale X
        sorted_X[:, :, 1] = (sorted_X[:, :, 1] - mean) / std
        sorted_X[:, :, 2] = sorted_X[:, :, 2] / std

        # reshape classes to be 1D vector and convert from float to int
        classes = classes[:, 0]
        classes = classes.long()

        return sorted_X, sorted_mask, classes

    def store_weights(self, epoch):
        torch.save(self.model.state_dict(), os.path.join(self.weights_path, f'weights-{epoch}.pth'))

    def train_epoch(self, train_dataloader):
        self.model.train()
        total_loss = []
        total_correct_predictions = 0
        total_predictions = 0

        for el in tqdm(train_dataloader):
            if self.dataset_class == 'VGDataset':
                X, mask, y = el
            else:
                batch, masks = el
                X, mask, y = self.preprocess_batch(batch, masks)

            X, mask, y = X.to(self.device), mask.to(self.device), y.to(self.device)

            self.optimizer.zero_grad()

            logits = self.model(X, mask)
            loss = self.criterion(logits, y)
            total_loss.append(loss.item())

            if self.use_wandb:
                wandb.log({'step_loss': loss.item()})

            probabilities = torch.nn.functional.softmax(logits, dim=1)
            _, predicted_labels = torch.max(probabilities, dim=1)
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
            for el in tqdm(val_dataloader):
                if self.dataset_class == 'VGDataset':
                    X, mask, y = el
                    X, mask, y = X.to(self.device), mask.to(self.device), y.to(self.device)
                else:
                    batch, masks = el
                    X, mask, y = self.preprocess_batch(batch, masks)
                    X, mask, y = X.to(self.device), mask.to(self.device), y.to(self.device)
                    X = X[:, :, 1:]

                logits = self.model(X, mask)
                loss = self.criterion(logits, y)
                total_loss.append(loss.item())

                probabilities = torch.nn.functional.softmax(logits, dim=1)
                _, predicted_labels = torch.max(probabilities, dim=1)
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

        for el in tqdm(val_dataloader):
            with torch.no_grad():
                if self.dataset_class == 'VGDataset':
                    X, mask, y = el
                    X, mask = X.to(self.device), mask.to(self.device)
                else:
                    batch, masks = el
                    X, mask, y = self.preprocess_batch(batch, masks)
                    X, mask = X.to(self.device), mask.to(self.device)
                    X = X[:, :, 1:]

                logits = self.model(X, mask)
                probabilities = torch.nn.functional.softmax(logits, dim=1)
                _, predicted_labels = torch.max(probabilities, dim=1)

                all_true_labels.extend(y.numpy())
                all_predicted_labels.extend(predicted_labels.cpu().numpy())

        # Calculate confusion matrix
        conf_matrix = confusion_matrix(all_true_labels, all_predicted_labels)

        # Calculate percentage values for confusion matrix
        conf_matrix_percent = 100 * conf_matrix / conf_matrix.sum(axis=1)[:, np.newaxis]

        # Get the labels from the id2target mapping
        labels = [id2target[i] for i in range(len(conf_matrix))]

        # Plot both confusion matrices side by side
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
