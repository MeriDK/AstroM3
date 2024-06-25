import torch
from torch import nn
import torch.nn.functional as F
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
    def __init__(self, model, optimizer, scheduler, device, config, use_wandb=False):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.save_weights = config['save_weights']
        self.weights_path = config['weights_path']
        self.use_wandb = use_wandb
        self.early_stopping = EarlyStopping(patience=config['early_stopping_patience'])

        self.total_loss = []
        self.total_correct_predictions = 0
        self.total_predictions = 0

    def store_weights(self, epoch):
        torch.save(self.model.state_dict(), os.path.join(self.weights_path, f'weights-{epoch}.pth'))

    def loss_fn(self, logits_ps, logits_sm, logits_mp):
        labels = torch.arange(logits_ps.shape[0], dtype=torch.int64, device=logits_ps.device)

        loss_ps = F.cross_entropy(logits_ps, labels) + F.cross_entropy(logits_ps.transpose(-1, -2), labels)
        loss_sm = F.cross_entropy(logits_sm, labels) + F.cross_entropy(logits_sm.transpose(-1, -2), labels)
        loss_mp = F.cross_entropy(logits_mp, labels) + F.cross_entropy(logits_mp.transpose(-1, -2), labels)

        return loss_ps, loss_sm, loss_mp

    def zero_stats(self):
        self.total_loss = []
        self.total_correct_predictions = 0
        self.total_predictions = 0

    def update_stats(self, loss, logits_ps, logits_sm, logits_mp):
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

    def calculate_stats(self):
        return sum(self.total_loss) / len(self.total_loss), self.total_correct_predictions / self.total_predictions

    def train_epoch(self, train_dataloader):
        self.model.train()
        self.zero_stats()

        for photometry, photometry_mask, spectra, metadata, _ in tqdm(train_dataloader):
            photometry, photometry_mask = photometry.to(self.device), photometry_mask.to(self.device)
            spectra, metadata = spectra.to(self.device), metadata.to(self.device)

            self.optimizer.zero_grad()
            logits_ps, logits_sm, logits_mp = self.model(photometry, photometry_mask, spectra, metadata)
            loss_ps, loss_sm, loss_mp = self.loss_fn(logits_ps, logits_sm, logits_mp)
            loss = loss_ps + loss_sm + loss_mp

            self.update_stats(loss, logits_ps, logits_sm, logits_mp)
            if self.use_wandb:
                wandb.log({'step_loss': loss.item(), 'loss_ps': loss_ps.item(), 'loss_sm': loss_sm.item(),
                           'loss_mp': loss_mp.item()})

            loss.backward()
            self.optimizer.step()

        loss, acc = self.calculate_stats()

        return loss, acc

    def val_epoch(self, val_dataloader):
        self.model.eval()
        self.zero_stats()

        with torch.no_grad():
            for photometry, photometry_mask, spectra, metadata, _ in tqdm(val_dataloader):
                photometry, photometry_mask = photometry.to(self.device), photometry_mask.to(self.device)
                spectra, metadata = spectra.to(self.device), metadata.to(self.device)

                logits_ps, logits_sm, logits_mp = self.model(photometry, photometry_mask, spectra, metadata)
                loss_ps, loss_sm, loss_mp = self.loss_fn(logits_ps, logits_sm, logits_mp)
                loss = loss_ps + loss_sm + loss_mp

                self.update_stats(loss, logits_ps, logits_sm, logits_mp)

        loss, acc = self.calculate_stats()

        return loss, acc

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
