import numpy as np
import random
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
import torch.nn.functional as F
from transformers import TimeSeriesTransformerConfig
from transformers.models.time_series_transformer.modeling_time_series_transformer import TimeSeriesTransformerEncoder
import json
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
import seaborn as sns

from pathlib import Path
from core.multimodal.dataset import collate_fn, ASASSNVarStarDataset
from functools import partial
import matplotlib.pyplot as plt


def preprocess_batch(batch, masks):
    lcs, classes = batch
    lcs_mask, classes_mask = masks

    # shape now [128, 1, 3, 759], make [128, 3, 759]
    X = lcs[:, 0, :, :]

    # change axises, shape now [128, 3, 759], make [128, 759, 3]
    X = X.transpose(1, 2)

    # since mask is the same for time flux and flux err we can make it 2D
    mask = lcs_mask[:, 0, 0, :]

    # context length 200, crop X and MASK if longer, pad if shorter
    if X.shape[1] < context_length:
        X_padding = (0, 0, 0, context_length - X.shape[1], 0, 0)
        mask_padding = (0, context_length - X.shape[1])
        X = F.pad(X, X_padding)
        mask = F.pad(mask, mask_padding, value=True)
    else:
        X = X[:, :context_length, :]
        mask = mask[:, :context_length]

    # the last dimention is (time, flux, flux_err), sort it based on time
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
    with open(datapath / 'scales.json', 'r') as f:
        scales = json.load(f)
        mean, std = scales['v']['mean'], scales['v']['std']

    # scale X
    sorted_X[:, :, 1] = (sorted_X[:, :, 1] - mean) / std
    sorted_X[:, :, 2] = sorted_X[:, :, 2] / std

    # reshape classes to be 1D vector and convert from float to int
    classes = classes[:, 0]
    classes = classes.long()

    return sorted_X, sorted_mask, classes


class CustomModel(nn.Module):
    def __init__(self, encoder, num_classes):
        super(CustomModel, self).__init__()

        self.encoder = encoder
        self.num_classes = num_classes
        self.classifier = nn.Linear(self.encoder.config.d_model, num_classes)

    def forward(self, values, mask):
        encoder_outputs = self.encoder(inputs_embeds=values, attention_mask=mask)
        emb = encoder_outputs.last_hidden_state[:, 0, :]  # we will use the 1 element only, analog to CLS?
        res = self.classifier(emb)

        return res


def train_epoch():
    model.train()

    total_loss = []
    total_correct_predictions = 0
    total_predictions = 0

    for batch, masks in tqdm(train_dataloader):
        X, m, y = preprocess_batch(batch, masks)
        X, m, y = X.to(device), m.to(device), y.to(device)

        optimizer.zero_grad()

        logits = model(X[:, :, 1:], m)
        loss = criterion(logits, y)
        total_loss.append(loss.item())

        probabilities = torch.nn.functional.softmax(logits, dim=1)
        _, predicted_labels = torch.max(probabilities, dim=1)
        correct_predictions = (predicted_labels == y).sum().item()

        total_correct_predictions += correct_predictions
        total_predictions += y.size(0)

        loss.backward()
        optimizer.step()

    print(f'Train Total Loss: {round(sum(total_loss) / len(total_loss), 5)} '
          f'Accuracy: {round(total_correct_predictions / total_predictions, 3)}')

def val_epoch():
    model.eval()

    total_loss = []
    total_correct_predictions = 0
    total_predictions = 0

    with torch.no_grad():
        for batch, masks in tqdm(val_dataloader):
            X, m, y = preprocess_batch(batch, masks)
            X, m, y = X.to(device), m.to(device), y.to(device)

            logits = model(X[:, :, 1:], m)
            loss = criterion(logits, y)
            total_loss.append(loss.item())

            probabilities = torch.nn.functional.softmax(logits, dim=1)
            _, predicted_labels = torch.max(probabilities, dim=1)
            correct_predictions = (predicted_labels == y).sum().item()

            total_correct_predictions += correct_predictions
            total_predictions += y.size(0)

    print(
        f'Val Total Loss: {round(sum(total_loss) / len(total_loss), 5)} '
        f'Accuracy: {round(total_correct_predictions / total_predictions, 3)}')

def plot_confusion(all_true_labels, all_predicted_labels):
    # Calculate confusion matrix
    conf_matrix = confusion_matrix(all_true_labels, all_predicted_labels)

    # Calculate percentage values for confusion matrix
    conf_matrix_percent = 100 * conf_matrix / conf_matrix.sum(axis=1)[:, np.newaxis]

    # Plot both confusion matrices side by side
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(10, 20))

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


random_seed = 42
torch.manual_seed(random_seed)
np.random.seed(random_seed)
random.seed(random_seed)
torch.backends.cudnn.deterministic = True

datapath = Path('data/asaasn')
ds_train = ASASSNVarStarDataset(datapath, mode='train', verbose=True, only_periodic=True, recalc_period=False,
                                prime=True, use_bands=['v'], only_sources_with_spectra=True, return_phased=True,
                                fill_value=0)

ds_val = ASASSNVarStarDataset(datapath, mode='val', verbose=True, only_periodic=True, recalc_period=False, prime=True,
                              use_bands=['v'], only_sources_with_spectra=True, return_phased=True, fill_value=0)

no_spectra_data_keys = ['lcs', 'classes']
no_spectra_collate_fn = partial(collate_fn, data_keys=no_spectra_data_keys, fill_value=0)

train_dataloader = DataLoader(ds_train, batch_size=512, shuffle=True, pin_memory=True, num_workers=8,
                              collate_fn=no_spectra_collate_fn)
val_dataloader = DataLoader(ds_val, batch_size=512, shuffle=False, pin_memory=True, collate_fn=no_spectra_collate_fn)

context_length = 200

config = TimeSeriesTransformerConfig(
    prediction_length=20,    # doesn't matter but it's required by hf
    context_length=context_length,
    num_time_features=1,
    num_static_real_features=0,
    encoder_layers=2,
    d_model=64,
    distribution_output='normal',
    scaling=None,
    dropout=0,
    encoder_layerdrop=0,
    attention_dropout=0,
    activation_dropout=0
)

config.feature_size = 2    # flux and flux err
device = torch.device('cuda:6' if torch.cuda.is_available() else 'cpu')
print('Using', device)

encoder = TimeSeriesTransformerEncoder(config)
model = CustomModel(encoder, num_classes=len(ds_train.target_lookup.keys()))
model = model.to(device)
# model.load_state_dict(torch.load('weights.pth'))

optimizer = Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

for i in range(300):
    print(f'Epoch {i}', end=' ')
    train_epoch()
    val_epoch()

    if i % 50 == 0:
        torch.save(model.state_dict(), f'weights-{i}.pth')

torch.save(model.state_dict(), 'weights.pth')
