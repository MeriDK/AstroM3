import torch
import numpy as np
from tqdm import tqdm
from evaluate import load
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix


class PredictionTrainer:
    def __init__(self, model, optimizer, device):
        self.model = model
        self.optimizer = optimizer
        self.device = device

    def step(self, batch):
        past_times, future_times, past_values, future_values, past_mask, future_mask, _ = batch

        outputs = self.model(
            past_time_features=past_times.to(self.device),
            past_values=past_values.to(self.device),
            future_time_features=future_times.to(self.device),
            future_values=future_values.to(self.device),
            past_observed_mask=past_mask.to(self.device),
            future_observed_mask=future_mask.to(self.device),
        )

        return outputs

    def train_epoch(self, train_dataloader):
        self.model.train()
        total_loss = []

        for idx, batch in enumerate(train_dataloader):
            self.optimizer.zero_grad()

            outputs = self.step(batch)
            loss = outputs.loss
            total_loss.append(loss.item())

            loss.backward()
            self.optimizer.step()

        return sum(total_loss) / len(total_loss)

    def val_epoch(self, val_dataloader):
        self.model.eval()
        total_loss = []

        for idx, batch in enumerate(val_dataloader):
            with torch.no_grad():
                outputs = self.step(batch)
                loss = outputs.loss
                total_loss.append(loss.item())

        return sum(total_loss) / len(total_loss)

    def train(self, train_dataloader, val_dataloader, epochs):

        for epoch in range(epochs):
            train_loss = self.train_epoch(train_dataloader)
            val_loss = self.val_epoch(val_dataloader)

            print(f'Epoch {epoch}: Train Loss {round(train_loss, 4)} Val Loss {round(val_loss, 4)}')

    def get_forecasts(self, val_dataloader):
        forecasts = []

        for idx, batch in enumerate(tqdm(val_dataloader)):
            with torch.no_grad():
                past_times, future_times, past_values, future_values, past_mask, future_mask, label = batch

                outputs = self.model.generate(
                    past_time_features=past_times.to(self.device),
                    past_values=past_values.to(self.device),
                    future_time_features=future_times.to(self.device),
                    past_observed_mask=past_mask.to(self.device),
                )

                forecasts.append(outputs.sequences.cpu().numpy())

        forecasts = np.vstack(forecasts)
        forecast_median = np.median(forecasts, 1)

        return forecast_median

    def get_metrics(self, val_dataset, forecasts):
        mase_metric = load('evaluate-metric/mase')
        smape_metric = load('evaluate-metric/smape')

        mase_metrics = []
        smape_metrics = []

        for i, ts in enumerate(tqdm(val_dataset)):
            _, _, past_values, future_values, _, _, _ = val_dataset[i]

            mase = mase_metric.compute(
                predictions=forecasts[i],
                references=np.array(future_values),
                training=np.array(past_values)
            )
            mase_metrics.append(mase['mase'])

            smape = smape_metric.compute(
                predictions=forecasts[i],
                references=np.array(future_values),
            )
            smape_metrics.append(smape['smape'])

        return np.mean(mase_metrics), np.mean(smape_metrics)

    # TODO combine functions get forecasts and get metrics into one
    def evaluate(self, val_dataloader, val_dataset):
        self.model.eval()

        forecasts = self.get_forecasts(val_dataloader)
        mase, smape = self.get_metrics(val_dataset, forecasts)

        print(f'MASE: {mase} sMAPE: {smape}')


class ClassificationTrainer:
    def __init__(self, model, optimizer, criterion, device):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device

    def train_epoch(self, train_dataloader):
        self.model.train()
        total_loss = []
        total_correct_predictions = 0
        total_predictions = 0

        for batch in train_dataloader:
            past_times, future_times, past_values, future_values, past_mask, future_mask, labels = batch
            labels = labels.to(self.device)

            self.optimizer.zero_grad()

            logits = self.model(past_times, past_values, future_times, past_mask)
            loss = self.criterion(logits, labels)
            total_loss.append(loss.item())

            probabilities = torch.nn.functional.softmax(logits, dim=1)
            _, predicted_labels = torch.max(probabilities, dim=1)
            correct_predictions = (predicted_labels == labels).sum().item()

            total_correct_predictions += correct_predictions
            total_predictions += labels.size(0)

            loss.backward()
            self.optimizer.step()

        return sum(total_loss) / len(total_loss), total_correct_predictions / total_predictions

    def val_epoch(self, val_dataloader):
        self.model.eval()
        total_loss = []
        total_correct_predictions = 0
        total_predictions = 0

        with torch.no_grad():
            for batch in val_dataloader:
                past_times, future_times, past_values, future_values, past_mask, future_mask, labels = batch
                labels = labels.to(self.device)

                logits = self.model(past_times, past_values, future_times, past_mask)
                loss = self.criterion(logits, labels)
                total_loss.append(loss.item())

                probabilities = torch.nn.functional.softmax(logits, dim=1)
                _, predicted_labels = torch.max(probabilities, dim=1)
                correct_predictions = (predicted_labels == labels).sum().item()

                total_correct_predictions += correct_predictions
                total_predictions += labels.size(0)

        return sum(total_loss) / len(total_loss), total_correct_predictions / total_predictions

    def train(self, train_dataloader, val_dataloader, epochs):
        train_losses, val_losses = [], []
        train_accuracies, val_accuracies = [], []

        for epoch in range(epochs):

            train_loss, train_acc = self.train_epoch(train_dataloader)
            train_losses.append(train_loss)
            train_accuracies.append(train_acc)

            val_loss, val_acc = self.val_epoch(val_dataloader)
            val_losses.append(val_loss)
            val_accuracies.append(val_acc)

            print(f'Epoch {epoch}: Train Loss {round(train_loss, 4)} \t Val Loss {round(val_loss, 4)} \t \
                    Train Acc {round(train_acc, 4)} \t Val Acc {round(val_acc, 4)}')

    def evaluate(self, val_dataloader):
        self.model.eval()

        all_true_labels = []
        all_predicted_labels = []

        for batch in val_dataloader:
            with torch.no_grad():
                past_times, future_times, past_values, future_values, past_mask, future_mask, label = batch
                label = label.to(self.device)

                logits = self.model(past_times, past_values, future_times, past_mask)
                probabilities = torch.nn.functional.softmax(logits, dim=1)
                _, predicted_labels = torch.max(probabilities, dim=1)

                all_true_labels.extend(label.cpu().numpy())
                all_predicted_labels.extend(predicted_labels.cpu().numpy())

        conf_matrix = confusion_matrix(all_true_labels, all_predicted_labels)
        plt.figure(figsize=(10, 7))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        plt.show()



