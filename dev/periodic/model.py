import torch.nn as nn
from transformers import TimeSeriesTransformerForPrediction


class ClassificationModel(nn.Module):
    def __init__(self, device, pretrained_model=None, pretrained_model_path=None, num_labels=8):
        super(ClassificationModel, self).__init__()

        assert (pretrained_model is None) != (pretrained_model_path is None), \
            'You must provide either pretrained_model or pretrained_model_path, but not both'

        if pretrained_model:
            self.pretrained_model = pretrained_model
        else:
            self.pretrained_model = TimeSeriesTransformerForPrediction.from_pretrained(pretrained_model_path)
            print(f'Loaded TimeSeriesTransformer from {pretrained_model_path}')

        self.pretrained_model.to(device)
        self.device = device

        self.classifier = nn.Linear(self.pretrained_model.config.d_model, num_labels)
        self.classifier.to(self.device)

    def forward(self, past_times, past_values, future_times, past_mask, aux):
        static_real_features = aux.to(self.device) if self.pretrained_model.config.num_static_real_features else None

        outputs = self.pretrained_model(
            past_time_features=past_times.to(self.device),
            past_values=past_values.to(self.device),
            future_time_features=future_times.to(self.device),
            past_observed_mask=past_mask.to(self.device),
            static_real_features=static_real_features
        )

        # embedding = torch.mean(outpu  ts.encoder_last_hidden_state, dim=1)
        embedding = outputs.encoder_last_hidden_state[:, 0, :]
        logits = self.classifier(embedding)

        return logits
