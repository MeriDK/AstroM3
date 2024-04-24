import torch.nn as nn


class ClassificationModel(nn.Module):
    def __init__(self, encoder, num_classes):
        super(ClassificationModel, self).__init__()

        self.encoder = encoder
        self.num_classes = num_classes
        self.classifier = nn.Linear(self.encoder.config.d_model, num_classes)

    def forward(self, values, mask):
        encoder_outputs = self.encoder(inputs_embeds=values, attention_mask=mask)
        emb = encoder_outputs.last_hidden_state[:, 0, :]  # we will use the 1 element only, analog to CLS?
        res = self.classifier(emb)

        return res