from torch import nn


class MetaClassifier(nn.Module):
    def __init__(self, input_dim=36, hidden_dim=128, num_classes=15, dropout=0.5):
        super(MetaClassifier, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.layer2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.fc(x)

        return x


class MetaClassifierV2(nn.Module):
    def __init__(self, input_dim=36, hidden_dim=128, n_layers=5, num_classes=15, dropout=0.5):
        super(MetaClassifierV2, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        if n_layers - 2 > 0:
            self.layers = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ) for _ in range(n_layers - 2)
            ])
        else:
            self.layers = None

        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = self.layer1(x)

        if self.layers:
            for layer in self.layers:
                x = layer(x)

        x = self.fc(x)

        return x
