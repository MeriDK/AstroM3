from torch import nn


class GalSpecNet(nn.Module):

    def __init__(self, num_classes, dropout=0.5):
        super(GalSpecNet, self).__init__()

        self.conv1 = nn.Sequential(nn.Conv1d(1, 64, kernel_size=3), nn.ReLU())
        self.mp1 = nn.MaxPool1d(kernel_size=4)
        self.conv2 = nn.Sequential(nn.Conv1d(64, 64, kernel_size=3), nn.ReLU())
        self.mp2 = nn.MaxPool1d(kernel_size=4)
        self.conv3 = nn.Sequential(nn.Conv1d(64, 32, kernel_size=3), nn.ReLU())
        self.mp3 = nn.MaxPool1d(kernel_size=4)
        self.conv4 = nn.Sequential(nn.Conv1d(32, 32, kernel_size=3), nn.ReLU())

        self.dropout = nn.Dropout(dropout)
        self.mlp = nn.Sequential(
            nn.Linear(32 * 37, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.mp1(x)
        x = self.conv2(x)
        x = self.mp2(x)
        x = self.conv3(x)
        x = self.mp3(x)
        x = self.conv4(x)

        x = x.view(x.shape[0], -1)
        x = self.dropout(x)
        x = self.mlp(x)

        return x


class ResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=5, stride=3, short_cut=True):
        super(ResNetBlock, self).__init__()

        self.short_cut = short_cut
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=1, padding=kernel_size // 2)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size, stride=stride)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.pool = nn.MaxPool1d(kernel_size=kernel_size, stride=1, padding=kernel_size // 2)
        self.activation = nn.ReLU()

        if self.short_cut:
            self.shortcut_conv = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.activation(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.activation(out)
        out = self.pool(out)

        if self.short_cut:
            identity = self.shortcut_conv(identity)

        out += identity

        return out


class ResNet1(nn.Module):
    def __init__(self, num_classes, dropout=0.5):
        super(ResNet1, self).__init__()

        self.model = nn.Sequential(
            ResNetBlock(1, 16),
            ResNetBlock(16, 32),
            ResNetBlock(32, 64),
            ResNetBlock(64, 128),
            ResNetBlock(128, 256),
            ResNetBlock(256, 512)
        )
        self.dropout = nn.Dropout(dropout)
        self.mlp = nn.Sequential(
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.model(x)
        x = x.view(x.shape[0], -1)
        x = self.dropout(x)
        x = self.mlp(x)

        return x


class ResNetV2(nn.Module):
    def __init__(self, num_classes, dropout=0.5):
        super(ResNetV2, self).__init__()

        self.model = nn.Sequential(
            ResNetBlock(1, 16, kernel_size=3, stride=2),
            ResNetBlock(16, 32, kernel_size=3, stride=2),
            ResNetBlock(32, 64, kernel_size=3, stride=2),
            ResNetBlock(64, 128, kernel_size=3, stride=2),
            ResNetBlock(128, 256, kernel_size=3, stride=2),
            ResNetBlock(256, 512, kernel_size=3, stride=2),
            ResNetBlock(512, 1024, kernel_size=3, stride=2)
        )
        self.dropout = nn.Dropout(dropout)
        self.mlp = nn.Sequential(
            nn.Linear(19456, 2048),
            nn.ReLU(),
            nn.Linear(2048, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.model(x)
        x = x.view(x.shape[0], -1)
        x = self.dropout(x)
        x = self.mlp(x)

        return x


class GalSpecNetV2(nn.Module):
    def __init__(self, num_classes, dropout=0.5):
        super(GalSpecNetV2, self).__init__()

        self.conv1 = nn.Sequential(nn.Conv1d(1, 64, kernel_size=3), nn.ReLU())
        self.mp1 = nn.MaxPool1d(kernel_size=4)
        self.conv2 = nn.Sequential(nn.Conv1d(64, 64, kernel_size=3), nn.ReLU())
        self.mp2 = nn.MaxPool1d(kernel_size=4)
        self.conv3 = nn.Sequential(nn.Conv1d(64, 128, kernel_size=3), nn.ReLU())
        self.mp3 = nn.MaxPool1d(kernel_size=4)
        self.conv4 = nn.Sequential(nn.Conv1d(128, 128, kernel_size=3), nn.ReLU())

        self.dropout = nn.Dropout(dropout)
        self.mlp = nn.Sequential(
            nn.Linear(4736, 1024),
            nn.ReLU(),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.mp1(x)
        x = self.conv2(x)
        x = self.mp2(x)
        x = self.conv3(x)
        x = self.mp3(x)
        x = self.conv4(x)

        x = x.view(x.shape[0], -1)
        x = self.dropout(x)
        x = self.mlp(x)

        return x


class GalSpecNetV3(nn.Module):
    def __init__(self, num_classes, dropout=0.5):
        super(GalSpecNetV3, self).__init__()

        self.model = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 64, kernel_size=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=4),

            nn.Conv1d(64, 128, kernel_size=3),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 128, kernel_size=3),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=4),

            nn.Conv1d(128, 256, kernel_size=3),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Conv1d(256, 256, kernel_size=3),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=4)
        )

        self.dropout = nn.Dropout(dropout)
        self.mlp = nn.Sequential(
            nn.Linear(9728, 1024),
            nn.ReLU(),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.model(x)
        x = x.view(x.shape[0], -1)
        x = self.dropout(x)
        x = self.mlp(x)

        return x
