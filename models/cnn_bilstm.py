import torch.nn as nn


class CNNBiLSTM(nn.Module):
    """
    Hybrid CNN-BiLSTM architecture.

    CNN extracts local patterns from the input representation.
    BiLSTM models temporal dependencies.
    """

    def __init__(
        self,
        input_channels=1,
        lstm_hidden_size=128,
        num_classes=10
    ):
        super().__init__()

        self.cnn = nn.Sequential(

            nn.Conv2d(
                input_channels,
                32,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(
                32,
                64,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.lstm = nn.LSTM(
            input_size=64,
            hidden_size=lstm_hidden_size,
            batch_first=True,
            bidirectional=True
        )

        self.classifier = nn.Linear(
            lstm_hidden_size * 2,
            num_classes
        )

    def forward(self, x):

        x = self.cnn(x)

        # Input shape:
        # [batch, channels, frequency, time]

        x = x.mean(dim=2)

        # Shape:
        # [batch, channels, time]

        x = x.transpose(1, 2)

        # Shape:
        # [batch, time, channels]

        x, _ = self.lstm(x)

        x = x[:, -1, :]

        x = self.classifier(x)

        return x
