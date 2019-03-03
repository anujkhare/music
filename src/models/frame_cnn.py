import torch
import torch.nn.functional as F


class ConvBlock(torch.nn.Module):
    def __init__(self, n_in, n_out, kernel_size, stride, padding):
        super().__init__()

        self.block = torch.nn.Sequential(
            torch.nn.Conv2d(n_in, n_out, kernel_size=kernel_size, stride=stride, padding=padding),
            torch.nn.ReLU(inplace=True),
            torch.nn.BatchNorm2d(n_out),
            torch.nn.Dropout2d(p=0, inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class SimpleFrameCNN(torch.nn.Module):
    """
    Make one prediction per frame. Simplest model possible - used for the onset prediction baseline.
    """

    def __init__(self, n_feats, n_channels_in=1, n_classes=2) -> None:
        super().__init__()

        self.feature_extractor = torch.nn.Sequential(
            ConvBlock(n_channels_in, 16, kernel_size=9, stride=1, padding=4),
            ConvBlock(16, 32, kernel_size=7, stride=1, padding=3),
            ConvBlock(32, 64, kernel_size=5, stride=1, padding=2),
            ConvBlock(64, 128, kernel_size=3, stride=1, padding=1),
        )

        self.classifier = torch.nn.Sequential(
            torch.nn.Conv2d(128, n_classes, kernel_size=(n_feats, 23), stride=1, padding=[0, 11])
        )

    def forward(self, x):
        feats = self.feature_extractor(x)
        probs = F.log_softmax(self.classifier(feats), dim=1)
        return probs


class SimpleFrameCNNWithNotes(torch.nn.Module):
    """
    Make one prediction per frame. Simplest model possible - used for the onset prediction baseline.
    """

    def __init__(self, n_feats, n_channels_in=1, n_note_classes=88) -> None:
        super().__init__()

        self.feature_extractor = torch.nn.Sequential(
            ConvBlock(n_channels_in, 16, kernel_size=9, stride=1, padding=4),
            ConvBlock(16, 32, kernel_size=7, stride=1, padding=3),
            ConvBlock(32, 64, kernel_size=5, stride=1, padding=2),
            ConvBlock(64, 128, kernel_size=3, stride=1, padding=1),
        )

        self.onset_classifier = torch.nn.Sequential(
            torch.nn.Conv2d(128, 2, kernel_size=(n_feats, 23), stride=1, padding=[0, 11])
        )
        self.notes_classifier = torch.nn.Sequential(
            torch.nn.Conv2d(128, n_note_classes, kernel_size=(n_feats, 23), stride=1, padding=[0, 11]),
        )

    def forward(self, x):
        feats = self.feature_extractor(x)
        onset_probs = F.log_softmax(self.onset_classifier(feats), dim=1)
        notes_activations = self.notes_classifier(feats)

        # NOTE: we're not applying the sigmoid for notes since we'll use BCEWithLogitsLoss for numerical stability
        return onset_probs, notes_activations


if __name__ == '__main__':
    import numpy as np
    from src.dataset import SignalWindowDataset

    dataset = SignalWindowDataset(folder_path='../../data',)
    np.random.seed(10)
    datum = dataset[0]

    inputs = datum['features']
    model = SimpleFrameCNNWithNotes(n_feats=513)
    onsets, notes = model(inputs)
    print(onsets.shape, notes.shape)
