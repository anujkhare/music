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
    def __init__(self, n_feats, n_channels_in=1, n_classes=2,) -> None:
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