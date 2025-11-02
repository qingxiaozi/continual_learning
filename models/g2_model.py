import torch
import torch.nn as nn
import torchvision.models as models
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.parameters import config


class globalModel(nn.Module):
    def __init__(self, num_classes=config.NUM_CLASSES):
        super().__init__()
        self.num_classes = num_classes
        self.model = models.resnet18(pretrained=False)
        self.num_features = self.model.fc.in_features
        self.model.fc = nn.Linear(self.num_features, self.num_classes)

    def forward(self, x):
        y = self.model(x)
        return y

if __name__ == "__main__":
    print(globalModel)
    a = globalModel()
    x = torch.randn(1, 3, 224, 224)
    print(a(x))