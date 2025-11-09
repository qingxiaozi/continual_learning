import torch
import torch.nn as nn
import torchvision.models as models
from config.parameters import Config


class globalModel(nn.Module):
    def __init__(self, dataset_name):
        super().__init__()
        self.num_classes = self._get_num_classes(dataset_name)
        self.model = models.resnet18(pretrained=False)
        self.num_features = self.model.fc.in_features
        self.model.fc = nn.Linear(self.num_features, self.num_classes)

    def forward(self, x):
        y = self.model(x)
        return y

    def _get_num_classes(self, dataset_name):
        """根据数据集名称获取类别数量"""
        if dataset_name == "office31":
            return Config.OFFICE31_CLASSES
        elif dataset_name == "digit10":
            return Config.DIGIT10_CLASSES
        elif dataset_name == "domainnet":
            return Config.DOMAINNET_CLASSES
        else:
            return 10  # 默认值


if __name__ == "__main__":
    print(globalModel)
    a = globalModel("office31")
    x = torch.randn(1, 3, 224, 224)
    print(a(x))