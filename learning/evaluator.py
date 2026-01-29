import torch
import numpy as np
from config.parameters import Config
from torch.utils.data import DataLoader

class ModelEvaluator:
    """模型评估器：给定模型和带标签数据，返回准确率和损失"""

    def __init__(self):
        self.criterion = torch.nn.CrossEntropyLoss()

    def evaluate_model(self, model, dataloader):
        """
        评估模型性能
        Args:
            model: PyTorch 模型
            dataloader: 包含 (inputs, labels) 的 DataLoader
        Returns:
            dict: {"accuracy": float, "loss": float}
        """
        model.eval()
        total_loss = 0.0
        correct_predictions = 0
        total_samples = 0

        with torch.no_grad():
            for inputs, labels in dataloader:
                inputs, labels = inputs.to(Config.DEVICE), labels.to(Config.DEVICE)
                outputs = model(inputs)
                loss = self.criterion(outputs, labels)
                total_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                correct_predictions += (predicted == labels).sum().item()
                total_samples += inputs.size(0)

        average_loss = total_loss / total_samples
        accuracy = correct_predictions / total_samples

        return accuracy, average_loss
