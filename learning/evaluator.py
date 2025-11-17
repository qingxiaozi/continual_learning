import torch
import numpy as np
from config.parameters import Config


class ModelEvaluator:
    """模型评估器"""

    def __init__(self, gold_model):
        self.gold_model = gold_model
        self.criterion = torch.nn.CrossEntropyLoss()

    def evaluate_model(self, model, data):
        """评估模型性能"""
        model.eval()
        self.gold_model.eval()

        total_correct = 0
        total_samples = 0
        total_loss = 0.0

        # 检查数据是否为 DataLoader，如果不是则转换
        from torch.utils.data import DataLoader

        if not isinstance(data, DataLoader):
            dataloader = DataLoader(data, batch_size=32, shuffle=False)
        else:
            dataloader = data

        with torch.no_grad():
            for batch in dataloader:
                if isinstance(batch, (list, tuple)):
                    inputs, true_targets = batch
                else:
                    inputs = batch
                    true_targets = self.gold_model(inputs).argmax(dim=1)

                # 修复：确保 true_targets 是 Tensor 且数据类型正确
                if not isinstance(true_targets, torch.Tensor):
                    true_targets = torch.tensor(
                        true_targets, dtype=torch.long, device=inputs.device
                    )

                # 确保目标是 long 类型
                if true_targets.dtype != torch.long:
                    true_targets = true_targets.long()

                if inputs.dim() == 3:
                    # 添加batch维度 [C, H, W] -> [1, C, H, W]
                    inputs = inputs.unsqueeze(0)

                outputs = model(inputs)
                predictions = outputs.argmax(dim=1)

                loss = self.criterion(outputs, true_targets)
                total_loss += loss.item()

                total_correct += (predictions == true_targets).sum().item()
                total_samples += inputs.size(
                    0
                )  # 使用 inputs.size(0) 而不是 len(inputs)

        accuracy = total_correct / total_samples if total_samples > 0 else 0
        avg_loss = total_loss / len(dataloader) if len(dataloader) > 0 else 0

        return accuracy, avg_loss
