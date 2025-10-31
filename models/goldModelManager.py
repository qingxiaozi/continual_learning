import torch
import torch.nn as nn
import torchvision.models as models
from torch.utils.data import DataLoader
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.parameters import config
from config.paths import paths
from environment.dataSimu_env import DomainIncrementalDataSimulator

class GoldenModelManager:
    """
    黄金模型管理器

    功能：
    1. 加载预训练的ResNet18作为基础模型
    2. 根据任务需求微调模型
    3. 管理黄金模型的保存和加载
    4. 为无标签数据生成高质量标签
    """

    def __init__(self, dataset_name):
        self.dataset_name = dataset_name
        self.num_classes = self._get_num_classes(dataset_name)
        self.device = config.DEVICE

        # 初始化模型
        self.model = self._initialize_model()

        # 训练参数
        self.fine_tune_epochs = 10
        self.learning_rate = 0.001
        self.batch_size = 32

        # 模型路径
        self.model_path = os.path.join(paths.RESULTS_DIR, f'golden_model_{dataset_name}.pth')

        # 如果模型已存在，直接加载；否则进行微调
        if os.path.exists(self.model_path):
            self.load_model()
            print(f"加载已训练的黄金模型: {self.model_path}")
        else:
            print(f"未找到预训练黄金模型，需要先进行微调")

    def _get_num_classes(self, dataset_name):
        """根据数据集名称获取类别数量"""
        if dataset_name == 'office31':
            return config.OFFICE31_CLASSES
        elif dataset_name == 'digit10':
            return config.DIGIT10_CLASSES
        elif dataset_name == 'domainnet':
            return config.DOMAINNET_CLASSES
        else:
            return 10  # 默认值

    def _initialize_model(self):
        """初始化ResNet18模型"""
        # 加载预训练的ResNet18
        model = models.resnet18(pretrained = True)

        # 冻结前面的层（可选，根据数据量决定）
        # self._freeze_layers(model)

        # 替换最后的全连接层以适应我们的任务
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, self.num_classes)

        # 移动到设备
        model = model.to(self.device)

        return model

    def _freeze_layers(self, model):
        """冻结模型的前面层，只训练最后几层"""
        # 冻结所有层
        for param in model.parameters():
            param.requires_grad = False

        # 只解冻最后两层
        for param in model.layer4.parameters():
            param.requires_grad = True
        for param in model.fc.parameters():
            param.requires_grad = True

    def fine_tune(self, train_dataset, val_dataset = None):
        """
        微调黄金模型
        参数：
            train_dataset：训练数据集
            val_dataset：验证数据集（可选）
        """
        print(f"开始微调黄金模型，数据集：{self.dataset_name}")
        print(f"类别数：{self.num_classes}")
        print(f"训练集样本数：{len(train_dataset)}")
        print(f"测试集样本数：{len(val_dataset)}")

        # 创建数据加载器
        train_loader = DataLoader(
            train_dataset,
            batch_size = self.batch_size,
            shuffle = True,
            num_workers = 4
        )
        val_loader = None
        if val_dataset:
            val_loader = DataLoader(
                val_dataset,
                batch_size = self.batch_size,
                shuffle = False,
                num_workers = 4
            )

        # 定义损失函数和优化器
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr = self.learning_rate
        )

        # 学习率调度器
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
        # 训练循环
        best_val_accuracy = 0.0

        for epoch in range(self.fine_tune_epochs):
            # 训练阶段
            self.model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0

            for batch_idx, (inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                _, predicted = outputs.max(1)
                train_total += targets.size(0)
                train_correct += predicted.eq(targets).sum().item()

            # 计算训练准确率
            train_accuracy = 100. * train_correct / train_total
            avg_train_loss = train_loss / len(train_loader)

            # 验证阶段
            val_accuracy = 0.0
            if val_loader:
                self.model.eval()
                val_correct = 0
                val_total = 0

                with torch.no_grad():
                    for inputs, targets in val_loader:
                        inputs, targets = inputs.to(self.device), targets.to(self.device)
                        outputs = self.model(inputs)
                        _, predicted = outputs.max(1)
                        val_total += targets.size(0)
                        val_correct += predicted.eq(targets).sum().item()

                val_accuracy = 100. * val_correct / val_total

            # 更新学习率
            scheduler.step()
            # 打印进度
            print(f'Epoch [{epoch+1}/{self.fine_tune_epochs}] '
                  f'Train Loss: {avg_train_loss:.4f}, '
                  f'Train Acc: {train_accuracy:.2f}%', end='')

            if val_loader:
                print(f', Val Acc: {val_accuracy:.2f}%')

                # 保存最佳模型
                if val_accuracy > best_val_accuracy:
                    best_val_accuracy = val_accuracy
                    self.save_model()
            else:
                print()

        # 如果没有验证集，在训练结束后保存模型
        if not val_loader:
            self.save_model()

        print(f"黄金模型微调完成，保存到: {self.model_path}")

        # 最终验证（如果有验证集）
        if val_loader:
            final_val_accuracy = self.evaluate(val_loader)
            print(f"最终验证准确率: {final_val_accuracy:.2f}%")

    def predict(self, inputs):
        """为输入数据生成预测"""
        self.model.eval()

        with torch.no_grad():
            if isinstance(inputs, list) or (isinstance(inputs, torch.Tensor) and inputs.dim() == 3):
                inputs = inputs.unsqueeze(0)

            inputs = inputs.to(self.device)
            outputs = self.model(inputs)
            probabilities = torch.softmax(outputs, dim=1)
            confidence, predictions = torch.max(probabilities, 1)

        return predictions.cpu(), confidence.cpu()

    def save_model(self):
        """保存模型"""
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'dataset_name': self.dataset_name,
            'num_classes': self.num_classes
        }, self.model_path)

    def load_model(self):
        """加载模型"""
        checkpoint = torch.load(self.model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        print(f"黄金模型加载成功，数据集: {checkpoint['dataset_name']}, "
              f"类别数: {checkpoint['num_classes']}")

    def get_model_info(self):
        """获取模型信息"""
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        return {
            'model_name': 'ResNet18',
            'dataset': self.dataset_name,
            'num_classes': self.num_classes,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'model_path': self.model_path
        }

    def label_data(self, dataloader):
        """
        为无标签数据生成标签

        参数:
            dataloader: 包含无标签数据的DataLoader

        返回:
            list: 包含(数据, 标签)的元组列表
        """
        self.model.eval()
        labeled_data = []

        with torch.no_grad():
            for batch in dataloader:
                if isinstance(batch, (list, tuple)) and len(batch) == 2:
                    # 有标签数据，直接使用
                    inputs, true_labels = batch
                    labeled_data.append((inputs, true_labels))
                else:
                    # 无标签数据，使用模型预测
                    inputs = batch[0] if isinstance(batch, (list, tuple)) else batch
                    inputs = inputs.to(self.device)

                    outputs = self.model(inputs)
                    probabilities = torch.softmax(outputs, dim=1)
                    confidence, predictions = torch.max(probabilities, 1)

                    # 只保留高置信度的预测
                    high_conf_mask = confidence > 0.8  # 置信度阈值
                    if high_conf_mask.any():
                        high_conf_inputs = inputs[high_conf_mask].cpu()
                        high_conf_predictions = predictions[high_conf_mask].cpu()
                        labeled_data.append((high_conf_inputs, high_conf_predictions))

        return labeled_data

    def evaluate(self, data_loader):
        """评估模型性能"""
        self.model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, targets in data_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

        accuracy = 100. * correct / total
        return accuracy


if __name__ == "__main__":
    data_simulator = DomainIncrementalDataSimulator()
    train_datasets = []
    for domain in data_simulator.domain_sequences[data_simulator.current_dataset]:
        domain_data = data_simulator._load_domain_data(domain)
        train_datasets.append(domain_data)

    # 合并所有域的数据
    from torch.utils.data import ConcatDataset
    full_train_dataset = ConcatDataset(train_datasets)
    # 分割训练集和验证集
    train_size = int(0.8 * len(full_train_dataset))
    val_size = len(full_train_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_train_dataset, [train_size, val_size]
    )
    golden_model = GoldenModelManager(data_simulator.current_dataset)
    # golden_model.fine_tune(train_dataset, val_dataset)