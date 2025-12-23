import os
import torch
import torch.nn as nn
import torchvision.models as models
from torch.utils.data import DataLoader
from config.parameters import Config
from config.paths import Paths
from environment.dataSimu_env import DomainIncrementalDataSimulator


class GlobalModel(nn.Module):
    def __init__(self, dataset_name, auto_load=True):
        super().__init__()
        self.dataset_name = dataset_name
        self.num_classes = self._get_num_classes(dataset_name)
        self.device = Config.DEVICE
        self.model = models.resnet18(pretrained=False)
        num_features = self.model.fc.in_features
        self.model.fc = nn.Linear(num_features, self.num_classes)
        self.model_path = os.path.join(
            Paths.RESULTS_DIR, f"global_model_{dataset_name}.pth"
        )
        if auto_load:
            self.load_model()
            print(f"加载已训练的全局模型: {self.model_path}")
        else:
            print(
                f"全局模型不存在，需重训练，运行fine_tune(train_dataset, val_dataset)函数即可"
            )

    def forward(self, x):
        return self.model(x)

    def _get_num_classes(self, dataset_name):
        if dataset_name == "office31":
            return Config.OFFICE31_CLASSES
        elif dataset_name == "digit10":
            return Config.DIGIT10_CLASSES
        elif dataset_name == "domainnet":
            return Config.DOMAINNET_CLASSES
        else:
            return 10

    def load_state_dict(self, state_dict, strict=True):
        """重写load_state_dict以兼容旧格式"""
        if isinstance(state_dict, dict) and 'model_state_dict' in state_dict:
            state_dict = state_dict['model_state_dict']
        super().load_state_dict(state_dict, strict=strict)

    def save_model(self):
        """保存模型"""
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "dataset_name": self.dataset_name,
                "num_classes": self.num_classes,
            },
            self.model_path,
        )

    def load_model(self):
        """加载模型"""
        checkpoint = torch.load(self.model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])

    def fine_tune(self, train_dataset, val_dataset=None, epochs=10, lr=0.001, batch_size=32):
        device = Config.DEVICE
        self.to(device)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4) if val_dataset else None

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

        best_acc = 0.0
        for epoch in range(epochs):
            self.train()
            train_loss = 0.0
            correct = 0
            total = 0
            for inputs, targets in train_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                optimizer.zero_grad()
                outputs = self(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

                _, pred = outputs.max(1)
                total += targets.size(0)
                correct += (pred == targets).sum().item()
                train_loss += loss.item()

            train_acc = 100 * correct / total
            print(f"Epoch [{epoch+1}/{epochs}] Train Loss: {train_loss/len(train_loader):.4f}, Acc: {train_acc:.2f}%")

            if val_loader:
                val_acc = self.evaluate(val_loader)
                print(f"Val Acc: {val_acc:.2f}%")
                if val_acc > best_acc:
                    best_acc = val_acc
                    self.save_model()
            else:
                self.save_model()

        print("✅ 微调完成！")

    def evaluate(self, data_loader):
        self.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, targets in data_loader:
                inputs, targets = inputs.to(Config.DEVICE), targets.to(Config.DEVICE)
                outputs = self(inputs)
                _, pred = outputs.max(1)
                total += targets.size(0)
                correct += (pred == targets).sum().item()
        return 100 * correct / total

    def label_data(self, dataloader):
        self.eval()
        labeled_data = []
        with torch.no_grad():
            for batch in dataloader:
                if isinstance(batch, (list, tuple)) and len(batch) == 2:
                    inputs, labels = batch
                    labeled_data.append((inputs, labels))
                else:
                    inputs = batch[0] if isinstance(batch, (list, tuple)) else batch
                    inputs = inputs.to(Config.DEVICE)
                    outputs = self(inputs)
                    probs = torch.softmax(outputs, dim=1)
                    conf, preds = probs.max(1)
                    mask = conf > 0.8
                    if mask.any():
                        high_conf_inputs = inputs[mask].cpu()
                        high_conf_preds = preds[mask].cpu()
                        labeled_data.append((high_conf_inputs, high_conf_preds))
        return labeled_data


# ==================== 使用示例 ====================
if __name__ == "__main__":
    data_simulator = DomainIncrementalDataSimulator()
    train_datasets = []
    for domain in data_simulator.domain_sequences[data_simulator.current_dataset]:
        domain_data = data_simulator._load_domain_data(domain)
        train_datasets.append(domain_data)

    full_train_dataset = torch.utils.data.ConcatDataset(train_datasets)
    train_size = int(0.8 * len(full_train_dataset))
    val_size = len(full_train_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(full_train_dataset, [train_size, val_size])

    model = GlobalModel(data_simulator.current_dataset)
    if not model.load_model():
        model.fine_tune(train_dataset, val_dataset)
    else:
        print("模型已存在，无需训练。")