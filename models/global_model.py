import os
import torch
import torch.nn as nn
import torchvision.models as models
from config.parameters import Config
from config.paths import Paths
from learning.evaluator import ModelEvaluator
from torch.utils.data import DataLoader

class GlobalModel(nn.Module):
    """
    全局模型 f_θ，仅负责：
    - 网络结构
    - forward
    - 参数初始化 / 重置
    """
    def __init__(self, dataset_name="office31", init_mode="pretrained"):
        """
        init_mode:
        - "pretrained": reset 时加载全域预训练模型（推荐，论文主实验）
        - "random": reset 时完全随机初始化（对照 / 消融）
        """
        super().__init__()
        self.dataset_name = dataset_name
        self.num_classes = self._get_num_classes(dataset_name)
        self.device = Config.DEVICE
        self.init_mode = init_mode

        self.model_path = os.path.join(
            Paths.RESULTS_DIR, f"global_model_{dataset_name}.pth"
        )

        self.reset_parameters()
    
    def forward(self, x):
        return self.model(x)
    
    def _build_model(self):
        self.model = models.resnet18(pretrained=False)
        self.model.fc = nn.Linear(
            self.model.fc.in_features, self.num_classes
        )
        self.to(self.device)

    def reset_parameters(self):
        """"
        Episode-level reset only.
        Should NOT be called within an episode.
        """
        self._build_model()

        if self.init_mode == "pretrained" and os.path.exists(self.model_path):
            self._load_pretrained()
        elif self.init_mode == "random":
            pass  # 已是随机初始化
        else:
            print("未找到预训练全局模型，使用随机初始化")

    def get_state(self):
        """用于 RL / 联邦同步"""
        return {k: v.detach().cpu() for k, v in self.model.state_dict().items()}

    def set_state(self, state_dict):
        self.model.load_state_dict(state_dict)

    def _get_num_classes(self, dataset):
        return {
            "office31": Config.OFFICE31_CLASSES,
            "digit10": Config.DIGIT10_CLASSES,
            "domainnet": Config.DOMAINNET_CLASSES,
        }.get(dataset, 10)

    def _load_pretrained(self):
        checkpoint = torch.load(self.model_path, map_location=self.device)
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            self.model.load_state_dict(checkpoint["model_state_dict"])
        else:
            self.model.load_state_dict(checkpoint)

    def save_pretrained(self):
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "dataset_name": self.dataset_name,
                "num_classes": self.num_classes,
            },
            self.model_path,
        )

    def pretrain_on_all_domains(
        self,
        train_dataset,
        val_dataset=None,
        epochs=50,
        lr=1e-3,
        batch_size=32,
    ):
        device = self.device
        self._build_model()

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,
        )

        val_loader = (
            DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
            if val_dataset
            else None
        )

        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()
        evaluator = ModelEvaluator()

        best_acc = 0.0

        for epoch in range(epochs):
            self.train()
            total_loss = 0.0

            for x, y in train_loader:
                x, y = x.to(device), y.to(device)

                optimizer.zero_grad()
                out = self(x)
                loss = criterion(out, y)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            print(
                f"[Pretrain] Epoch {epoch+1}/{epochs} "
                f"Train Loss: {total_loss/len(train_loader):.4f}"
            )

            if val_loader:
                val_acc, val_loss = evaluator.evaluate_model(self, val_loader)
                print(
                    f"[Pretrain] Val Acc: {val_acc:.4f}, Val Loss: {val_loss:.4f}"
                )

                if val_acc > best_acc:
                    best_acc = val_acc
                    self.save_pretrained()
            else:
                self.save_pretrained()

# # ==================== 使用示例 ====================
# if __name__ == "__main__":
#     data_simulator = DomainIncrementalDataSimulator()
#     train_datasets = []
#     for domain in data_simulator.domain_sequences[data_simulator.current_dataset]:
#         domain_data = data_simulator._load_domain_data(domain)
#         train_datasets.append(domain_data)

#     full_train_dataset = torch.utils.data.ConcatDataset(train_datasets)
#     train_size = int(0.8 * len(full_train_dataset))
#     val_size = len(full_train_dataset) - train_size
#     train_dataset, val_dataset = torch.utils.data.random_split(full_train_dataset, [train_size, val_size])

#     model = GlobalModel(data_simulator.current_dataset)
#     if not model.load_model():
#         model.fine_tune(train_dataset, val_dataset)
#     else:
#         print("模型已存在，无需训练。")
#         val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)
#         final_val_acc = model.evaluate(val_loader)
#         print(f"已加载模型的验证准确率: {final_val_acc:.2f}%")