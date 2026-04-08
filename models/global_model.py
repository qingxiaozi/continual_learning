import logging
import os
import torch
import torch.nn as nn
import torchvision.models as models
from config.parameters import Config
from config.paths import Paths
from learning.evaluator import ModelEvaluator
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)

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
            logger.info("未找到预训练全局模型，使用随机初始化")
    
    def _build_model(self):
        self.model = models.resnet18(pretrained=False)
        if self.dataset_name == "office31":
            self.model.fc = nn.Sequential(
                nn.Dropout(p=0.5),
                nn.Linear(self.model.fc.in_features, self.num_classes)
            )
        else:
            self.model.fc = nn.Linear(
                self.model.fc.in_features, self.num_classes
            )
        self.to(self.device)
        logger.info(f"device:{self.device}")
        
    def forward(self, x):
        return self.model(x)

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
        data_simulator,
        domains,
        epochs=15,  # 15 epochs for digit10, 30 epochs for office31, 45 epochs for domainnet
        lr=1e-4,  # office31: 1e-4, digit10: 1e-3, domainnet: 1e-2
        batch_size=64,
    ):
        device = self.device
        self._build_model()

        visualizer_path = Paths.get_visualizer_path()
        visualizer_path = visualizer_path.replace("\\", "/")
        os.makedirs(os.path.dirname(visualizer_path), exist_ok=True) if os.path.dirname(visualizer_path) else None

        os.makedirs(Paths.RESULTS_DIR, exist_ok=True)

        train_datasets = []
        val_datasets = []
        for domain in domains:
            data_simulator._preload_domain_data(domain)
            domain_key = data_simulator._get_domain_key(domain)
            init_data = data_simulator.init_domain_cache.get(domain_key)

            if init_data is None:
                logger.warning(f"Domain {domain} init data not found, skipping...")
                continue

            train_datasets.append(init_data["train"])
            val_datasets.append(init_data["val"])

        from torch.utils.data import ConcatDataset
        combined_train_dataset = ConcatDataset(train_datasets)
        combined_val_dataset = ConcatDataset(val_datasets)

        train_loader = DataLoader(
            combined_train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
        )
        val_loader = DataLoader(
            combined_val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
        )

        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()

        best_acc = 0.0
        train_losses = []
        val_losses = []
        val_accuracies = []

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

            avg_train_loss = total_loss / len(train_loader)
            train_losses.append(avg_train_loss)

            self.eval()
            val_loss = 0.0
            correct = 0
            total = 0
            with torch.no_grad():
                for x, y in val_loader:
                    x, y = x.to(device), y.to(device)
                    out = self(x)
                    loss = criterion(out, y)
                    val_loss += loss.item()
                    _, predicted = out.max(1)
                    total += y.size(0)
                    correct += predicted.eq(y).sum().item()

            avg_val_loss = val_loss / len(val_loader)
            val_acc = 100.0 * correct / total
            val_losses.append(avg_val_loss)
            val_accuracies.append(val_acc)

            print(
                f"[Pretrain All Domains] Epoch {epoch+1}/{epochs} "
                f"Train Loss: {avg_train_loss:.4f} "
                f"Val Loss: {avg_val_loss:.4f} "
                f"Val Acc: {val_acc:.2f}%"
            )

            if val_acc > best_acc:
                best_acc = val_acc

        self.save_pretrained()

        from utils.visualizer import ResultVisualizer
        visualizer = ResultVisualizer(save_dir=os.path.dirname(visualizer_path) or "./")

        loss_plot_name = f"{self.dataset_name}_training_loss_all_domains.png"
        visualizer.plot_training_loss(
            epoch_losses=train_losses,
            val_losses=val_losses,
            save_plot=True,
            plot_name=loss_plot_name,
        )

        acc_plot_name = f"{self.dataset_name}_accuracy_curve_all_domains.png"
        visualizer.plot_accuracy_curve(
            val_accuracies=val_accuracies,
            save_plot=True,
            plot_name=acc_plot_name,
        )

        print(f"[Pretrain] All domains trained. Model saved to {self.model_path}")


def pretrain_initial_models():
    """预训练各域初始模型的入口函数"""
    from config.parameters import Config
    from environment.dataSimu_env import DomainIncrementalDataSimulator

    print("\n=== Pretraining Initial Models ===")
    Config.set_seed()

    data_simulator = DomainIncrementalDataSimulator()
    domains = Config.DOMAIN_SEQUENCES[Config.CURRENT_DATASET]
    print(f"Dataset: {Config.CURRENT_DATASET}")
    print(f"Domains: {domains}")
    # print(f"Batch size: 64, Epochs: 45")
    print("-" * 50)

    model = GlobalModel(dataset_name=Config.CURRENT_DATASET)
    model.pretrain_on_all_domains(
        data_simulator=data_simulator,
        domains=domains,
        epochs=10,  # 10 for digit10, 10 for office31, 10 epochs for domainnet
        lr=1e-4,  # office31: 1e-4, digit10: 5e-4, domainnet: 1e-4
        batch_size=64,
    )

    print("-" * 50)
    print(f"All initial models trained!")
    print(f"Models saved to: results/global_model_{Config.CURRENT_DATASET}.pth")
    return True


if __name__ == "__main__":
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    pretrain_initial_models()