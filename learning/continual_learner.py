import copy
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from config.parameters import Config
from models.mab_selector import MABDataSelector


class ContinualLearner:
    """持续学习器"""

    def __init__(self, model, gold_model):
        self.model = model
        self.gold_model = gold_model
        self.optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE, weight_decay=1e-4, betas=(0.9, 0.999))
        self.criterion = torch.nn.CrossEntropyLoss()
        self.epoch_count = 0
        self.init_epochs = Config.INIT_EPOCHS

    # def check_data_preprocessing(self, train_loader, val_loader):
    #     """检查训练集和验证集的数据预处理是否一致"""
    #     print("=== 数据预处理检查 ===")

    #     # 检查第一个批次
    #     train_batch = next(iter(train_loader))
    #     val_batch = next(iter(val_loader))

    #     train_inputs, train_targets = train_batch
    #     val_inputs, val_targets = val_batch

    #     print(f"训练集 - 输入范围: [{train_inputs.min():.3f}, {train_inputs.max():.3f}]")
    #     print(f"验证集 - 输入范围: [{val_inputs.min():.3f}, {val_inputs.max():.3f}]")
    #     print(f"训练集 - 输入均值: {train_inputs.mean():.3f}, 标准差: {train_inputs.std():.3f}")
    #     print(f"验证集 - 输入均值: {val_inputs.mean():.3f}, 标准差: {val_inputs.std():.3f}")

    #     print(f"训练集标签分布: {torch.bincount(train_targets)}")
    #     print(f"验证集标签分布: {torch.bincount(torch.tensor(val_targets))}")

    def train_with_mab_selection(self, train_loader, val_loader, num_epochs=1):
        # self.check_data_preprocessing(train_loader, val_loader)
        """在数据集上训练模型，集成MAB算法进行批次选择"""
        self.model.train()
        epoch_losses = []
        val_losses = []
        batch_list = list(train_loader)
        num_batches = len(batch_list)

        self.mab_selector = MABDataSelector(num_arms=num_batches)  # 初始化MAB选择器
        self.epoch_count = 0  # 重置epoch计数
        self.total_steps = 0  # 累计训练步骤

        best_val_loss = float('inf')
        patience_counter = 0
        best_model_state = copy.deepcopy(self.model.state_dict())

        for epoch in range(num_epochs):
            self.epoch_count += 1
            epoch_loss = 0.0
            ucb_selections = []

            for step in range(num_batches):
                self.total_steps += 1

                if self.epoch_count > self.init_epochs:
                    selected_arm = self.mab_selector.select_arm(self.total_steps)
                    self.mab_selector.record_ucb_selection(selected_arm)

                batch_idx = step % num_batches
                batch = batch_list[batch_idx]

                loss_before, inputs, targets = self._compute_batch_loss(batch)
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                loss.backward()
                self.optimizer.step()

                loss_after, _, _ = self._compute_batch_loss(batch)
                reward = loss_before - loss_after

                if (self.epoch_count >= self.init_epochs):
                    self.mab_selector.update_arm(batch_idx, reward)

                epoch_loss += loss.item()

            avg_loss = epoch_loss / num_batches
            epoch_losses.append(avg_loss)

            if val_loader is not None:
                patience = 5
                val_loss = self._evaluate_on_validation(val_loader)
                val_losses.append(val_loss)

                # 检查是否是最佳模型
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    best_model_state = copy.deepcopy(self.model.state_dict())
                    print(f"Epoch {self.epoch_count}, train_loss: {avg_loss:.4f}, val_loss: {val_loss:.4f} *")
                else:
                    patience_counter += 1
                    print(f"Epoch {self.epoch_count}, train_loss: {avg_loss:.4f}, val_loss: {val_loss:.4f}, patience_counter/patience: {patience_counter}/{patience}")

                # 早停检查
                if patience_counter >= patience:
                    print(f"早停触发！在epoch {self.epoch_count}停止")
                    self.model.load_state_dict(best_model_state)  # 恢复最佳模型
                    break
            else:
                print("验证集不存在")

        # 在训练结束后添加调试
        print(f"MAB调试 - counts: {self.mab_selector.counts}")
        print(f"MAB调试 - rewards: {self.mab_selector.rewards}")
        print(f"MAB调试 - avg_rewards: {self.mab_selector.avg_rewards}")
        print(f"MAB调试 - ucb_counts: {self.mab_selector.ucb_counts}")

        return avg_loss, epoch_losses

    def train_on_dataset(self, dataloader, num_epochs=1):
        """在数据集上训练模型"""
        self.model.train()
        # 记录每个epoch的损失值
        epoch_losses = []

        for epoch in range(num_epochs):
            epoch_loss = 0.0
            for batch_idx, batch in enumerate(dataloader):
                if isinstance(batch, (list, tuple)):
                    inputs, targets = batch
                else:
                    inputs = batch
                    # 使用黄金模型生成标签
                    with torch.no_grad():
                        targets = self.gold_model(inputs).argmax(dim=1)

                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()

            avg_loss = epoch_loss / len(dataloader)
            epoch_losses.append(avg_loss)
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")

        return avg_loss, epoch_losses

    def _evaluate_on_validation(self, val_dataset):
        """在验证集上评估模型"""
        val_loader = DataLoader(
            val_dataset,
            batch_size=Config.BATCH_SIZE,        # 设置合适的批次大小
            shuffle=False,
            drop_last=False,
            num_workers=0
        )

        self.model.eval()
        total_loss = 0.0
        batch_count = 0

        with torch.no_grad():
            for batch in val_loader:
                inputs, targets = batch[0], batch[1]
                if targets.dim() == 0:
                    targets = targets.unsqueeze(0)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                total_loss += loss.item()
                batch_count += 1

        self.model.train()
        return total_loss / batch_count if batch_count > 0 else float('inf')

    def compute_test_loss(self, dataloader):
        """计算测试损失"""
        self.model.eval()
        total_loss = 0.0

        with torch.no_grad():
            for batch in dataloader:
                if isinstance(batch, (list, tuple)):
                    inputs, targets = batch
                else:
                    inputs = batch
                    with torch.no_grad():
                        targets = self.gold_model(inputs).argmax(dim=1)

                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                total_loss += loss.item()

        return total_loss / len(dataloader)

    def update_model_parameters(self, new_params):
        """更新模型参数"""
        self.model.load_state_dict(new_params)

    def get_model_parameters(self):
        """获取模型参数"""
        return self.model.state_dict().copy()

    def _compute_batch_loss(self, batch, use_gold_model=False):
        """
        计算单个批次的损失（合并了数据准备逻辑）
        参数:
            batch: 输入数据批次
            use_gold_model: 是否强制使用黄金模型生成标签
        """
        self.model.eval()
        with torch.no_grad():
            if isinstance(batch, (list, tuple)) and len(batch) >= 2 and not use_gold_model:
                inputs, targets = batch[0], batch[1]
            else:
                inputs = batch
                # 使用黄金模型生成标签
                targets = self.gold_model(inputs).argmax(dim=1)
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)

        self.model.train()
        return loss.item(), inputs, targets

    def get_batch_rankings(self):
        """获取批次排序（基于UCB选择次数）"""
        return self.mab_selector.get_batch_rankings()
