import torch
import torch.optim as optim
from config.parameters import Config
from models.mab_selector import MABDataSelector


class ContinualLearner:
    """持续学习器"""

    def __init__(self, model, gold_model):
        self.model = model
        self.gold_model = gold_model
        self.optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)
        self.criterion = torch.nn.CrossEntropyLoss()

        # # 集成MAB选择器
        # self.mab_selector = MABDataSelector(num_arms=0)
        self.epoch_count = 0
        self.init_epochs = Config.INIT_EPOCHS

    def train_with_mab_selection(self, data_loader, num_epochs=1):
        """在数据集上训练模型，集成MAB算法进行批次选择"""
        self.model.train()
        epoch_losses = []
        batch_list = list(data_loader)
        num_batches = len(batch_list)

        # 初始化MAB选择器，臂的数量等于批次数量
        self.mab_selector = MABDataSelector(num_arms=num_batches)  # 初始化MAB选择器
        self.epoch_count = 0  # 重置epoch计数
        self.total_steps = 0  # 累计训练步骤

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

                # 训练步骤
                loss_before, inputs, targets = self._compute_batch_loss(batch)
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                loss.backward()
                self.optimizer.step()

                # 计算奖励和更新统计
                loss_after, _, _ = self._compute_batch_loss(batch)
                reward = loss_before - loss_after

                if (self.epoch_count >= self.init_epochs):
                    self.mab_selector.update_arm(batch_idx, reward)

                epoch_loss += loss.item()

            avg_loss = epoch_loss / num_batches
            epoch_losses.append(avg_loss)
            print(f"Epoch {self.epoch_count}, Loss: {avg_loss:.4f}")

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
