import copy
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from config.parameters import Config
from learning.mab_selector import MABDataSelector


class ContinualLearner:
    """持续学习器"""

    def __init__(self, model, gold_model):
        self.model = model
        self.gold_model = gold_model
        # 弹性网络正则化参数
        self.elastic_alpha = 0.5  # L1和L2的权重平衡，0.5表示各占一半
        self.l1_lambda = 0.001    # L1正则化强度
        self.l2_lambda = 0.001    # L2正则化强度（已包含在weight_decay中）

        self.optimizer = optim.Adam(
            model.parameters(),
            lr=Config.LEARNING_RATE,
            weight_decay=self.l2_lambda,  # L2正则化
            betas=(0.9, 0.999),
        )
        self.criterion = torch.nn.CrossEntropyLoss()
        self.epoch_count = 0
        self.init_epochs = Config.INIT_EPOCHS

    def _elastic_net_loss(self, outputs, targets, model):
            """计算带弹性网络正则化的损失"""
            # 1. 计算原始损失
            ce_loss = self.criterion(outputs, targets)

            # 2. 计算弹性网络正则化项
            l1_regularization = 0.0
            l2_regularization = 0.0

            for param in model.parameters():
                if param.requires_grad:
                    # L1正则化（Lasso）
                    l1_regularization += torch.norm(param, p=1)
                    # L2正则化（Ridge）已经在Adam的weight_decay中实现
                    # 这里也可以显式计算：l2_regularization += torch.norm(param, p=2) ** 2

            # 3. 组合弹性网络损失
            elastic_loss = ce_loss + self.elastic_alpha * self.l1_lambda * l1_regularization

            return elastic_loss
    
    def train_with_cache(self, cache_manager, data_simulator, current_domain, num_epochs):
        """
        使用缓存数据进行全局模型训练，并基于 MAB 更新缓存质量
        """
        # 1. 构建全局训练数据 + batch 映射
        global_batches = []
        batch_mapping = {}
        batch_counter = 0
        for vehicle_id in range(Config.NUM_VEHICLES):
            cache = cache_manager.get_vehicle_cache(vehicle_id)
            for batch_idx, batch in enumerate(cache["old_data"]):
                global_batches.append(batch)
                batch_mapping[batch_counter] = {
                    "vehicle_id": vehicle_id,
                    "data_type": "old",
                    "local_batch_idx": batch_idx,
                }
                batch_counter += 1

            for batch_idx, batch in enumerate(cache["new_data"]):
                global_batches.append(batch)
                batch_mapping[batch_counter] = {
                    "vehicle_id": vehicle_id,
                    "data_type": "new",
                    "local_batch_idx": batch_idx,
                }
                batch_counter += 1

        if len(global_batches) == 0:
            return {
                "loss_before": 1.0,
                "loss_after": 1.0,
                "actual_epochs": 0,
            }
        # 2. 构建验证集
        val_dataset = data_simulator.get_val_dataset(current_domain)
        # 3. 训练前损失
        loss_before = self._compute_loss_on_batches(global_batches)
        # 4. 训练模型
        train_result = self.train_with_mab_selection(
            train_loader=global_batches,
            val_loader=val_dataset,
            num_epochs=num_epochs,
        )
        # 5.训练后损失
        loss_after = self._compute_loss_on_batches(global_batches)
        # 6. 更新缓存质量
        self._update_cache_quality(cache_manager, batch_mapping)

        return {
            "loss_before": loss_before,
            "loss_after": loss_after,
            "training_ce_loss": train_result["training_ce_loss"],
            "actual_epochs": train_result["actual_epochs"],
        }

    def train_with_mab_selection(self, train_loader, val_loader, num_epochs=1):
        """在数据集上训练模型，集成MAB算法进行批次选择"""
        self.model.train()
        epoch_elastic_losses = []  # 带正则的总损失（用于优化）
        epoch_ce_losses = []  # 纯CE损失（用于监控/对比）
        val_losses = []
        batch_list = list(train_loader)
        num_batches = len(batch_list)

        self.mab_selector = MABDataSelector(num_arms=num_batches)  # 初始化MAB选择器
        self.epoch_count = 0  # 重置epoch计数
        self.total_steps = 0  # 累计训练步骤

        best_val_loss = float("inf")
        patience_counter = 0
        best_model_state = copy.deepcopy(self.model.state_dict())

        actual_epochs = 0  # 记录实际训练的epoch数

        for epoch in range(num_epochs):
            self.epoch_count += 1
            actual_epochs += 1  # 每开始一个epoch就计数
            total_elastic_loss = 0.0
            total_ce_loss = 0.0

            for step in range(num_batches):
                self.total_steps += 1

                if self.epoch_count > self.init_epochs:
                    selected_arm = self.mab_selector.select()

                batch_idx = step % num_batches
                batch = batch_list[batch_idx]

                ce_loss_before, inputs, targets = self._compute_batch_loss(batch)
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                ce_loss = self.criterion(outputs, targets)  # 原始损失
                elastic_loss = self._elastic_net_loss(outputs, targets, self.model)  # 弹性正则损失

                elastic_loss.backward()
                self.optimizer.step()

                ce_loss_after, _, _ = self._compute_batch_loss(batch)
                reward = ce_loss_before - ce_loss_after

                if self.epoch_count >= self.init_epochs:
                    self.mab_selector.update(batch_idx, reward)

                total_elastic_loss += elastic_loss.item()
                total_ce_loss += ce_loss.item()

            avg_elastic_loss = total_elastic_loss / num_batches
            avg_ce_loss = total_ce_loss / num_batches

            epoch_elastic_losses.append(avg_elastic_loss)
            epoch_ce_losses.append(avg_ce_loss)

            if val_loader is not None:
                patience = 5
                val_loss = self._evaluate_on_validation(val_loader)
                val_losses.append(val_loss)

                # 检查是否是最佳模型
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    best_model_state = copy.deepcopy(self.model.state_dict())
                    print(
                        f"Epoch {self.epoch_count}, "
                        f"train_elastic: {avg_elastic_loss:.4f}, "
                        f"train_CE: {avg_ce_loss:.4f}, "
                        f"val_CE: {val_loss:.4f} *"
                    )
                else:
                    patience_counter += 1
                    print(
                        f"Epoch {self.epoch_count}, "
                        f"train_elastic: {avg_elastic_loss:.4f}, "
                        f"train_CE: {avg_ce_loss:.4f}, "
                        f"val_CE: {val_loss:.4f}, "
                        f"patience: {patience_counter}/{patience}"
                    )

                # 早停检查
                if patience_counter >= patience:
                    print(f"早停触发！在epoch {self.epoch_count}停止")
                    self.model.load_state_dict(best_model_state)  # 恢复最佳模型
                    break
            else:
                print(f"Epoch {self.epoch_count}, train_elastic: {avg_elastic_loss:.4f}, train_CE: {avg_ce_loss:.4f}")

        return {
            "training_elastic_loss": avg_elastic_loss,
            "training_ce_loss": avg_ce_loss,
            "epoch_elastic_losses": epoch_elastic_losses,
            "epoch_ce_losses": epoch_ce_losses,
            "val_losses": val_losses,
            "actual_epochs": actual_epochs,
            "config_epochs": num_epochs,
        }

    def _compute_loss_on_batches(self, batch_list):
        """
        计算模型在一组 batch 上的加权总损失
        等价于 main.py 中 _compute_weighted_loss_on_uploaded_data
        """
        total_loss = 0.0
        criterion = torch.nn.CrossEntropyLoss()

        self.model.eval()
        with torch.no_grad():
            for batch in batch_list:
                if isinstance(batch, (list, tuple)):
                    inputs, targets = batch
                else:
                    inputs = batch
                    targets = self.gold_model.model(inputs).argmax(dim=1)
                device = next(self.model.parameters()).device
                inputs = inputs.to(device)
                targets = targets.to(device)
                outputs = self.model(inputs)
                loss = criterion(outputs, targets)
                # batch-size 加权
                total_loss += loss.item() * inputs.shape[0]

        self.model.train()
        return total_loss if total_loss > 0 else 1.0

    def _update_cache_quality(self, cache_manager, batch_mapping):
        quality_scores = self.mab_selector.get_quality_scores()

        vehicle_scores = {}
        for global_idx, score in enumerate(quality_scores):
            if global_idx not in batch_mapping:
                continue

            info = batch_mapping[global_idx]
            vid = info["vehicle_id"]
            dtype = info["data_type"]

            vehicle_scores.setdefault(vid, {"old": [], "new": []})
            vehicle_scores[vid][dtype].append(
                (info["local_batch_idx"], score)
            )

        for vid, scores in vehicle_scores.items():
            cache = cache_manager.get_vehicle_cache(vid)

            scores["old"].sort()
            scores["new"].sort()

            cache["quality_scores"] = (
                [s for _, s in scores["old"]] +
                [s for _, s in scores["new"]]
            )
    
    def _evaluate_on_validation(self, val_dataset):
        """在验证集上评估模型"""
        val_loader = DataLoader(
            val_dataset,
            batch_size=Config.BATCH_SIZE,  # 设置合适的批次大小
            shuffle=False,
            drop_last=False,
            num_workers=0,
        )

        self.model.eval()
        total_loss = 0.0
        batch_count = 0

        with torch.no_grad():
            for batch in val_loader:
                inputs, targets = batch[0], batch[1]
                if targets.dim() == 0:
                    targets = targets.unsqueeze(0)
                device = next(self.model.parameters()).device
                inputs = inputs.to(device)
                targets = targets.to(device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                total_loss += loss.item()
                batch_count += 1

        self.model.train()
        return total_loss / batch_count if batch_count > 0 else float("inf")

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
            if (
                isinstance(batch, (list, tuple))
                and len(batch) >= 2
                and not use_gold_model
            ):
                inputs, targets = batch[0], batch[1]
            else:
                inputs = batch
                # 使用黄金模型生成标签
                targets = self.gold_model(inputs).argmax(dim=1)
            device = next(self.model.parameters()).device
            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)

        self.model.train()
        return loss.item(), inputs, targets

