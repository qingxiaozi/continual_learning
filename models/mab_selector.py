import numpy as np
import torch
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.parameters import config

class MABDataSelector:
    """多臂老虎机数据选择器"""
    def __init__(self, num_arms):
        self.num_arms = num_arms
        self.counts = np.zeros(num_arms)  # 每个臂被选择的次数
        self.rewards = np.zeros(num_arms)  # 每个臂的累积奖励
        self.avg_rewards = np.zeros(num_arms)  # 每个臂的平均奖励
        self.exploration_factor = config.MAB_EXPLORATION_FACTOR

    def select_arm(self, epoch, init_epochs=config.INIT_EPOCHS):
        """选择臂（数据批次）"""
        if epoch < init_epochs:
            # 初始阶段：均匀探索
            return epoch % self.num_arms

        # UCB算法
        total_counts = np.sum(self.counts)
        ucb_values = np.zeros(self.num_arms)

        for arm in range(self.num_arms):
            if self.counts[arm] == 0:
                ucb_values[arm] = float('inf')
            else:
                exploitation = self.avg_rewards[arm]
                exploration = self.exploration_factor * np.sqrt(
                    np.log(total_counts) / self.counts[arm]
                )
                ucb_values[arm] = exploitation + exploration

        return np.argmax(ucb_values)

    def update_arm(self, arm, reward):
        """更新臂的奖励统计"""
        self.counts[arm] += 1
        self.rewards[arm] += reward

        # 更新平均奖励
        self.avg_rewards[arm] = self.rewards[arm] / self.counts[arm]

    def calculate_batch_reward(self, model, batch, criterion):
        """计算使用批次更新后的奖励（损失下降）"""
        if not batch:
            return 0.0

        # 获取当前损失
        model.eval()
        current_loss = 0.0
        count = 0

        with torch.no_grad():
            for data in batch:
                if isinstance(data, (list, tuple)):
                    inputs, targets = data
                else:
                    inputs = data
                    # 对于无标签数据，使用模型预测作为伪标签
                    with torch.no_grad():
                        targets = torch.argmax(model(inputs), dim=1)

                outputs = model(inputs)
                loss = criterion(outputs, targets)
                current_loss += loss.item()
                count += 1

        avg_current_loss = current_loss / count if count > 0 else 0.0

        # 这里简化：实际应该比较更新前后的损失
        # 返回负损失作为奖励（损失越小越好）
        return -avg_current_loss

    def get_batch_quality_scores(self):
        """获取所有批次的归一化质量评分"""
        if np.sum(self.counts) == 0:
            return np.ones(self.num_arms) / self.num_arms

        # 基于选择次数和平均奖励计算质量评分
        quality_scores = self.counts * self.avg_rewards
        if np.sum(quality_scores) > 0:
            quality_scores = quality_scores / np.sum(quality_scores)

        return quality_scores

    def reset(self):
        """重置MAB状态"""
        self.counts = np.zeros(self.num_arms)
        self.rewards = np.zeros(self.num_arms)
        self.avg_rewards = np.zeros(self.num_arms)


# def main_detailed():
#     """包含简单模型的MAB使用示例"""
#     import torch
#     import torch.nn as nn

#     # 简单的线性模型
#     class SimpleModel(nn.Module):
#         def __init__(self):
#             super().__init__()
#             self.linear = nn.Linear(10, 2)

#         def forward(self, x):
#             return self.linear(x)

#     # 初始化
#     num_batches = 3
#     mab_selector = MABDataSelector(num_arms=num_batches)
#     model = SimpleModel()
#     criterion = nn.CrossEntropyLoss()

#     # 模拟一些数据批次
#     batches = []
#     for i in range(num_batches):
#         # 每个批次包含一些随机数据
#         data = torch.randn(32, 10)  # 32个样本，10维特征
#         targets = torch.randint(0, 2, (32,))  # 随机标签
#         batches.append((data, targets))

#     # 训练循环
#     for epoch in range(15):
#         # 选择批次
#         batch_idx = mab_selector.select_arm(epoch)
#         selected_batch = batches[batch_idx]

#         print(f"Epoch {epoch}: 使用批次 {batch_idx}")

#         # 计算奖励（使用当前模型在批次上的损失）
#         reward = mab_selector.calculate_batch_reward(model, [selected_batch], criterion)

#         # 更新MAB
#         mab_selector.update_arm(batch_idx, reward)

#         print(f"  奖励: {reward:.3f}")

#     # 结果分析
#     print("\n最终选择统计:")
#     for i in range(num_batches):
#         print(f"批次 {i}: 被选择 {mab_selector.counts[i]} 次")

# if __name__ == "__main__":
#     main_detailed()  # 运行最简版本
#     # main_detailed()  # 运行详细版本