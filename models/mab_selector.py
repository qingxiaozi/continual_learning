import numpy as np
import torch
from config.parameters import Config


class MABDataSelector:
    """多臂老虎机数据选择器"""

    def __init__(self, num_arms):
        self.num_arms = num_arms
        self.counts = np.zeros(num_arms)  # 每个臂被实际训练的次数
        self.rewards = np.zeros(num_arms)  # 每个臂的累积奖励
        self.avg_rewards = np.zeros(num_arms)  # 每个臂的平均奖励
        self.ucb_counts = np.zeros(num_arms)  # 每个臂被UCB算法选择的次数
        self.exploration_factor = Config.MAB_EXPLORATION_FACTOR

    def select_arm(self, total_steps):
        """选择臂（数据批次）"""
        ucb_values = np.zeros(self.num_arms)  # 初始化每个臂的ucb值

        for arm in range(self.num_arms):
            if self.counts[arm] == 0:
                ucb_values[arm] = float("inf")
            else:
                exploitation = self.avg_rewards[arm]  # 利用项为该臂的平均奖励
                exploration = self.exploration_factor * np.sqrt(
                    np.log(total_steps) / self.counts[arm]
                )  # 探索项基于选择次数
                ucb_values[arm] = exploitation + exploration

        return np.argmax(ucb_values)

    def update_arm(self, arm, reward):
        """更新臂的奖励统计"""
        self.counts[arm] += 1
        self.rewards[arm] += reward
        self.avg_rewards[arm] = self.rewards[arm] / self.counts[arm]

    def get_batch_rankings(self):
        """获取批次排序（基于UCB选择次数）"""
        # 按UCB选择次数降序排列
        ranked_indices = np.argsort(self.ucb_counts)[::-1]
        return ranked_indices

    def record_ucb_selection(self, arm):
        """记录UCB选择"""
        self.ucb_counts[arm] += 1

    def get_batch_quality_scores(self):
        """获取所有批次的归一化质量评分"""
        if np.sum(self.ucb_counts) == 0:
            return np.zeros(self.num_arms)
        # 质量评分 = MAB选择次数 * 平均奖励
        quality_scores = self.ucb_counts * self.avg_rewards
        if np.sum(quality_scores) > 0:
            quality_scores = quality_scores / self.num_arms

        return quality_scores

    def get_batch_statistics(self):
        """获取所有批次的详细统计信息"""
        stats = {}
        for arm in range(self.num_arms):
            stats[arm] = {
                'count': self.counts[arm],
                'total_reward': self.rewards[arm],
                'avg_reward': self.avg_rewards[arm],
                'ucb_count': self.ucb_counts[arm],
                'quality_score': self.ucb_counts[arm] * self.avg_rewards[arm] / self.num_arms if self.ucb_counts[arm] > 0 else 0
            }
        return stats

    # def calculate_batch_reward(self, model, batch, criterion):
    #     """计算使用批次更新后的奖励（损失下降）"""
    #     if batch is None:
    #         return 0.0

    #     # 获取当前损失
    #     model.eval()
    #     current_loss = 0.0
    #     count = 0

    #     with torch.no_grad():
    #         for data in batch:
    #             if isinstance(data, (list, tuple)):
    #                 inputs, targets = data
    #             else:
    #                 inputs = data
    #                 # 对于无标签数据，使用模型预测作为伪标签
    #                 with torch.no_grad():
    #                     targets = torch.argmax(model(inputs), dim=1)

    #             outputs = model(inputs)
    #             loss = criterion(outputs, targets)
    #             current_loss += loss.item()
    #             count += 1

    #     avg_current_loss = current_loss / count if count > 0 else 0.0

    #     # 这里简化：实际应该比较更新前后的损失
    #     # 返回负损失作为奖励（损失越小越好）
    #     return -avg_current_loss

    def reset(self):
        """重置MAB状态"""
        self.counts = np.zeros(self.num_arms)
        self.rewards = np.zeros(self.num_arms)
        self.avg_rewards = np.zeros(self.num_arms)
        self.ucb_counts = np.zeros(self.num_arms)
