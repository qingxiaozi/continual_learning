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
                exploration = np.sqrt(
                    self.exploration_factor * np.log(np.sum(self.ucb_counts)+1) / (self.ucb_counts[arm] + 1e-6)
                )  # 探索项基于UCB选择次数
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
        """获取所有批次的UCB质量评分（利用项 + 探索项）"""
        # 确保至少有被选择的臂
        if np.sum(self.ucb_counts) == 0:
            return np.zeros(self.num_arms)

        total_ucb_selections = np.sum(self.ucb_counts)
        scores = np.zeros(self.num_arms)

        for arm in range(self.num_arms):
            if self.ucb_counts[arm] > 0:
                exploitation = self.avg_rewards[arm]
                exploration = np.sqrt(
                    self.exploration_factor * np.log(total_ucb_selections) / self.ucb_counts[arm]
                )
                scores[arm] = exploitation + exploration

        return scores

    def get_batch_statistics(self):
        """获取所有批次的详细统计信息"""
        stats = {}
        for arm in range(self.num_arms):
            stats[arm] = {
                "count": self.counts[arm],
                "total_reward": self.rewards[arm],
                "avg_reward": self.avg_rewards[arm],
                "ucb_count": self.ucb_counts[arm],
                "quality_score": (
                    self.avg_rewards[arm] + np.sqrt(
                        self.exploration_factor * np.log(np.sum(self.ucb_counts)) / (self.ucb_counts[arm])
                    )
                    if self.ucb_counts[arm] > 0
                    else 0
                ),
            }
        return stats

    def reset(self):
        """重置MAB状态"""
        self.counts = np.zeros(self.num_arms)
        self.rewards = np.zeros(self.num_arms)
        self.avg_rewards = np.zeros(self.num_arms)
        self.ucb_counts = np.zeros(self.num_arms)
