import numpy as np
from config.parameters import Config
from learning.batch_selector import BatchSelector


class MABDataSelector(BatchSelector):
    """UCB-based Multi-Armed Bandit"""

    def __init__(self, num_arms):
        self.num_arms = num_arms
        self.exploration_factor = Config.MAB_EXPLORATION_FACTOR
        self.reset()
    
    def reset(self):
        self.counts = np.zeros(self.num_arms)  # 每个臂的选择次数
        self.rewards = np.zeros(self.num_arms)  # 每个臂的累计奖励
        self.avg_rewards = np.zeros(self.num_arms)  # 每个臂的平均奖励
        self.ucb_counts = np.zeros(self.num_arms)  # 每个臂被UCB选择的次数

    def select(self):
        """选择臂"""
        ucb_values = np.zeros(self.num_arms)  # 初始化每个臂的ucb值
        total_ucb = np.sum(self.ucb_counts) + 1e-6
        for arm in range(self.num_arms):
            if self.counts[arm] == 0:
                ucb_values[arm] = float("inf")
            else:
                exploit = self.avg_rewards[arm]  # 利用项为该臂的平均奖励
                explore = np.sqrt(
                    self.exploration_factor * np.log(total_ucb) / (self.ucb_counts[arm] + 1e-6)
                )  # 探索项基于UCB选择次数
                ucb_values[arm] = exploit + explore

        chosen = int(np.argmax(ucb_values))
        self.ucb_counts[chosen] += 1
        return chosen

    def update(self, arm, reward):
        """更新臂的奖励统计"""
        self.counts[arm] += 1
        self.rewards[arm] += reward
        self.avg_rewards[arm] = self.rewards[arm] / self.counts[arm]

    def get_quality_scores(self):
        """获取所有批次的UCB质量评分"""
        scores = np.zeros(self.num_arms)
        total = np.sum(self.ucb_counts) + 1e-6

        for arm in range(self.num_arms):
            if self.ucb_counts[arm] > 0:
                scores[arm] = (
                    self.avg_rewards[arm]
                    + np.sqrt(
                        self.exploration_factor
                        * np.log(total)
                        / self.ucb_counts[arm]
                    )
                )
        return scores
