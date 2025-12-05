import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import math
from collections import deque
from config.parameters import Config
from environment.vehicle_env import VehicleEnvironment


class DRLNetwork(nn.Module):
    """DRL策略网络 - 独立决策"""

    def __init__(
        self,
        state_dim,
        num_vehicles,
        num_batch_choices,
        hidden_dim=Config.DRL_HIDDEN_SIZE,
    ):
        """
        Args:
            state_dim：状态维度
            num_vehicles：车辆数量
            num_batch_choices：每辆车可选的数据批次数
        """
        super(DRLNetwork, self).__init__()
        self.num_vehicles = num_vehicles
        self.num_batch_choices = num_batch_choices
        # 共享特征提取层
        self.shared_layers = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        # 为每辆车独立输出Q值
        # 总输出维度：num_vehicles × num_batch_choices
        self.q_value_layer = nn.Linear(
            hidden_dim, self.num_vehicles * self.num_batch_choices
        )

        # self.net = nn.Sequential(
        #     nn.Linear(state_dim, hidden_dim),
        #     nn.ReLU(),
        #     nn.Linear(hidden_dim, hidden_dim),
        #     nn.ReLU(),
        #     nn.Linear(hidden_dim, action_dim),
        # )

    def forward(self, x):
        # return self.net(x)
        """
        输入: [batch_size, state_dim]
        输出: [batch_size, num_vehicles, num_batch_choices]
        """
        features = self.shared_layers(x)
        q_values = self.q_value_layer(features)
        batch_size = x.shape[0]
        q_values = q_values.view(batch_size, self.num_vehicles, self.num_batch_choices)
        return q_values


class PrioritizedReplayBuffer:
    """优先经验回放缓冲区（非均匀采样）"""

    def __init__(
        self, capacity=Config.DRL_BUFFER_SIZE, alpha=0.6, beta=0.4, beta_increment=0.001
    ):
        self.capacity = capacity
        self.alpha = alpha  # 优先级指数
        self.beta = beta  # 重要性采样权重指数
        self.beta_increment = beta_increment
        self.buffer = []
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.position = 0
        self.size = 0

    def push(self, state, actions, reward, next_state, done, td_error=None):
        """存储经验，带初始优先级"""
        priority = (abs(td_error) + 1e-5) ** self.alpha if td_error is not None else 1.0

        if self.size < self.capacity:
            self.buffer.append((state, actions, reward, next_state, done))
        else:
            self.buffer[self.position] = (state, actions, reward, next_state, done)

        self.priorities[self.position] = priority
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        """根据优先级采样批次"""
        if self.size < batch_size:
            return None

        # 计算采样概率
        priorities = self.priorities[: self.size]
        probs = priorities / priorities.sum()

        # 根据概率采样索引
        indices = np.random.choice(self.size, batch_size, p=probs)

        # 计算重要性采样权重
        total = self.size * probs[indices]
        weights = total**-self.beta
        weights = weights / weights.max()  # 归一化
        self.beta = min(1.0, self.beta + self.beta_increment)  # 增加beta

        # 提取样本
        states, actions, rewards, next_states, dones = [], [], [], [], []
        for idx in indices:
            state, action, reward, next_state, done = self.buffer[idx]
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            next_states.append(next_state)
            dones.append(done)

        return (
            np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones, dtype=np.bool_),
            indices,
            weights,
        )

    def update_priorities(self, indices, td_errors):
        """更新样本的优先级"""
        for idx, td_error in zip(indices, td_errors):
            priority = (abs(td_error) + 1e-5) ** self.alpha
            self.priorities[idx] = priority

    def __len__(self):
        return self.size


class DRLAgent:
    """深度强化学习策略"""

    def __init__(self, state_dim, action_dim):
        self.state_dim = state_dim
        self.num_vehicles = Config.NUM_VEHICLES
        self.num_batch_choices = Config.MAX_UPLOAD_BATCHES + 1
        # 网络
        self.policy_net = DRLNetwork(state_dim, action_dim)
        self.target_net = DRLNetwork(state_dim, action_dim)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        # 优化器
        self.optimizer = optim.Adam(
            self.policy_net.parameters(), lr=Config.DRL_LEARNING_RATE
        )
        # 经验回放
        # 优先经验回放
        self.memory = PrioritizedReplayBuffer(
            capacity=Config.DRL_BUFFER_SIZE,
            alpha=0.6,  # 优先级指数
            beta=0.4,  # 重要性采样初始值
            beta_increment=0.001,
        )
        # 超参数
        self.gamma = Config.DRL_GAMMA
        self.batch_size = Config.DRL_BATCH_SIZE
        self.update_target_every = 100
        self.steps_done = 0

    def select_action(self, state, epsilon=0.1):
        """选择动作"""
        if random.random() < epsilon:
            # 随机探索
            action = np.random.randn(self.action_dim)
        else:
            # 利用策略
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad():
                action = self.policy_net(state_tensor).squeeze(0).numpy()
        return self._process_action(action)

    def _process_action(self, raw_action):
        """处理原始动作输出"""
        # 将动作分为数据标注量和带宽分配
        num_vehicles = Config.NUM_VEHICLES
        action_processed = np.zeros(2 * num_vehicles)

        for i in range(num_vehicles):
            # 数据标注量决策 (0到MAX_UPLOAD_BATCHES)
            upload_action = raw_action[i * 2] if i * 2 < len(raw_action) else 0
            upload_batches = int(
                np.clip(
                    (upload_action + 1) * Config.MAX_UPLOAD_BATCHES / 2,
                    0,
                    Config.MAX_UPLOAD_BATCHES,
                )
            )
            # 带宽分配决策 (0到1)
            bw_action = raw_action[i * 2 + 1] if i * 2 + 1 < len(raw_action) else 0
            bandwidth_ratio = 1 / (1 + np.exp(-bw_action))  # Sigmoid

            action_processed[i * 2] = upload_batches
            action_processed[i * 2 + 1] = bandwidth_ratio

        # 归一化带宽分配
        total_bandwidth = np.sum(action_processed[1::2])
        if total_bandwidth > 0:
            action_processed[1::2] = action_processed[1::2] / total_bandwidth

        return action_processed

    def optimize_model(self):
        """优化DRL模型"""
        if len(self.memory) < self.batch_size:
            return

        # 从回放缓冲区采样
        states, actions, rewards, next_states, dones = self.memory.sample(
            self.batch_size
        )

        # 转换为张量
        states = torch.FloatTensor(states)
        actions = torch.FloatTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        dones = torch.BoolTensor(dones)

        # 计算当前Q值
        current_q_values = (
            self.policy_net(states).gather(1, actions.long().unsqueeze(1)).squeeze(1)
        )

        # 计算目标Q值
        with torch.no_grad():
            next_q_values = self.target_net(next_states).max(1)[0]
            target_q_values = rewards + (self.gamma * next_q_values * ~dones)

        # 计算损失
        loss = nn.MSELoss()(current_q_values, target_q_values)

        # 优化
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # 更新目标网络
        self.steps_done += 1
        if self.steps_done % self.update_target_every == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

    def save_model(self, path):
        """保存模型"""
        torch.save(
            {
                "policy_net_state_dict": self.policy_net.state_dict(),
                "target_net_state_dict": self.target_net.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
            },
            path,
        )

    def load_model(self, path):
        """加载模型"""
        checkpoint = torch.load(path)
        self.policy_net.load_state_dict(checkpoint["policy_net_state_dict"])
        self.target_net.load_state_dict(checkpoint["target_net_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])


if __name__ == "__main__":
    env = VehicleEnvironment(None, None, None, None)
    state_dim = 3 * Config.NUM_VEHICLES  # 置信度、测试损失、质量评分
    action_dim = 2 * Config.NUM_VEHICLES  # 上传批次、带宽分配
    drlAgent = DRLAgent(state_dim, action_dim)
