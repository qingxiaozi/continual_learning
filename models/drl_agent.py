import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.parameters import config


class DRLNetwork(nn.Module):
    """DRL策略网络 - 输出连续动作"""
    def __init__(self, state_dim, action_dim, hidden_dim=config.DRL_HIDDEN_SIZE):
        super(DRLNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )

    def forward(self, x):
        return self.net(x)


class ReplayBuffer:
    """经验回放缓冲区"""
    def __init__(self, capacity= config.DRL_BUFFER_SIZE):
        self.buffer = deque(maxlen = capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append(state, action, reward, next_state, done)

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack,zip(*batch))
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)


class DRLAgent:
    """深度强化学习策略"""
    def __init__(self, state_dim, action_dim):
        self.state_dim = state_dim
        self.action_dim = action_dim
        # 网络
        self.policy_net = DRLNetwork(state_dim, action_dim)
        self.target_net = DRLNetwork(state_dim, action_dim)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        # 优化器
        self.optimizer = optim.Adam(
            self.policy_net.parameters(),
            lr = config.DRL_LEARNING_RATE
        )
        # 经验回放
        self.memory = ReplayBuffer()
        # 超参数
        self.gamma = config.DRL_GAMMA
        self.batch_size = config.DRL_BATCH_SIZE
        self.update_target_every = 10
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
        num_vehicles = config.NUM_VEHICLES
        action_processed = np.zeros(2 * num_vehicles)

        for i in range(num_vehicles):
            # 数据标注量决策 (0到MAX_UPLOAD_BATCHES)
            upload_action = raw_action[i * 2] if i * 2 < len(raw_action) else 0
            upload_batches = int(np.clip(
                (upload_action + 1) * config.MAX_UPLOAD_BATCHES / 2,
                0, config.MAX_UPLOAD_BATCHES
            ))
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
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)

        # 转换为张量
        states = torch.FloatTensor(states)
        actions = torch.FloatTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        dones = torch.BoolTensor(dones)

        # 计算当前Q值
        current_q_values = self.policy_net(states).gather(1,
            actions.long().unsqueeze(1)).squeeze(1)

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
        torch.save({
            'policy_net_state_dict': self.policy_net.state_dict(),
            'target_net_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, path)

    def load_model(self, path):
        """加载模型"""
        checkpoint = torch.load(path)
        self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        self.target_net.load_state_dict(checkpoint['target_net_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])


if __name__ == "__main__":
    print(DRLNetwork)
    state_dim = 3 * config.NUM_VEHICLES  # 置信度、测试损失、质量评分
    action_dim = 2 * config.NUM_VEHICLES  # 上传批次、带宽分配
    drl = DRLNetwork(state_dim, action_dim)
    x = torch.randn(1, state_dim)
    print(drl(x))

    drlAgent = DRLAgent(state_dim, action_dim)
    print(drlAgent)