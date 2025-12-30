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
        # 为每辆车独立输出Q值：num_vehicles × num_batch_choices
        self.q_value_layer = nn.Linear(
            hidden_dim, self.num_vehicles * self.num_batch_choices
        )

    def forward(self, x):
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
    """深度强化学习策略，独立决策DQN"""

    def __init__(self, state_dim):
        self.state_dim = state_dim
        self.num_vehicles = Config.NUM_VEHICLES
        self.num_batch_choices = Config.MAX_UPLOAD_BATCHES + 1
        # 网络
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy_net = DRLNetwork(state_dim, self.num_vehicles, self.num_batch_choices)
        self.target_net = DRLNetwork(state_dim, self.num_vehicles, self.num_batch_choices)
        self.policy_net.to(self.device)
        self.target_net.to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        # 优化器
        self.optimizer = optim.Adam(
            self.policy_net.parameters(), lr=Config.DRL_LEARNING_RATE
        )
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
        self.tau = 0.005  # 软更新参数

        # epsilon-greedy参数
        self.epsilon_start = Config.DRL_EPSILON_START
        self.epsilon_end = Config.DRL_EPSILON_END
        self.epsilon_decay = Config.DRL_EPSILON_DECAY
        self.steps_done = 0  # 已完成的训练更新次数

    def _get_epsilon(self):
        """计算当前epsilon值（指数衰减）"""
        epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * \
                  math.exp(-1. * self.steps_done / self.epsilon_decay)
        return epsilon

    def select_action(self, state, available_batches=None, training=True):
        """选择动作 - 考虑车辆实际可用批次数量限制
        Args:
            state: 状态向量
            available_batches: 每辆车实际可用的批次数量列表 [num_vehicles]
                            如果为None，则假设无限可用
            training: 是否处于训练模式
        """
        epsilon = self._get_epsilon() if training else 0.0

        state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)

        with torch.no_grad():
            q_values = self.policy_net(state_tensor)  # [1, num_vehicles, num_batch_choices]
            q_values = q_values.squeeze(0)  # [num_vehicles, num_batch_choices]

        # # 如果未提供可用批次，假设可以上传最大数量
        # if available_batches is None:
        #     available_batches = [self.num_batch_choices - 1] * self.num_vehicles

        batch_choices = []
        for v in range(self.num_vehicles):
            max_allowed = min(available_batches[v], Config.MAX_UPLOAD_BATCHES)
            if max_allowed <= 0:
                # 没有可用数据或超过限制，只能选择0
                batch = 0
            else:
                if random.random() < epsilon:
                    # 随机探索：在允许范围内随机选择
                    batch = random.randint(0, max_allowed)
                else:
                    # 利用：选择最大Q值的批次，但不超过允许数量
                    valid_q_values = q_values[v, :max_allowed + 1]
                    batch = valid_q_values.argmax().item()

            batch_choices.append(batch)

        # 构建完整的动作向量
        action_vector = np.zeros(2 * self.num_vehicles)
        for i, batch in enumerate(batch_choices):
            action_vector[i * 2] = batch  # 批次决策
            action_vector[i * 2 + 1] = 0.0  # 带宽分配占位符

        return action_vector, batch_choices

    def store_experience(self, state, action_vector, reward, next_state, done):
        """存储经验"""
        # 将动作向量转换为批次选择列表
        batch_choices = action_vector.astype(np.int32).tolist()

        # 存储到优先回放缓冲区（不提供初始TD误差）
        self.memory.push(state, batch_choices, reward, next_state, done, td_error=None)

    def optimize_model(self):
        """优化模型（使用Double DQN和优先经验回放）"""
        batch_data = self.memory.sample(self.batch_size)
        if batch_data is None:
            return None

        states, batch_actions, rewards, next_states, dones, indices, weights = batch_data

        # 转换为张量
        states = torch.tensor(states, dtype=torch.float32, device=self.device)
        next_states = torch.tensor(next_states, dtype=torch.float32, device=self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        dones = torch.tensor(dones, dtype=torch.bool, device=self.device)
        weights = torch.tensor(weights, dtype=torch.float32, device=self.device)

        # 批次动作转换为张量 [batch_size, num_vehicles]
        batch_actions = torch.tensor(batch_actions, dtype=torch.long, device=self.device)

        # 收集所有车辆的TD误差
        batch_td_errors = []
        total_loss = 0.0

        for v in range(self.num_vehicles):
            # 获取该车辆的动作
            vehicle_actions = batch_actions[:, v]  # [batch_size]

            # 计算当前Q值
            current_q = self.policy_net(states)[:, v, :]  # [batch_size, num_batch_choices]
            chosen_q = current_q.gather(1, vehicle_actions.unsqueeze(1)).squeeze(1)  # [batch_size]

            # Double DQN：使用policy_net选择动作，target_net评估
            with torch.no_grad():
                # 选择下一个状态的最佳动作（根据policy_net）
                next_q_policy = self.policy_net(next_states)[:, v, :]  # [batch_size, num_batch_choices]
                next_actions = next_q_policy.argmax(dim=1, keepdim=True)  # [batch_size, 1]

                # 评估这些动作的Q值（根据target_net）
                next_q_target = self.target_net(next_states)[:, v, :]  # [batch_size, num_batch_choices]
                next_max_q = next_q_target.gather(1, next_actions).squeeze(1)  # [batch_size]

                # 计算目标Q值
                target_q = rewards + self.gamma * next_max_q * (~dones)  # [batch_size]

            # 计算TD误差和损失
            td_errors = target_q - chosen_q
            batch_td_errors.append(td_errors.detach().cpu().numpy())

            # 使用重要性采样权重
            loss = (td_errors ** 2 * weights).mean()
            total_loss += loss

        # 平均损失
        loss = total_loss / self.num_vehicles

        # 优化
        self.optimizer.zero_grad()
        loss.backward()

        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)

        self.optimizer.step()

        # 更新经验优先级
        avg_td_errors = np.mean(np.array(batch_td_errors), axis=0)
        self.memory.update_priorities(indices, avg_td_errors)

        # 软更新目标网络
        self.soft_update_target_network()

        self.steps_done += 1

        return loss.item()

    def soft_update_target_network(self):
        """软更新目标网络（更平滑）"""
        for target_param, policy_param in zip(self.target_net.parameters(), self.policy_net.parameters()):
            target_param.data.copy_(self.tau * policy_param.data + (1.0 - self.tau) * target_param.data)

    def hard_update_target_network(self):
        """硬更新目标网络（定期更新）"""
        if self.steps_done % self.update_target_every == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

    def save_model(self, path):
        """保存模型"""
        torch.save({
            "policy_net_state_dict": self.policy_net.state_dict(),
            "target_net_state_dict": self.target_net.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "steps_done": self.steps_done,
        }, path)

    def load_model(self, path):
        """加载模型"""
        checkpoint = torch.load(path)
        self.policy_net.load_state_dict(checkpoint["policy_net_state_dict"])
        self.target_net.load_state_dict(checkpoint["target_net_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.steps_done = checkpoint.get("steps_done", 0)
