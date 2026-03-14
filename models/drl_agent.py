from models.base_agent import BaseAgent
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
    def __init__(self, capacity=Config.DRL_BUFFER_SIZE, alpha=0.6, beta=0.4, beta_increment=0.001):
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.buffer = []
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.position = 0
        self.size = 0

    def push(self, state, actions, reward, next_state, done, td_error=None):
        priority = (abs(td_error) + 1e-5) ** self.alpha if td_error is not None else 1.0
        
        if self.size < self.capacity:
            self.buffer.append((state, actions, reward, next_state, done))
        else:
            self.buffer[self.position] = (state, actions, reward, next_state, done)

        self.priorities[self.position] = priority
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
        return self.position

    def sample(self, batch_size):
        if self.size < batch_size:
            return None

        priorities = self.priorities[:self.size]
        total_p = priorities.sum()
        
        if total_p == 0:
            probs = np.ones(self.size, dtype=np.float32) / self.size
        else:
            probs = priorities / total_p
            
        # 强制归一化，防止浮点误差
        probs /= probs.sum() 

        indices = np.random.choice(self.size, batch_size, p=probs)

        weights = (self.size * probs[indices]) ** -self.beta
        weights /= weights.max()
        self.beta = min(1.0, self.beta + self.beta_increment)

        batch = [self.buffer[i] for i in indices]
        states, actions, rewards, next_states, dones = zip(*batch)
        
        return (
            np.stack(states).astype(np.float32),
            np.stack(actions).astype(np.int64),
            np.stack(rewards).astype(np.float32),
            np.stack(next_states).astype(np.float32),
            np.stack(dones).astype(np.uint8),
            indices,
            weights
        )

    def update_priorities(self, indices, td_errors):
        new_prios = (np.abs(td_errors) + 1e-5) ** self.alpha
        self.priorities[indices] = new_prios

    def __len__(self):
        return self.size


class DRLAgent(BaseAgent):
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
        self.update_target_every = Config.DRL_TARGET_UPDATE_EVERY
        self.tau = Config.DRL_TAU

        # epsilon-greedy参数
        self.epsilon_start = Config.DRL_EPSILON_START
        self.epsilon_end = Config.DRL_EPSILON_END
        self.epsilon_decay = Config.DRL_EPSILON_DECAY
        self.steps_done = 0  # 已完成的训练更新次数

        self.training_mode = True  # 训练/预测模式标志

    def set_train_mode(self):
        """设置为训练模式"""
        self.training_mode = True
        self.policy_net.train()

    def set_eval_mode(self):
        """设置为评估/预测模式"""
        self.training_mode = False
        self.policy_net.eval()
        self.target_net.eval()

    def _get_epsilon(self):
        """计算当前epsilon值（指数衰减）"""
        epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * \
                  math.exp(-1. * self.steps_done / self.epsilon_decay)
        return epsilon

    def select_action(self, state, available_batches=None):
        """选择动作 - 考虑车辆实际可用批次数量限制
        Args:
            state: 状态向量
            available_batches: 每辆车实际可用的批次数量列表 [num_vehicles]
                            如果为None，则假设无限可用
        Return:
            batch_choices: 每辆车选择的批次数列表 [num_vehicles]
        """
        if available_batches is None:
            available_batches = [Config.MAX_UPLOAD_BATCHES] * self.num_vehicles

        should_explore = self.training_mode
        epsilon = self._get_epsilon() if should_explore else 0.0

        state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)

        with torch.no_grad():
            q_values = self.policy_net(state_tensor).squeeze(0)  # [num_vehicles, num_batch_choices]

        batch_choices = []
        for v in range(self.num_vehicles):
            max_allowed = min(available_batches[v], Config.MAX_UPLOAD_BATCHES)
            if max_allowed <= 0:
                batch_choices.append(0)
            elif random.random() < epsilon:
                # 随机探索：在允许范围内随机选择
                batch_choices.append(random.randint(0, max_allowed))
            else:
                # 利用：选择最大Q值的批次，但不超过允许数量
                valid_q = q_values[v, :max_allowed + 1]
                batch_choices.append(valid_q.argmax().item())

        return batch_choices

    def store_experience(self, state, action, reward, next_state, done):
        """存储经验"""
        self.memory.push(state, action, reward, next_state, done, td_error=None)

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

        # 计算所有车辆的Q值一次
        # states: [batch_size, state_dim]
        # q_values: [batch_size, num_vehicles, num_batch_choices]
        q_values = self.policy_net(states)

        # 选择过的动作
        # batch_actions: [batch_size, num_vehicles]
        chosen_q = q_values.gather(2, batch_actions.unsqueeze(2)).squeeze(2)  # [batch_size, num_vehicles]

        # Double DQN：先选动作再评估
        with torch.no_grad():
            next_q_policy = self.policy_net(next_states)  # [batch, num_vehicles, num_batch_choices]
            next_actions = next_q_policy.argmax(dim=2, keepdim=True)  # [batch, num_vehicles, 1]

            next_q_target = self.target_net(next_states)  # [batch, num_vehicles, num_batch_choices]
            next_max_q = next_q_target.gather(2, next_actions).squeeze(2)  # [batch, num_vehicles]

            # 计算目标Q值，注意dones扩展维度并转float
            target_q = rewards.unsqueeze(1) + self.gamma * next_max_q * (~dones).float().unsqueeze(1)

        # TD误差
        td_errors = target_q - chosen_q  # [batch, num_vehicles]
        # 用最大绝对误差更新优先级（与原实现一致）
        batch_td_errors = td_errors.detach().cpu().numpy()
        avg_td_errors = np.max(np.abs(batch_td_errors), axis=1)

        # 损失计算：使用huber并加权
        huber_loss = torch.nn.functional.smooth_l1_loss(
            chosen_q,
            target_q,
            reduction='none'
        )  # [batch, num_vehicles]
        loss = (huber_loss * weights.unsqueeze(1)).mean()

        # 优化
        self.optimizer.zero_grad()
        loss.backward()

        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)

        self.optimizer.step()

        # 更新经验优先级（使用之前计算的每条样本最大绝对TD误差）
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
        if torch.cuda.is_available():
            checkpoint = torch.load(path)
        else:
            checkpoint = torch.load(path, map_location=torch.device('cpu'))
        self.policy_net.load_state_dict(checkpoint["policy_net_state_dict"])
