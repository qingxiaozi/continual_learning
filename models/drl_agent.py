import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
from config.parameters import config


class DRLNetwork(nn.Module):
    """DRL策略网络 - 输出连续动作"""
    def __init__(self, state_dim, action_dim, hidden_dim=config.DRL_HIDDEN_SIZE):
        super(DRLNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)

        # 分别输出动作均值和方差（用于连续动作）
        self.action_mean = nn.Linear(hidden_dim, action_dim)
        self.action_std = nn.Linear(hidden_dim, action_dim)

        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))

        action_mean = torch.tanh(self.action_mean(x))  # 限制在[-1,1]
        action_std = torch.nn.functional.softplus(self.action_std(x))  # 正数

        return action_mean, action_std