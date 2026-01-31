import torch
import random
import numpy as np
from config.parameters import Config
from environment.communication_env import CommunicationSystem
from environment.dataSimu_env import DomainIncrementalDataSimulator
from environment.vehicle_env import VehicleEnvironment
from learning.cache_manager import CacheManager
from learning.continual_learner import ContinualLearner
from models.global_model import GlobalModel
from models.gold_model import GoldModel
from models.bandwidth_allocator import BandwidthAllocator
from utils.metrics import ResultVisualizer


class VehicleEdgeEnv:
    """强化学习环境，模拟车辆边缘计算场景"""
    def __init__(self):
        self.config = Config
        self.data_simulator = DomainIncrementalDataSimulator()
        self.gold_model = GoldModel(self.data_simulator.current_dataset)
        self.global_model = GlobalModel(self.data_simulator.current_dataset, auto_load=True)
        self.cache_manager = CacheManager()
        self.continual_learner = ContinualLearner(self.global_model, self.gold_model)
        self.vehicle_env = VehicleEnvironment(
            self.global_model, self.gold_model, self.cache_manager, self.data_simulator
        )
        self.communication_system = CommunicationSystem(self.vehicle_env)
        self.visualize = ResultVisualizer()
        self.current_domain = self.data_simulator.get_current_domain()

    def reset(self):
        """重置环境，开始新的episode"""
        self.vehicle_env.reset()
        self.data_simulator.reset()
        self.cache_manager.reset()
        self.current_domain = self.data_simulator.get_current_domain()
        self.session = 0

        return self._get_state()

    def step(self, action):
        """执行一步环境交互"""
        """
        执行一个动作（每辆车的上传决策），返回 (next_state, reward, done, info)
        Args:
            action: 动作，长度为 NUM_VEHICLES，每个元素 ∈ [0, MAX_LOCAL_BATCHES]
        Returns:
            next_state, reward, done, info
        """
        self._apply_action(action)
        reward = self._calculate_reward()
        done = self._is_done()
        next_state = self._get_state()
        info = self._get_info()

        return next_state, reward, done, info

    def _apply_action(self, action):
        """应用动作到环境"""
        self._upload_datas(self, action)
        allocator = BandwidthAllocator(
            action, self.communication_system, self.vehicle_env
        )
        bandwidth_ratios = allocator.allocate_bandwidth()
        self._manage_cache_and_data_selection()
        training_results = self._train_and_update_global_model()
        commu_results = self._calculate_communication_delay(
                action, bandwidth_ratios, training_results
            )
        
        return commu_results, training_results

    def _calculate_reward(self):
        """计算当前状态下的奖励"""
        return self.vehicle_env.get_reward()

    def _is_done(self):
        """判断是否结束episode"""
        self.session += 1
        return (self.session >= Config.NUM_TRAINING_SESSIONS)

    def _get_info(self):
        """获取环境信息"""
        eval_results = self._evaluate_model_performance()
        return eval_results

    def _get_state(self):
        """获取当前环境状态表示"""
        state = self.vehicle_env.get_environment_state()
        return state
