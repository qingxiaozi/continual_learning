from xml.parsers.expat import model
import torch
import random
import numpy as np
from config.parameters import Config
from environment.communication_env import CommunicationSystem
from environment.dataSimu_env import DomainIncrementalDataSimulator
from environment.vehicle_env import VehicleEnvironment
from learning.cache_manager import CacheManager
from learning.continualLearner import ContinualLearner
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
        self.global_model = GlobalModel(self.data_simulator.current_dataset, init_mode="pretrained")
        self.cache_manager = CacheManager()
        self.continual_learner = ContinualLearner(self.global_model, self.gold_model)
        self.vehicle_env = VehicleEnvironment(
            self.global_model, self.gold_model, self.cache_manager, self.data_simulator
        )
        self.communication_system = CommunicationSystem(self.vehicle_env, self.global_model)
        self.visualize = ResultVisualizer()
        self.current_domain = self.data_simulator.get_current_domain()
        self.session = 0
        dummy_state = self._get_state()
        self.state_dim = dummy_state.shape[0]
        print(self.state_dim)

    def reset(self):
        """重置环境，开始新的episode"""
        # self.gold_model.reset_parameters()
        self.global_model.reset_parameters()
        self.vehicle_env.reset()
        self.data_simulator.reset()
        self.cache_manager.reset()
        self.current_domain = self.data_simulator.get_current_domain()
        self.session = 0

        return self._get_state()

    def step(self, action):
        """
        执行一个 RL step
        一个 step = 一个完整的持续学习训练阶段

        Args:
            action: ndarray / list
                每辆车的上传批次数决策，例如：
                action[v] = m_v^s

        Returns:
            next_state: ndarray
            reward: float
            done: bool
            info: dict
        """
        batch_choices = action
        # 1. 执行动作，分配带宽，上传数据，更新缓存
        bandwidth_ratios = self._allocate_bandwidth(batch_choices)
        self._upload_datas(batch_choices)
        self._update_cache()
        # 2. 训练
        training_results = self._train_global_model()

        total_samples = self._get_total_samples()
        # 3. 通信开销
        comm_results = self._calculate_delay(batch_choices, bandwidth_ratios, training_results, total_samples)

        reward = self._calculate_reward(comm_results, training_results, total_samples)
        # 5. 更新环境
        self._update_session_environment()
        next_state = self._get_state()
        done = self._is_done()
        info = self._get_info()
        self.session += 1

        return next_state, reward, done, info

    def _allocate_bandwidth(self, batch_choices):
        allocator = BandwidthAllocator(
            batch_choices, self.communication_system, self.vehicle_env,
        )
        ratios, _ = allocator.allocate_minmaxdelay_bandwidth(self.session)
        return ratios

    def _upload_datas(self, batch_choices):
        for vehicle_id, n in enumerate(batch_choices):
            vehicle = self.vehicle_env._get_vehicle_by_id(vehicle_id)
            selected = random.sample(vehicle.data_batches, n) if n > 0 else []
            vehicle.set_uploaded_data(selected)
        return

    def _update_cache(self):
        """更新缓存"""
        for vehicle in self.vehicle_env.vehicles:
            if vehicle.uploaded_data:
                self.cache_manager.update_cache(vehicle.id, vehicle.uploaded_data)

    def _train_global_model(self):
        return self.continual_learner.train_with_cache(
            self.cache_manager, self.data_simulator, self.current_domain, Config.NUM_EPOCH
        )
    
    def _calculate_delay(self, batch_choices, bandwidth_ratios, training_results, total_samples):
        return self.communication_system.calculate_total_training_delay(
            upload_decisions=[(i, n) for i, n in enumerate(batch_choices) if n > 0],
            bandwidth_allocations={i: r for i, r in enumerate(bandwidth_ratios) if r > 0},
            total_samples=total_samples,
            actual_epochs=training_results.get("actual_epochs", 0),
        )
    
    def _get_total_samples(self):
        total = 0
        for vid in range(Config.NUM_VEHICLES):
            cache = self.cache_manager.get_vehicle_cache(vid)
            total += (len(cache["old_data"]) + len(cache["new_data"])) * Config.BATCH_SIZE
        return total

    def _calculate_reward(self, comm, train, total_samples):
        loss_reduction = train.get("loss_before", 1.0) - train.get("loss_after", 1.0)
        delay = comm["total_delay"]
        return loss_reduction / (delay * total_samples) if delay > 0 and total_samples > 0 else 0.0
    
    def _update_session_environment(self):
        """更新会话和环境状态"""
        domain_changed, previous_domain, current_domain = self.data_simulator.update_session_dataset(self.session)

        # 域发生变化，提升所有车辆的缓存
        if domain_changed:
            for vehicle_id in range(Config.NUM_VEHICLES):
                self.cache_manager.promote_new_to_old(vehicle_id)
            print(
                f"已提升缓存中的数据。"
            )

        # 更新车辆位置
        self.vehicle_env.update_vehicle_positions(time_delta=250)

        # 为车辆生成新数据
        available_batches = self._refresh_vehicle_data()
        self.current_domain = current_domain
        print(f"\n 环境更新完成 - 当前域: {current_domain}")

        return available_batches

    def _refresh_vehicle_data(self):
        """为所有车辆刷新数据"""
        available_batches = []
        for vehicle in self.vehicle_env.vehicles:
            # 生成新的数据批次
            new_data = self.data_simulator.generate_vehicle_data(vehicle.id)
            vehicle.data_batches = new_data
            available_batches.append(len(new_data))
        return available_batches

    def _is_done(self):
        """判断是否结束episode"""
        return (self.session >= Config.NUM_TRAINING_SESSIONS)

    def _get_info(self):
        """获取环境信息"""
        # eval_results = self._evaluate_model_performance()
        # return eval_results
        pass

    def _get_state(self):
        """获取当前环境状态表示"""
        state = self.vehicle_env.get_environment_state()
        # 获取每辆车当前可用批次数量
        available_batches = []
        for vehicle in self.vehicle_env.vehicles:
            available_batches.append(len(vehicle.data_batches))
        state = np.concatenate([state, available_batches], dtype=np.float32)
        return state
        