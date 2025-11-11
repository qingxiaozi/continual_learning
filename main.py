import torch
import time
from collections import defaultdict
import numpy as np
from config.parameters import Config
from environment.communication_env import CommunicationSystem
from environment.dataSimu_env import DomainIncrementalDataSimulator
from environment.vehicle_env import VehicleEnvironment
from learning.cache_manager import CacheManager
from learning.evaluator import ModelEvaluator
from learning.continual_learner import ContinualLearner
from models.drl_agent import DRLAgent
from models.global_model import GlobalModel
from models.gold_model import GoldModel
from models.mab_selector import MABDataSelector


class BaselineComparison:
    """基线算法比较实验"""
    def __init__(self):
        # 初始化所有组件
        self.config = Config
        self.data_simulator = DomainIncrementalDataSimulator()

        # 初始化模型
        self.gold_model = GoldModel(self.data_simulator.current_dataset)
        self.global_model = GlobalModel(self.data_simulator.current_dataset)

        # 初始化学习组件
        self.cache_manager = CacheManager()
        self.evaluator = ModelEvaluator(self.gold_model)
        self.continual_learner = ContinualLearner(self.global_model, self.gold_model)

        # 初始化环境
        self.vehicle_env = VehicleEnvironment(
            self.global_model,
            self.gold_model,
            self.cache_manager,
            self.data_simulator
        )

        # 初始化通信系统
        self.communication_system = CommunicationSystem(self.vehicle_env)

        # 初始化MAB选择器
        self.mab_selector = MABDataSelector(num_arms=Config.MAX_LOCAL_BATCHES)

        # 初始化DRL智能体
        state_dim = 3 * Config.NUM_VEHICLES  # 置信度、测试损失、质量评分
        action_dim = 2 * Config.NUM_VEHICLES  # 上传批次、带宽分配
        self.drl_agent = DRLAgent(state_dim, action_dim)

        # 记录实验数据
        self.results = {
            'session_accuracies': [],
            'session_losses': [],
            'communication_delays': [],
            'cache_utilization': [],
            'domain_performance': defaultdict(list)
        }

    def run_joint_optimization(self, num_sessions=None):
        """运行完整的联合优化过程"""
        if num_sessions is None:
            num_sessions = Config.NUM_TRAINING_SESSIONS
        print("=" * 60)
        print("开始车路协同持续学习联合优化")
        print(f"数据集: {self.data_simulator.current_dataset}")
        print(f"车辆数量: {Config.NUM_VEHICLES}")
        print(f"训练会话数: {num_sessions}")
        print("=" * 60)

        for session in range(num_sessions):
            print(f"\n=== 训练会话 {session + 1}/{num_sessions} ===")
            # 步骤1: 更新会话和环境
            self._update_session_environment(session)
            # 步骤2: 获取环境状态
            state = self._get_environment_state()
            print(f"state:{state}")
            # 步骤3: DRL智能体决策
            action = self._drl_decision_making(state, session)
            print(f"action:{action}")
            # 步骤4: 执行通信和数据收集
            communication_results = self._execute_communication(action, session)
            # 步骤5: 缓存管理和数据选择
            cache_updates = self._manage_cache_and_data_selection()
            exit()



    def _update_session_environment(self, session):
            """更新会话和环境状态"""
            # 更新数据模拟器会话
            self.data_simulator.update_session(session)
            # 更新车辆位置（模拟移动）
            self.vehicle_env.update_vehicle_positions(time_delta=1.0)
            # 为车辆生成新数据
            self._refresh_vehicle_data()
            print(f"环境更新完成 - 当前域: {self.data_simulator.get_current_domain()}")

    def _refresh_vehicle_data(self):
        """为所有车辆刷新数据"""
        for vehicle in self.vehicle_env.vehicles:
            # 生成新的数据批次
            new_data = self.data_simulator.generate_vehicle_data(vehicle.id)
            vehicle.data_batches = new_data

    def _get_environment_state(self):
        """获取环境状态用于DRL决策"""
        return self.vehicle_env.get_environment_state()

    def _drl_decision_making(self, state, session):
        """DRL智能体决策过程"""
        # 使用epsilon-greedy策略，随着训练进行减少探索
        epsilon = max(0.1, 0.5 * (1 - session / Config.NUM_TRAINING_SESSIONS))
        action = self.drl_agent.select_action(state, epsilon=epsilon)

        # 解析动作
        upload_decisions = []
        bandwidth_allocations = {}

        for i in range(Config.NUM_VEHICLES):
            upload_batches = int(action[i * 2])
            bandwidth_ratio = action[i * 2 + 1]

            upload_decisions.append((i, upload_batches))
            bandwidth_allocations[i] = bandwidth_ratio

        print(f"DRL决策 - 总上传批次: {sum([ud[1] for ud in upload_decisions])}")

        return {
            'upload_decisions': upload_decisions,
            'bandwidth_allocations': bandwidth_allocations
        }

    def _execute_communication(self, action, session):
        """执行通信和数据收集"""
        upload_decisions = action['upload_decisions']
        bandwidth_allocations = action['bandwidth_allocations']

        # 收集上传数据
        uploaded_data = {}
        corrected_upload_decisions = []  # 修正后的上传决策，保持原格式
        for vehicle_id, planned_upload_batches in upload_decisions:
            actual_upload_batches = 0
            if planned_upload_batches > 0:
                vehicle = self.vehicle_env._get_vehicle_by_id(vehicle_id)
                if vehicle and vehicle.data_batches:
                    # 确保不超过实际可用的批次数量
                    available_batches = len(vehicle.data_batches)
                    actual_upload_batches = min(planned_upload_batches, available_batches)

                    # 选择前actual_upload_batches个批次上传
                    uploaded_data[vehicle_id] = vehicle.data_batches[:actual_upload_batches]
                    vehicle.set_uploaded_data(uploaded_data[vehicle_id])

                    # 记录差异（用于调试）
                    if actual_upload_batches != planned_upload_batches:
                        print(f"车辆 {vehicle_id}: 计划上传 {planned_upload_batches} 批次, "
                            f"实际可上传 {available_batches} 批次, "
                            f"实际上传 {actual_upload_batches} 批次")

            # 添加到修正后的决策列表
            corrected_upload_decisions.append((vehicle_id, actual_upload_batches))

        # 使用修正后的upload_decisions计算通信时延
        delay_breakdown = self.communication_system.calculate_total_training_delay(
            corrected_upload_decisions, bandwidth_allocations, session, Config.NUM_VEHICLES
        )

        print(f"通信时延 - 传输: {delay_breakdown['transmission_delay']:.2f}s, "
            f"总时延: {delay_breakdown['total_delay']:.2f}s")

        return {
            'delay_breakdown': delay_breakdown,
            'uploaded_data': uploaded_data,
            'corrected_upload_decisions': corrected_upload_decisions  # 返回修正后的决策
        }

    def _manage_cache_and_data_selection(self):
        """缓存管理和数据选择"""
        cache_updates = {}

        for vehicle in self.vehicle_env.vehicles:
            if vehicle.uploaded_data:
                # 使用MAB选择器评估数据质量
                quality_scores = []
                for batch in vehicle.uploaded_data:
                    if (isinstance(batch, list) and len(batch) >= 2 and
                    isinstance(batch[0], torch.Tensor) and batch[0].dim() == 4):
                        images = batch[0]

                        reward = self.mab_selector.calculate_batch_reward(
                            self.global_model, [images], torch.nn.CrossEntropyLoss()
                        )
                        quality_scores.append(reward)

                # 更新缓存
                self.cache_manager.update_cache(
                    vehicle.id, vehicle.uploaded_data, quality_scores
                )

                cache_updates[vehicle.id] = {
                    'new_batches': len(vehicle.uploaded_data),
                    'avg_quality': np.mean(quality_scores) if quality_scores else 0
                }

        # 打印缓存统计
        cache_stats = self.cache_manager.get_cache_stats()
        total_batches = sum([stats['total_size'] for stats in cache_stats.values()])
        print(f"缓存管理 - 总批次: {total_batches}")

        return cache_updates

if __name__ == "__main__":
    a = BaselineComparison()
    a.run_joint_optimization()
