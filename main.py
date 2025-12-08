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
from models.bandwidth_allocator import BandwidthAllocator
from utils.metrics import ResultVisualizer


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
            self.global_model, self.gold_model, self.cache_manager, self.data_simulator
        )

        # 初始化通信系统
        self.communication_system = CommunicationSystem(self.vehicle_env)

        # 初始化MAB选择器
        self.mab_selector = MABDataSelector(num_arms=Config.MAX_LOCAL_BATCHES)

        # 初始化DRL智能体
        state_dim = 3 * Config.NUM_VEHICLES  # 置信度、测试损失、质量评分
        # action_dim = 2 * Config.NUM_VEHICLES  # 上传批次、带宽分配
        self.drl_agent = DRLAgent(state_dim)
        # 新增：初始化集成控制器
        # self.integrated_controller = self.drl_agent  # DRLAgent现在包含了集成控制功能

        self.current_domain = self.data_simulator.get_current_domain()

        # 记录实验数据
        self.results = {
            "session_accuracies": [],
            "session_losses": [],
            "communication_delays": [],
            "cache_utilization": [],
            "domain_performance": defaultdict(list),
        }
        self.visualize = ResultVisualizer()

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

            # 步骤3：获取车辆可用批次数量
            available_batches = self._get_available_batches()

            # 步骤4：集成决策（上传批次 + 带宽分配）
            action_vector, batch_choices, allocation_info = self._integrated_decision(state, available_batches, session)

            # 步骤5: 执行通信和数据收集
            upload_results = self._upload_datas(batch_choices)
            # for vec in self.vehicle_env.vehicles:
            #     print(f"车辆 {vec.id} 上传数据批次数量: {len(vec.uploaded_data)}")

            # 步骤6: 缓存管理和数据选择
            cache_updates = self._manage_cache_and_data_selection()
            # print(f"调试信息：车辆数据上传后缓存信息为{cache_updates}")

            # 步骤7: 模型训练和更新
            training_results = self._train_and_update_global_model(session)

            # 步骤8：计算通信时延
            communication_results = self._calculate_communication_delay(
                batch_choices, allocation_info, session, training_results
            )

            # 步骤9: 性能评估
            evaluation_results = self._evaluate_model_performance()

            # 步骤10: 计算奖励和优化
            reward = self._calculate_reward_and_optimize(
                state,
                batch_choices,
                evaluation_results,
                communication_results,
                training_results
            )

            # 步骤10: 记录结果
            self._record_session_results(
                session, evaluation_results, communication_results, training_results
            )

            # 步骤11: 模型广播和更新
            self._broadcast_and_update_models()

            print(
                f"会话 {session + 1} 完成 - 准确率: {evaluation_results['current_accuracy']:.4f}"
            )

        # 最终评估和结果汇总
        self._final_evaluation_and_summary()

        return self.results

    def _update_session_environment(self, session):
        """更新会话和环境状态"""
        # 更新数据模拟器会话
        previous_domain = self.current_domain
        self.data_simulator.update_session(session)
        self.current_domain = self.data_simulator.get_current_domain()

        # 如果域发生变化，提升所有车辆的缓存
        if previous_domain != self.current_domain:
            for vehicle_id in range(Config.NUM_VEHICLES):
                self.cache_manager.promote_new_to_old(vehicle_id)
            print(
                f"域从 {previous_domain} 切换到 {self.current_domain}，已提升缓存中的数据。"
            )

        # 清空所有车辆的上传数据
        for vehicle in self.vehicle_env.vehicles:
            vehicle.set_uploaded_data([])

        # 更新车辆位置（模拟移动）
        self.vehicle_env.update_vehicle_positions(time_delta=1.0)
        # 为车辆生成新数据
        self._refresh_vehicle_data()
        print(f"\n 环境更新完成 - 当前域: {self.data_simulator.get_current_domain()}")

    def _refresh_vehicle_data(self):
        """为所有车辆刷新数据"""
        for vehicle in self.vehicle_env.vehicles:
            # 生成新的数据批次
            new_data = self.data_simulator.generate_vehicle_data(vehicle.id)
            vehicle.data_batches = new_data

    def _get_environment_state(self):
        """获取环境状态用于DRL决策"""
        return self.vehicle_env.get_environment_state()

    def _get_available_batches(self):
        """获取每辆车实际可用的批次数量"""
        available_batches = []
        for v in range(Config.NUM_VEHICLES):
            vehicle = self.vehicle_env._get_vehicle_by_id(v)
            if vehicle and vehicle.data_batches:
                available_batches.append(len(vehicle.data_batches))
            else:
                available_batches.append(0)
        return available_batches

    def _integrated_decision(self, state, available_batches, session):
        """集成决策：DRL选择批次 + 带宽分配优化"""
        # 1. DRL选择批次
        action_vector, batch_choices = self.drl_agent.select_action(
            state, available_batches=available_batches, training=True
        )

        # 2. 带宽分配优化
        allocator = BandwidthAllocator(
            batch_choices,
            communication_system=self.communication_system,
            vehicle_env=self.vehicle_env
        )

        # 使用优化分配方法
        bandwidth_ratios, min_max_delay = allocator.allocate_minmaxdelay_bandwidth(session)

        # 3. 更新动作向量中的带宽部分
        for i in range(len(batch_choices)):
            action_vector[i * 2 + 1] = bandwidth_ratios[i]

        allocation_info = {
            'method': 'optimized',
            'bandwidth_ratios': bandwidth_ratios,
            'min_max_delay': min_max_delay
        }
        print(f"\n决策结果")
        print(f"总上传批次: {sum(batch_choices)}")
        print(f"上传批次: {batch_choices}")
        print(f"带宽分配: {bandwidth_ratios}")
        return action_vector, batch_choices, allocation_info

    def _upload_datas(self, batch_choices):
        """执行通信和数据收集 - 最小化版本"""
        uploaded_data = {}

        for vehicle_id, planned_batches in enumerate(batch_choices):
            if planned_batches > 0:
                vehicle = self.vehicle_env._get_vehicle_by_id(vehicle_id)
                if vehicle and vehicle.data_batches:
                    # 直接提取计划数量的批次
                    uploaded_data[vehicle_id] = vehicle.data_batches[:planned_batches]
                    vehicle.set_uploaded_data(uploaded_data[vehicle_id])

        return uploaded_data

    def _manage_cache_and_data_selection(self):
        """缓存管理和数据选择"""
        cache_updates = {}

        for vehicle in self.vehicle_env.vehicles:
            if vehicle.uploaded_data:
                # 使用MAB选择器评估数据质量
                quality_scores = []
                # 更新缓存
                self.cache_manager.update_cache(
                    vehicle.id, vehicle.uploaded_data, quality_scores
                )

                cache_updates[vehicle.id] = {
                    "new_batches": len(vehicle.uploaded_data),
                    "avg_quality": np.mean(quality_scores) if quality_scores else 0,
                }
        # 打印缓存统计
        cache_stats = self.cache_manager.get_cache_stats()
        new_batches = sum([stats["new_data_size"] for stats in cache_stats.values()])
        old_batches = sum([stats["old_data_size"] for stats in cache_stats.values()])
        total_batches = sum([stats["total_size"] for stats in cache_stats.values()])
        print(f"\n缓存管理")
        print(f"当前缓存总批次: {total_batches}")
        print(f"缓存中新数据批次：{new_batches}")
        print(f"缓存中旧数据批次：{old_batches}")

        return cache_updates

    def _calculate_communication_delay(self, batch_choices, allocation_info, session, training_results=None):
        """计算通信时延 - 使用分配器的计算方法"""
        # 构建带宽分配字典
        bandwidth_allocations = {}
        for i, ratio in enumerate(allocation_info['bandwidth_ratios']):
            if ratio > 0:
                bandwidth_allocations[i] = ratio

        # 构建上传决策列表
        upload_decisions = []
        for i, batches in enumerate(batch_choices):
            if batches > 0:
                upload_decisions.append((i, batches))

        # 计算总样本数
        total_samples = self._get_total_samples()

        # 获取实际训练epoch数（从训练结果中）
        actual_epochs = None
        if training_results and 'actual_epochs' in training_results:
            actual_epochs = training_results['actual_epochs']

        # 计算时延
        delay_breakdown = self.communication_system.calculate_total_training_delay(
            upload_decisions, bandwidth_allocations, session, total_samples, actual_epochs
        )

        print(f"\n通信计算")
        print(f"传输时延: {delay_breakdown['transmission_delay']:.2f}s")
        print(f"标注时延: {delay_breakdown['labeling_delay']:.2f}s")
        print(f"训练时延: {delay_breakdown['retraining_delay']:.2f}s")
        print(f"广播时延: {delay_breakdown['broadcast_delay']:.2f}s")
        print(f"通信时延: {delay_breakdown['total_delay']:.2f}s")

        return {
            "total_delay": delay_breakdown["total_delay"],
            "total_samples": total_samples,
        }

    def _get_total_samples(self):
        """获取总样本数"""
        total_samples = 0
        for vehicle_id in range(Config.NUM_VEHICLES):
            cache = self.cache_manager.get_vehicle_cache(vehicle_id)
            total_samples += (len(cache["old_data"]) + len(cache["new_data"])) * Config.BATCH_SIZE
        return total_samples

    def _train_and_update_global_model(self, session):
        """训练和更新全局模型"""
        # 收集所有缓存数据构建全局数据集
        global_data_batches = []
        batch_mapping = {}  # 记录全局批次索引到车辆缓存的映射
        batch_counter = 0

        for vehicle_id in range(Config.NUM_VEHICLES):
            cache = self.cache_manager.get_vehicle_cache(vehicle_id)

            # 记录旧数据的映射
            for batch_idx, batch in enumerate(cache["old_data"]):
                global_data_batches.append(batch)
                batch_mapping[batch_counter] = {
                    "vehicle_id": vehicle_id,
                    "data_type": "old",
                    "local_batch_idx": batch_idx,
                }
                batch_counter += 1

            # 记录新数据的映射
            for batch_idx, batch in enumerate(cache["new_data"]):
                global_data_batches.append(batch)
                batch_mapping[batch_counter] = {
                    "vehicle_id": vehicle_id,
                    "data_type": "new",
                    "local_batch_idx": batch_idx,
                }
                batch_counter += 1

        if not global_data_batches:
            print("警告: 全局数据集为空，跳过训练")
            return {
                "loss_before": 1.0,
                "loss_after": 1.0,
                "training_loss": float("inf"),
                "actual_epochs": 0,
            }

        current_val_data = self.data_simulator.get_val_dataset(self.current_domain)

        loss_before = self._compute_weighted_loss_on_uploaded_data(self.global_model)
        training_result = self.continual_learner.train_with_mab_selection(
            global_data_batches, current_val_data, num_epochs=Config.NUM_EPOCH
        )
        print(f"调试信息：")
        print(training_result)

        # 初始化默认值
        training_loss = float("inf")
        epoch_losses = []
        val_losses = []
        actual_epochs = Config.NUM_EPOCH  # 默认使用配置值

        if training_result is not None:
            training_loss = training_result.get("training_loss", float("inf"))
            val_losses = training_result.get("val_losses", [])
            epoch_losses = training_result.get("epoch_losses", [])
            actual_epochs = training_result.get("actual_epochs", len(epoch_losses))

        loss_after = self._compute_weighted_loss_on_uploaded_data(self.global_model)

        # 训练完成后，根据MAB统计信息更新缓存质量评分
        self._update_cache_with_mab_scores(batch_mapping)

        self.visualize.plot_training_loss(
            epoch_losses,
            val_losses,
            save_plot=True,
            plot_name=f"training_loss_session_{session}.png",
        )

        return {
            "loss_before": loss_before,
            "loss_after": loss_after,
            "training_loss": training_loss,
            "actual_epochs": actual_epochs,  # 关键：返回实际训练epoch数
        }

    def _update_cache_with_mab_scores(self, batch_mapping):
        """根据MAB统计信息更新缓存质量评分"""
        quality_scores = self.continual_learner.mab_selector.get_batch_quality_scores()

        print(f"调试信息：经训练后，根据mab选择器结果，所获得的quality_scores为：")
        print(quality_scores)

        # 按车辆分组质量评分
        vehicle_scores = {}
        for global_batch_idx, quality_score in enumerate(quality_scores):
            if global_batch_idx in batch_mapping:
                mapping = batch_mapping[global_batch_idx]
                vehicle_id = mapping["vehicle_id"]
                data_type = mapping["data_type"]
                if vehicle_id not in vehicle_scores:
                    vehicle_scores[vehicle_id] = {"old": [], "new": []}
                vehicle_scores[vehicle_id][data_type].append(
                    {
                        "local_batch_idx": mapping["local_batch_idx"],
                        "quality_score": quality_score,
                    }
                )
        # 更新每个车辆缓存的质量评分
        for vehicle_id, scores_by_type in vehicle_scores.items():
            cache = self.cache_manager.get_vehicle_cache(vehicle_id)

            # 合并所有质量评分，按原始顺序存储
            all_quality_scores = []

            # 处理旧数据质量评分（按原始顺序）
            old_scores_info = scores_by_type["old"]
            if old_scores_info:
                old_scores_info.sort(key=lambda x: x["local_batch_idx"])
                old_quality_scores = [info["quality_score"] for info in old_scores_info]
                all_quality_scores.extend(old_quality_scores)

            # 处理新数据质量评分（按原始顺序）
            new_scores_info = scores_by_type["new"]
            if new_scores_info:
                new_scores_info.sort(key=lambda x: x["local_batch_idx"])
                new_quality_scores = [info["quality_score"] for info in new_scores_info]
                all_quality_scores.extend(new_quality_scores)

            # 验证并更新缓存
            total_expected_batches = len(cache["old_data"]) + len(cache["new_data"])
            if len(all_quality_scores) == total_expected_batches:
                cache["quality_scores"] = all_quality_scores
            else:
                print(
                    f"警告: 车辆 {vehicle_id} 质量评分数量不匹配 - "
                    f"预期: {total_expected_batches}, 实际: {len(all_quality_scores)}"
                )

            # 统计信息
            total_batches = len(old_scores_info) + len(new_scores_info)
            all_scores = [
                info["quality_score"] for info in old_scores_info + new_scores_info
            ]

            print(
                f"车辆 {vehicle_id} 质量评分更新: {total_batches} 个批次参与训练, "
                f"平均质量 {np.mean(all_scores):.4f}"
            )

        # import json
        # cache_stats = self.cache_manager.get_cache_stats()
        # formatted_stats = json.dumps(cache_stats, indent=2, ensure_ascii=False, default=str)
        # print("缓存更新后车辆的缓存信息:")
        # print(formatted_stats)

    def _compute_weighted_loss_on_uploaded_data(self, model):
        """计算模型在上传数据上的损失"""
        total_loss = 0.0
        # batch_count = 0

        for vehicle in self.vehicle_env.vehicles:
            if vehicle.uploaded_data:
                vehicle_data_size = len(vehicle.uploaded_data) * Config.BATCH_SIZE
                for batch in vehicle.uploaded_data:
                    loss = self._compute_batch_loss(model, batch)
                    total_loss += loss * Config.BATCH_SIZE
                    # batch_count += 1

        # return total_loss / batch_count if batch_count > 0 else 1.0
        return total_loss if total_loss > 0 else 1.0

    def _compute_batch_loss(self, model, batch):
        """计算单个批次的损失"""
        model.eval()
        criterion = torch.nn.CrossEntropyLoss()

        with torch.no_grad():
            if isinstance(batch, (list, tuple)):
                inputs, targets = batch
            else:
                inputs = batch
                # 使用黄金模型生成标签
                with torch.no_grad():
                    targets = self.gold_model.model(inputs).argmax(dim=1)

            outputs = model(inputs)
            loss = criterion(outputs, targets)

        return loss.item()

    def _evaluate_model_performance(self):
        """评估模型性能"""
        # 评估当前域性能
        current_test_data = self.data_simulator.get_test_dataset(self.current_domain)
        if current_test_data:
            accuracy, loss = self.evaluator.evaluate_model(
                self.global_model, current_test_data
            )
        else:
            accuracy, loss = 0.0, float("inf")

        # 累积评估（所有已见域）
        cumulative_results = self.data_simulator.evaluate_model(
            self.global_model, strategy="cumulative"
        )

        eval_results = {
            "current_accuracy": accuracy,
            "current_loss": loss,
            "cumulative_results": cumulative_results,
            "current_domain": self.current_domain,
        }
        print(f"eval_results:{eval_results}")
        return eval_results

    def _calculate_reward_and_optimize(
        self, state, batch_choices, eval_results, comm_results, training_results
    ):
        """根据原始建模重新设计奖励计算函数"""
        # 获取必要的参数
        total_delay = comm_results["total_delay"]
        global_dataset_size = comm_results["total_samples"]
        # accuracy = eval_results["current_accuracy"]

        print(f"total_delay:{total_delay}")
        print(f"global_dataset_size:{global_dataset_size}")

        loss_before = training_results.get("loss_before", 1.0)
        loss_after = training_results.get("loss_after", 1.0)
        total_loss_reduction = loss_before - loss_after

        # 计算奖励
        if total_delay > 0 and global_dataset_size > 0:
            reward = total_loss_reduction / (global_dataset_size * total_delay)
        else:
            reward = 0.0
        # 获取下一个状态
        next_state = self._get_environment_state()
        action_vector = np.array(batch_choices, dtype=np.float32)

        # 存储经验并优化DRL模型
        self.drl_agent.store_experience(state, action_vector, reward, next_state, False)

        if len(self.drl_agent.memory) >= Config.DRL_BATCH_SIZE:
            self.drl_agent.optimize_model()

        print(
            f"奖励计算 - 损失降幅: {total_loss_reduction:.4f}, 奖励: {reward:.4f}"
        )

        return reward

    def _record_session_results(
        self, session, eval_results, comm_results, training_results
    ):
        """记录会话结果"""
        self.results["session_accuracies"].append(eval_results["current_accuracy"])
        self.results["session_losses"].append(eval_results["current_loss"])
        self.results["communication_delays"].append(comm_results["total_delay"])

        # 记录域性能
        current_domain = eval_results["current_domain"]
        self.results["domain_performance"][current_domain].append(
            eval_results["current_accuracy"]
        )

        # 记录缓存利用率
        cache_stats = self.cache_manager.get_cache_stats()
        avg_utilization = np.mean(
            [
                stats["total_size"] / Config.MAX_LOCAL_BATCHES
                for stats in cache_stats.values()
            ]
        )
        self.results["cache_utilization"].append(avg_utilization)
        # 添加数据异质性可视化
        if session % Config.DOMAIN_CHANGE_INTERVAL == 0:  # 每次域切换时绘制
            self.visualize.plot_data_heterogeneity(
                self.data_simulator,
                session,
                save_plot=True,
                plot_name=f"data_distribution_session_{session}.png",
            )

    def _broadcast_and_update_models(self):
        """广播和更新模型"""
        # 在实际系统中，这里会将更新后的全局模型参数广播给所有车辆
        # 简化实现：直接更新车辆环境中的模型引用
        print("模型更新完成 - 新全局模型已就绪")

    def _final_evaluation_and_summary(self):
        """最终评估和结果汇总"""
        print("\n" + "=" * 60)
        print("联合优化实验完成")
        print("=" * 60)

        # 计算总体统计
        final_accuracy = (
            self.results["session_accuracies"][-1]
            if self.results["session_accuracies"]
            else 0
        )
        avg_accuracy = np.mean(self.results["session_accuracies"])
        avg_delay = np.mean(self.results["communication_delays"])

        print(f"最终准确率: {final_accuracy:.4f}")
        print(f"平均准确率: {avg_accuracy:.4f}")
        print(f"平均通信时延: {avg_delay:.2f}s")
        print(f"平均缓存利用率: {np.mean(self.results['cache_utilization']):.2f}")

        # 打印各域性能
        print("\n各域性能:")
        for domain, performances in self.results["domain_performance"].items():
            if performances:
                avg_perf = np.mean(performances)
                print(f"  {domain}: {avg_perf:.4f}")


if __name__ == "__main__":
    a = BaselineComparison()
    a.run_joint_optimization()
