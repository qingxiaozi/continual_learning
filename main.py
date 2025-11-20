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
        action_dim = 2 * Config.NUM_VEHICLES  # 上传批次、带宽分配
        self.drl_agent = DRLAgent(state_dim, action_dim)

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
            # print(f"当前环境状态 state 为:\n{state}")

            # 步骤3: DRL智能体决策
            action = self._drl_decision_making(state, session)
            # print(f"智能体的 action 为:\n{action}")

            # 步骤4: 执行通信和数据收集
            upload_results = self._execute_communication(action, session)

            # 步骤5: 缓存管理和数据选择
            cache_updates = self._manage_cache_and_data_selection()
            # print(f"调试信息：车辆数据上传后缓存信息为{cache_updates}")

            # 步骤6：计算通信时延
            communication_results = self._calculate_communication_delay(action, session, upload_results['corrected_upload_decisions'])

            # 步骤7: 模型训练和更新
            training_results = self._train_and_update_global_model(session)

            # 步骤8: 性能评估
            evaluation_results = self._evaluate_model_performance(session)

            # 步骤9: 计算奖励和优化
            reward = self._calculate_reward_and_optimize(
                state,
                action,
                evaluation_results,
                communication_results,
                training_results,
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
            print(f"域从 {previous_domain} 切换到 {self.current_domain}，已提升缓存中的数据。")

        # 清空所有车辆的上传数据
        for vehicle in self.vehicle_env.vehicles:
            vehicle.set_uploaded_data([])

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
            "upload_decisions": upload_decisions,
            "bandwidth_allocations": bandwidth_allocations,
        }

    def _execute_communication(self, action, session):
        """执行通信和数据收集"""
        upload_decisions = action["upload_decisions"]
        bandwidth_allocations = action["bandwidth_allocations"]

        # 收集上传数据
        uploaded_data = {}
        corrected_upload_decisions = []  # 修正后的上传决策，保持原格式

        print("\n上传数据验证:")
        print("车辆ID | 计划上传 | 实际可上传 | 实际上传 | 状态")
        print("-" * 50)

        for vehicle_id, planned_upload_batches in upload_decisions:
            actual_upload_batches = 0
            available_batches = 0

            if planned_upload_batches > 0:
                vehicle = self.vehicle_env._get_vehicle_by_id(vehicle_id)
                if vehicle and vehicle.data_batches:
                    # 确保不超过实际可用的批次数量
                    available_batches = len(vehicle.data_batches)
                    actual_upload_batches = min(
                        planned_upload_batches, available_batches
                    )

                    # 选择前actual_upload_batches个批次上传
                    uploaded_data[vehicle_id] = vehicle.data_batches[
                        :actual_upload_batches
                    ]
                    vehicle.set_uploaded_data(uploaded_data[vehicle_id])
                else:
                    available_batches = 0
            else:
                # 如果计划上传为0，检查车辆是否有数据
                vehicle = self.vehicle_env._get_vehicle_by_id(vehicle_id)
                available_batches = len(vehicle.data_batches) if vehicle and vehicle.data_batches else 0

            # 记录差异（用于调试）- 只要不一致就打印
            if actual_upload_batches != planned_upload_batches or planned_upload_batches == 0:
                status = "不一致" if actual_upload_batches != planned_upload_batches else "计划不上传"
                print(f"车辆 {vehicle_id:2d} | {planned_upload_batches:6d}   | {available_batches:8d}   | {actual_upload_batches:6d}   | {status}")
            else:
                print(f"车辆 {vehicle_id:2d} | {planned_upload_batches:6d}   | {available_batches:8d}   | {actual_upload_batches:6d}   | 一致")

            corrected_upload_decisions.append((vehicle_id, actual_upload_batches))

        # 计算统计信息
        total_planned = sum([ud[1] for ud in upload_decisions])
        total_actual = sum([cd[1] for cd in corrected_upload_decisions])
        print("-" * 50)
        print(f"总计     | {total_planned:6d}   | {'N/A':8s}   | {total_actual:6d}   |")
        print(f"实际上传率: {total_actual/total_planned*100:.1f}% ({total_actual}/{total_planned})")

        return {
            "uploaded_data": uploaded_data,
            "corrected_upload_decisions": corrected_upload_decisions,  # 返回修正后的决策
        }

    def _manage_cache_and_data_selection(self):
        """缓存管理和数据选择"""
        cache_updates = {}

        for vehicle in self.vehicle_env.vehicles:
            if vehicle.uploaded_data:
                # 使用MAB选择器评估数据质量
                quality_scores = []
                # for batch in vehicle.uploaded_data:
                #     if (
                #         isinstance(batch, list)
                #         and len(batch) >= 2
                #         and isinstance(batch[0], torch.Tensor)
                #         and batch[0].dim() == 4
                #     ):
                #         images = batch[0]

                #         reward = self.mab_selector.calculate_batch_reward(
                #             self.global_model, [images], torch.nn.CrossEntropyLoss()
                #         )
                #         quality_scores.append(reward)

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
        total_batches = sum([stats["total_size"] for stats in cache_stats.values()])
        print(f"缓存管理 - 当前缓存总批次: {total_batches}")

        return cache_updates

    def _calculate_communication_delay(self, action, session, corrected_upload_decisions):
        """计算通信时延 - 在数据更新后调用"""
        bandwidth_allocations = action["bandwidth_allocations"]

        global_data_batches = []
        total_samples = 0  # 包含多少个样本数
        for vehicle_id in range(Config.NUM_VEHICLES):
            cache = self.cache_manager.get_vehicle_cache(vehicle_id)
            total_samples += (
                len(cache["old_data"]) + len(cache["new_data"])
            ) * Config.BATCH_SIZE

        # 使用修正后的upload_decisions计算通信时延
        delay_breakdown = self.communication_system.calculate_total_training_delay(
            corrected_upload_decisions, bandwidth_allocations, session, total_samples
        )

        print(
            f"""
        通信时延详情:
        ├─ 传输时延: {delay_breakdown['transmission_delay']:>8.2f} s
        ├─ 标注时延: {delay_breakdown['labeling_delay']:>8.2f} s
        ├─ 训练时延: {delay_breakdown['retraining_delay']:>8.2f} s
        ├─ 广播时延: {delay_breakdown['broadcast_delay']:>8.2f} s
        ╰─ 总时延:   {delay_breakdown['total_delay']:>8.2f} s
        训练样本数：{total_samples} samples
        """
        )
        return {
            "total_delay": delay_breakdown['total_delay'],
            "total_samples": total_samples,
        }

    def _train_and_update_global_model(self, session):
        """训练和更新全局模型"""
        # 收集所有缓存数据构建全局数据集
        global_data_batches = []
        global_dataset_size = 0  # 包含多少个样本数
        batch_mapping = {}  # 记录全局批次索引到车辆缓存的映射
        batch_counter = 0

        for vehicle_id in range(Config.NUM_VEHICLES):
            cache = self.cache_manager.get_vehicle_cache(vehicle_id)

            # 记录旧数据的映射
            for batch_idx, batch in enumerate(cache["old_data"]):
                global_data_batches.append(batch)
                batch_mapping[batch_counter] = {
                    'vehicle_id': vehicle_id,
                    'data_type': 'old',
                    'local_batch_idx': batch_idx
                }
                batch_counter += 1

            # 记录新数据的映射
            for batch_idx, batch in enumerate(cache["new_data"]):
                global_data_batches.append(batch)
                batch_mapping[batch_counter] = {
                    'vehicle_id': vehicle_id,
                    'data_type': 'new',
                    'local_batch_idx': batch_idx
                }
                batch_counter += 1
            global_dataset_size += (
                len(cache["old_data"]) + len(cache["new_data"])
            ) * Config.BATCH_SIZE

        if not global_data_batches:
            print("警告: 全局数据集为空，跳过训练")
            return {
                "loss_before": 1.0,
                "loss_after": 1.0,
                "training_loss": float("inf"),
                "global_dataset_size": 0,
            }

        from torch.utils.data import DataLoader, TensorDataset
        loss_before = self._compute_loss_on_uploaded_data(self.global_model)
        training_loss, epoch_losses = self.continual_learner.train_with_mab_selection(
            global_data_batches, num_epochs=Config.NUM_EPOCH
        )
        loss_after = self._compute_loss_on_uploaded_data(self.global_model)

        # 训练完成后，根据MAB统计信息更新缓存质量评分
        self._update_cache_with_mab_scores(batch_mapping)

        self.visualize.plot_training_loss(
            epoch_losses,
            save_plot=True,
            plot_name=f"training_loss_session_{session}.png",
        )

        return {
            "loss_before": loss_before,
            "loss_after": loss_after,
            "training_loss": training_loss,
            "global_dataset_size": global_dataset_size * Config.BATCH_SIZE,
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
                vehicle_id = mapping['vehicle_id']
                data_type = mapping['data_type']
                if vehicle_id not in vehicle_scores:
                    vehicle_scores[vehicle_id] = {'old': [], 'new': []}
                vehicle_scores[vehicle_id][data_type].append({
                'local_batch_idx': mapping['local_batch_idx'],
                'quality_score': quality_score
            })
        # 更新每个车辆缓存的质量评分
        for vehicle_id, scores_by_type in vehicle_scores.items():
            cache = self.cache_manager.get_vehicle_cache(vehicle_id)

            # 合并所有质量评分，按原始顺序存储
            all_quality_scores = []

            # 处理旧数据质量评分（按原始顺序）
            old_scores_info = scores_by_type['old']
            if old_scores_info:
                old_scores_info.sort(key=lambda x: x['local_batch_idx'])
                old_quality_scores = [info['quality_score'] for info in old_scores_info]
                all_quality_scores.extend(old_quality_scores)

            # 处理新数据质量评分（按原始顺序）
            new_scores_info = scores_by_type['new']
            if new_scores_info:
                new_scores_info.sort(key=lambda x: x['local_batch_idx'])
                new_quality_scores = [info['quality_score'] for info in new_scores_info]
                all_quality_scores.extend(new_quality_scores)

            # 验证并更新缓存
            total_expected_batches = len(cache["old_data"]) + len(cache["new_data"])
            if len(all_quality_scores) == total_expected_batches:
                cache["quality_scores"] = all_quality_scores
            else:
                print(f"警告: 车辆 {vehicle_id} 质量评分数量不匹配 - "
                    f"预期: {total_expected_batches}, 实际: {len(all_quality_scores)}")

            # 统计信息
            total_batches = len(old_scores_info) + len(new_scores_info)
            all_scores = [info['quality_score'] for info in old_scores_info + new_scores_info]

            print(f"车辆 {vehicle_id} 质量评分更新: {total_batches} 个批次参与训练, "
                f"平均质量 {np.mean(all_scores):.4f}")

        import json
        cache_stats = self.cache_manager.get_cache_stats()
        formatted_stats = json.dumps(cache_stats, indent=2, ensure_ascii=False, default=str)
        print("缓存更新后车辆的缓存信息:")
        print(formatted_stats)

    def _compute_loss_on_uploaded_data(self, model):
        """计算模型在上传数据上的损失"""
        total_loss = 0.0
        batch_count = 0

        for vehicle in self.vehicle_env.vehicles:
            if vehicle.uploaded_data:
                for batch in vehicle.uploaded_data:
                    loss = self._compute_batch_loss(model, batch)
                    total_loss += loss
                    batch_count += 1

        return total_loss / batch_count if batch_count > 0 else 1.0

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

    def _evaluate_model_performance(self, session):
        """评估模型性能"""
        current_domain = self.data_simulator.get_current_domain()

        # 评估当前域性能
        current_test_data = self.data_simulator.get_test_dataset(current_domain)
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
            "current_domain": current_domain,
        }
        print(f"eval_results:{eval_results}")
        return eval_results

    def _calculate_reward_and_optimize(
        self, state, action, eval_results, comm_results, training_results
    ):
        """根据原始建模重新设计奖励计算函数"""
        # 获取必要的参数
        total_delay = comm_results["total_delay"]
        global_dataset_size = training_results.get("global_dataset_size", 1)
        print(f"total_delay:{total_delay}")
        print(f"global_dataset_size:{global_dataset_size}")

        # 计算损失降幅
        total_loss_reduction = 0
        total_upload_samples = 0

        for vehicle in self.vehicle_env.vehicles:
            if vehicle.uploaded_data:
                # 计算该车辆上传的样本数
                vehicle_upload_samples = len(vehicle.uploaded_data) * Config.BATCH_SIZE
                total_upload_samples += vehicle_upload_samples

                # 这里需要计算模型在上传数据上的损失变化
                # 由于我们实际训练前后没有分别记录每个车辆上传数据的损失
                # 我们使用训练前后的整体损失变化作为近似
                loss_before = training_results.get("loss_before", 1.0)
                loss_after = training_results.get("loss_after", 1.0)
                loss_reduction = loss_before - loss_after

                total_loss_reduction += vehicle_upload_samples * loss_reduction

        # 计算奖励
        if total_delay > 0 and global_dataset_size > 0:
            reward = total_loss_reduction / (global_dataset_size * total_delay)
        else:
            reward = 0.0
        print(f"reward:{reward}")
        # 获取下一个状态
        next_state = self._get_environment_state()

        # 将动作转换为向量形式
        vector = []
        for i in range(Config.NUM_VEHICLES):
            upload_batches = action["upload_decisions"][i][1]
            bandwidth_ratio = action["bandwidth_allocations"][i]
            vector.extend([upload_batches, bandwidth_ratio])
        action_vector = np.array(vector, dtype=np.float32)

        # 存储经验并优化DRL模型
        self.drl_agent.memory.push(state, action_vector, reward, next_state, False)
        if len(self.drl_agent.memory) >= Config.DRL_BATCH_SIZE:
            self.drl_agent.optimize_model()

        print(
            f"奖励计算 - 损失降幅: {total_loss_reduction:.4f}, 时延: {total_delay:.2f}s, 奖励: {reward:.4f}"
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
