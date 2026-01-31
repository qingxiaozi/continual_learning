import torch
import random
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
        self.global_model = GlobalModel(self.data_simulator.current_dataset, auto_load=True)

        # 初始化学习组件
        self.cache_manager = CacheManager()
        self.continual_learner = ContinualLearner(self.global_model, self.gold_model)

        # 初始化车辆环境
        self.vehicle_env = VehicleEnvironment(
            self.global_model, self.gold_model, self.cache_manager, self.data_simulator
        )

        # 初始化通信系统
        self.communication_system = CommunicationSystem(self.vehicle_env)

        # 初始化MAB选择器
        self.mab_selector = MABDataSelector(num_arms=Config.MAX_LOCAL_BATCHES)

        # 初始化DRL智能体
        state_dim = 3 * Config.NUM_VEHICLES  # 置信度、测试损失、质量评分
        self.drl_agent = DRLAgent(state_dim)

        self.current_domain = self.data_simulator.get_current_domain()
        self.session_history = []
        self.visualize = ResultVisualizer()

    def run_single_episode(self, episode_id):
        """执行一个 RL episode：包含多个 steps（sessions）"""
        num_sessions = Config.NUM_TRAINING_SESSIONS
        episode_reward = 0

        for session in range(num_sessions):
            print(f"\n=== 训练会话 {session + 1}/{num_sessions} ===")
            # 步骤1: 更新会话和环境
            available_batches = self._update_session_environment(session)

            # 步骤2: 获取环境状态
            state = self._get_environment_state()

            # 步骤4：集成决策（上传批次 + 带宽分配）
            batch_choices, bandwidth_ratios = self._integrated_decision(state, available_batches, session)

            # 步骤5: 执行数据上传
            upload_results = self._upload_datas(batch_choices)

            # 步骤6: 缓存管理和数据选择
            self._manage_cache_and_data_selection()

            # 步骤7: 模型训练和更新
            training_results = self._train_and_update_global_model(session)

            # 步骤8：计算通信时延
            communication_results = self._calculate_communication_delay(
                batch_choices, bandwidth_ratios, session, training_results
            )

            # 步骤9: 性能评估
            evaluation_results = self._evaluate_model_performance()

            # 步骤10: 计算奖励
            reward = self._calculate_reward(
                communication_results,
                training_results
            )

            episode_reward += reward

            # 步骤11: 获取下一个状态（执行动作后的状态）
            next_state = self._get_environment_state()

            # 步骤12: 判断episode是否结束
            done = (session == num_sessions - 1)

            # 步骤13: 存储经验（关键修改：传入正确的done值）
            self.drl_agent.store_experience(
                state,
                batch_choices,
                reward,
                next_state,
                done  # 不再是固定的False
            )

            # 步骤14: 优化DRL模型
            if len(self.drl_agent.memory) >= Config.DRL_BATCH_SIZE:
                self.drl_agent.optimize_model()

            # 步骤15: 记录结果
            self._record_session_results(
                session, evaluation_results, communication_results, training_results
            )

            # 步骤16: 模型广播和更新
            self._broadcast_and_update_models()

            # 如果episode结束，跳出循环
            if done:
                print(f"Episode {episode_id+1} 在第{session+1}步结束")
                break

        # 最终评估和结果汇总
        self._final_evaluation_and_summary()
        print(f"Episode {episode_id+1} 总奖励: {episode_reward:.4f}")

        return episode_reward

    def _update_session_environment(self, session):
        """更新会话和环境状态"""
        domain_changed, previous_domain, current_domain = self.data_simulator.update_session_dataset(session)

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

    def _get_environment_state(self):
        """获取环境状态用于DRL决策"""
        state = self.vehicle_env.get_environment_state()
        return state

    def _integrated_decision(self, state, available_batches, session):
        """集成决策：DRL选择批次 + 带宽分配优化"""
        # 1. DRL选择批次
        batch_choices = self.drl_agent.select_action(
            state, available_batches=available_batches
        )

        # 2. 带宽分配优化
        allocator = BandwidthAllocator(
            batch_choices,
            communication_system=self.communication_system,
            vehicle_env=self.vehicle_env
        )

        # 使用优化分配方法
        bandwidth_ratios, min_max_delay = allocator.allocate_minmaxdelay_bandwidth(session)

        print(f"\n决策结果")
        print(f"总上传批次: {sum(batch_choices)}")
        print(f"上传批次: {batch_choices}")
        print(f"带宽分配: {bandwidth_ratios}")
        return batch_choices, bandwidth_ratios

    def _upload_datas(self, batch_choices):
        uploaded_data = {}
        for vehicle_id, n in enumerate(batch_choices):
            vehicle = self.vehicle_env._get_vehicle_by_id(vehicle_id)
            if n == 0:
                selected = []
            else:
                selected = random.sample(vehicle.data_batches, n)
                uploaded_data[vehicle_id] = selected
            vehicle.set_uploaded_data(selected)
        return uploaded_data

    def _manage_cache_and_data_selection(self):
        """缓存管理和数据选择"""
        for vehicle in self.vehicle_env.vehicles:
            # 如果没有uploaded_data，则不进行缓存操作
            if not vehicle.uploaded_data:
                continue

            # 更新缓存
            self.cache_manager.update_cache(
                vehicle.id, vehicle.uploaded_data, quality_scores=None
            )

        # 缓存统计
        cache_stats = self.cache_manager.get_cache_stats()
        new_batches = sum([stats["new_data_size"] for stats in cache_stats.values()])
        old_batches = sum([stats["old_data_size"] for stats in cache_stats.values()])
        total_batches = sum([stats["total_size"] for stats in cache_stats.values()])

        print(f"\n缓存管理")
        print(f"当前缓存总批次: {total_batches}")
        print(f"缓存中新数据批次：{new_batches}")
        print(f"缓存中旧数据批次：{old_batches}")

    def _calculate_communication_delay(self, batch_choices, bandwidth_ratios, session, training_results=None):
        """计算通信时延 - 使用分配器的计算方法"""
        # 构建带宽分配字典
        bandwidth_allocations = {}
        for i, ratio in enumerate(bandwidth_ratios):
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

        # 初始化默认值
        epoch_ce_losses = []
        val_losses = []
        epoch_elastic_losses = []
        actual_epochs = 0  # 默认使用配置值

        if training_result is not None:
            epoch_elastic_losses = training_result.get("epoch_elastic_losses")
            training_ce_loss = training_result.get("training_ce_loss")
            val_losses = training_result.get("val_losses")
            epoch_ce_losses = training_result.get("epoch_ce_losses")
            actual_epochs = training_result.get("actual_epochs")

        loss_after = self._compute_weighted_loss_on_uploaded_data(self.global_model)

        # 训练完成后，根据MAB统计信息更新缓存质量评分
        self._update_cache_with_mab_scores(batch_mapping)

        self.visualize.plot_training_loss(
            epoch_ce_losses,
            val_losses,
            epoch_elastic_losses,
            save_plot=True,
            plot_name=f"training_loss_session_{session}.png",
        )

        return {
            "loss_before": loss_before,
            "loss_after": loss_after,
            "training_loss": training_ce_loss,
            "actual_epochs": actual_epochs,
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

    def _compute_weighted_loss_on_uploaded_data(self, model):
        """计算模型在所有车辆上传数据上的损失和"""
        total_loss = 0.0

        for vehicle in self.vehicle_env.vehicles:
            if vehicle.uploaded_data:
                vehicle_data_size = len(vehicle.uploaded_data) * Config.BATCH_SIZE
                for batch in vehicle.uploaded_data:
                    loss = self._compute_batch_loss(model, batch)
                    total_loss += loss * Config.BATCH_SIZE

        return total_loss if total_loss > 0 else 1.0

    def _compute_batch_loss(self, model, batch):
        """计算单个批次所有样本损失的平均值"""
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

            device = next(model.parameters()).device  # 获取模型所在设备
            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

        return loss.item()

    def _evaluate_model_performance(self):
        """评估模型性能"""
        evaluation_results = self.data_simulator.evaluate_model(
            self.global_model, strategy="cumulative"
        )
        # 提取核心指标
        if evaluation_results is None:
            print("警告: 评估结果为空")
            eval_results = self._get_default_eval_results()
        elif "metrics" in evaluation_results:
            metrics = evaluation_results["metrics"]
            eval_results = {
                # 四个核心连续学习指标
                "AA": metrics["AA"],      # 平均准确率
                "AIA": metrics["AIA"],    # 平均增量准确率
                "FM": metrics["FM"],      # 遗忘度量
                "BWT": metrics["BWT"],    # 反向迁移
                "current_task": metrics["k"],  # 当前任务数
                "current_domain": self.current_domain,
            }
            # 保留当前域的详细结果
            if "current_domain" in evaluation_results:
                current_domain_results = evaluation_results["current_domain"]
                eval_results["current_domain_accuracy"] = current_domain_results.get("accuracy", 0.0)
                eval_results["current_domain_loss"] = current_domain_results.get("loss", 0.0)
        else:
            print("无指标信息")

        print(f"连续学习指标 (任务 {eval_results.get('current_task', 0)}):")
        print(f"  AA: {eval_results.get('AA', 0.0):.4f}   - 当前模型平均准确率")
        print(f"  AIA: {eval_results.get('AIA', 0.0):.4f}  - 平均增量准确率")
        print(f"  FM: {eval_results.get('FM', 0.0):.4f}   - 遗忘度量 (越小越好)")
        print(f"  BWT: {eval_results.get('BWT', 0.0):.4f}  - 反向迁移 (正为好)")
        print(f"  current_domain_accuracy: {eval_results.get('current_domain_accuracy', 0.0):.4f}")
        print(f"  current_domain_loss: {eval_results.get('current_domain_loss', 0.0):.4f}")

        return eval_results

    def _calculate_reward(
        self, comm_results, training_results
    ):
        """根据原始建模重新设计奖励计算函数"""
        # 获取必要的参数
        total_delay = comm_results["total_delay"]
        global_dataset_size = comm_results["total_samples"]

        total_loss_reduction = training_results.get("loss_before", 1.0) - training_results.get("loss_after", 1.0)

        # 计算奖励
        if total_delay > 0 and global_dataset_size > 0:
            reward = total_loss_reduction / (global_dataset_size * total_delay)
        else:
            reward = 0.0

        return reward

    def _record_session_results(self, session, eval_results, comm_results, training_results):
        """
        记录每个session的结果并进行整合，调用可视化函数

        Args:
            session: 当前session编号
            eval_results: 评估结果字典
            comm_results: 通信结果字典
            training_results: 训练结果字典
        """
        # 整合所有结果
        session_result = {
            "session": session,

            # 评估指标
            "AA": eval_results.get("AA", 0.0),           # 平均准确率
            "AIA": eval_results.get("AIA", 0.0),         # 平均增量准确率
            "FM": eval_results.get("FM", 0.0),           # 遗忘度量
            "BWT": eval_results.get("BWT", 0.0),         # 反向迁移
            "current_task": eval_results.get("current_task", 0),  # 当前任务数
            "current_domain": eval_results.get("current_domain", ""),  # 当前域

            # 当前域的性能
            "current_domain_accuracy": eval_results.get("current_domain_accuracy", 0.0),
            "current_domain_loss": eval_results.get("current_domain_loss", 0.0),

            # 训练指标
            "loss_before": training_results.get("loss_before", 0.0),
            "loss_after": training_results.get("loss_after", 0.0),
            "training_loss": training_results.get("training_loss", 0.0),
            "actual_epochs": training_results.get("actual_epochs", 0),  # 实际训练epoch数

            # 通信指标
            "total_delay": comm_results.get("total_delay", 0.0),
            "total_samples": comm_results.get("total_samples", 0),
        }

        # 将结果添加到历史记录中
        self.session_history.append(session_result)

        self.visualize.plot_data_heterogeneity(self.data_simulator, session)
        # 当有足够的数据时进行可视化
        if len(self.session_history) >= 2:
            self.visualize.save_all_plots(
                self.session_history,
                prefix=f"session{session}_"
            )

        return session_result

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

        if not self.session_history:
            print("没有可汇总的session结果")
            return

        # 从session历史中提取数据
        sessions = [r["session"] for r in self.session_history]
        last_session = sessions[-1]

        # 核心连续学习指标汇总
        print("\n=== 核心持续学习指标汇总 ===")

        # 获取最后session的指标
        final_aa = self.session_history[-1].get("AA", 0.0)
        final_aia = self.session_history[-1].get("AIA", 0.0)
        final_fm = self.session_history[-1].get("FM", 0.0)
        final_bwt = self.session_history[-1].get("BWT", 0.0)
        final_domain_acc = self.session_history[-1].get("current_domain_accuracy", 0.0)

        # 计算平均值
        aa_values = [r.get("AA", 0.0) for r in self.session_history]
        aia_values = [r.get("AIA", 0.0) for r in self.session_history]
        fm_values = [r.get("FM", 0.0) for r in self.session_history]
        bwt_values = [r.get("BWT", 0.0) for r in self.session_history]
        domain_acc_values = [r.get("current_domain_accuracy", 0.0) for r in self.session_history]

        avg_aa = np.mean(aa_values)
        avg_aia = np.mean(aia_values)
        avg_fm = np.mean(fm_values)
        avg_bwt = np.mean(bwt_values)
        avg_domain_acc = np.mean(domain_acc_values)

        print(f"最终会话 ({last_session}) 指标:")
        print(f"  AA: {final_aa:.4f} (平均准确率)")
        print(f"  AIA: {final_aia:.4f} (平均增量准确率)")
        print(f"  FM: {final_fm:.4f} (遗忘度量)")
        print(f"  BWT: {final_bwt:.4f} (反向迁移)")
        print(f"  当前域准确率: {final_domain_acc:.4f}")

        print(f"\n所有会话平均指标:")
        print(f"  平均AA: {avg_aa:.4f}")
        print(f"  平均AIA: {avg_aia:.4f}")
        print(f"  平均FM: {avg_fm:.4f} (越小越好)")
        print(f"  平均BWT: {avg_bwt:.4f} (正为好)")
        print(f"  平均当前域准确率: {avg_domain_acc:.4f}")

        # 通信和训练指标汇总
        print("\n=== 通信与训练指标汇总 ===")

        # 通信时延
        delays = [r.get("total_delay", 0.0) for r in self.session_history]
        total_delay = np.sum(delays)
        avg_delay = np.mean(delays)
        max_delay = np.max(delays)
        min_delay = np.min(delays)

        print(f"总通信时延: {total_delay:.2f}s")
        print(f"平均通信时延: {avg_delay:.2f}s")
        print(f"最大通信时延: {max_delay:.2f}s (会话 {np.argmax(delays)})")
        print(f"最小通信时延: {min_delay:.2f}s (会话 {np.argmin(delays)})")

        # 样本统计
        total_samples = [r.get("total_samples", 0) for r in self.session_history]
        avg_samples = np.mean(total_samples)
        total_all_samples = np.sum(total_samples)

        print(f"总训练样本数: {total_all_samples:,}")
        print(f"平均每会话样本数: {avg_samples:.0f}")

        # 训练损失变化
        loss_before_values = [r.get("loss_before", 1.0) for r in self.session_history]
        loss_after_values = [r.get("loss_after", 1.0) for r in self.session_history]

        final_improvement = (loss_before_values[-1] - loss_after_values[-1]) / loss_before_values[-1] * 100 if loss_before_values[-1] > 0 else 0
        avg_improvement = np.mean([
            (before - after) / before * 100 if before > 0 else 0
            for before, after in zip(loss_before_values, loss_after_values)
        ])

        print(f"\n训练损失改进:")
        print(f"  最终损失改进: {final_improvement:.1f}%")
        print(f"  平均损失改进: {avg_improvement:.1f}%")

        # 训练效率
        actual_epochs = [r.get("actual_epochs", 0) for r in self.session_history]
        total_epochs = np.sum(actual_epochs)
        avg_epochs = np.mean(actual_epochs)

        print(f"\n训练效率:")
        print(f"  总训练轮次: {total_epochs}")
        print(f"  平均每会话轮次: {avg_epochs:.1f}")

        # 域性能趋势分析
        print("\n=== 域性能趋势分析 ===")

        # 按域分组性能
        domain_performance = {}
        for result in self.session_history:
            domain = result.get("current_domain", "unknown")
            accuracy = result.get("current_domain_accuracy", 0.0)

            if domain not in domain_performance:
                domain_performance[domain] = []
            domain_performance[domain].append(accuracy)

        for domain, accuracies in domain_performance.items():
            if accuracies:
                avg_acc = np.mean(accuracies)
                min_acc = np.min(accuracies)
                max_acc = np.max(accuracies)
                print(f"  域 '{domain}': 平均={avg_acc:.4f}, 范围=[{min_acc:.4f}, {max_acc:.4f}]")

        # 输出关键结论
        print("\n=== 实验关键结论 ===")
        print("1. 连续学习性能:")
        if avg_fm < 0.2:
            print(f"   ✓ 遗忘控制良好 (FM={avg_fm:.4f} < 0.2)")
        else:
            print(f"   ✗ 存在遗忘问题 (FM={avg_fm:.4f} >= 0.2)")

        if avg_bwt > 0:
            print(f"   ✓ 具有正向知识迁移 (BWT={avg_bwt:.4f} > 0)")
        else:
            print(f"   ✗ 存在负迁移 (BWT={avg_bwt:.4f} <= 0)")

        print("2. 系统效率:")
        print(f"   - 平均通信开销: {avg_delay:.2f}s/会话")
        print(f"   - 平均训练样本: {avg_samples:.0f}/会话")
        print(f"   - 平均训练轮次: {avg_epochs:.1f}/会话")

        print("\n" + "=" * 60)

    def train_rl_agent(self):
        """完整的强化学习训练循环"""

        num_episodes = Config.NUM_EPISODES
        print(f"\n{'='*60}")
        print(f"开始强化学习训练，共{num_episodes}个episodes")
        print(f"{'='*60}")

        self.drl_agent.set_train_mode()
        # 初始化记录
        self.episode_rewards = []

        for episode in range(num_episodes):
            # 执行一个episode
            episode_data = self.run_single_episode(episode_id=episode)

            # 记录episode结果
            self.episode_rewards.append(episode_data)

            # 定期更新目标网络
            if episode % Config.TARGET_UPDATE_INTERVAL == 0:
                self.drl_agent.hard_update_target_network()
                print(f"已更新目标网络")

            # 打印训练进度
            if episode % 10 == 0:
                recent_rewards = self.episode_rewards[-10:] if len(self.episode_rewards) >= 10 else self.episode_rewards
                avg_reward = np.mean(recent_rewards) if recent_rewards else 0
                print(f"\n[训练进度] Episode {episode+1}/{num_episodes}")
                print(f"  最近10个episode平均奖励: {avg_reward:.4f}")
                print(f"  经验回放缓冲区大小: {len(self.drl_agent.memory)}")

        print(f"\n训练完成！")

        # 保存训练好的模型
        self.drl_agent.save_model("trained_drl_model.pth")

        return self.episode_rewards


if __name__ == "__main__":
    a = BaselineComparison()
    a.train_rl_agent()