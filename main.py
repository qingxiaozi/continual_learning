import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.parameters import config
from environment.dataSimu_env import DomainIncrementalDataSimulator
from learning.cache_manager import cacheManager
from learning.evaluator import ModelEvaluator
from environment.vehicle_env import VehicleEnvironment
from models.drl_agent import DRLAgent
from models.global_model import globalModel
from models.gold_model import GoldModel
from models.mab_selector import MABDataSelector


def _calculate_real_mab_rewards(all_data, batch_arm_mapping, mab_selector, session):
    """计算真实的MAB奖励"""
    real_rewards = {}

    # 方法1: 基于模型性能改进的奖励
    if session > 0 and hasattr('prev_global_model'):
        # 保存当前模型状态
        current_model_state = global_model.state_dict().copy()

        for arm_id, batch in batch_arm_mapping.items():
            try:
                # 计算使用该批次前的模型性能
                before_accuracy, before_loss = evaluator.evaluate_model(
                    global_model, batch
                )

                # 使用该批次进行一次训练更新
                learner = ContinualLearner(global_model, gold_model)
                learner.train_on_dataset([batch], num_epochs=1)

                # 计算使用该批次后的模型性能
                after_accuracy, after_loss = evaluator.evaluate_model(
                    global_model, batch
                )

                # 奖励 = 性能改进 (损失下降或准确率提升)
                reward = (before_loss - after_loss) + (after_accuracy - before_accuracy)
                real_rewards[arm_id] = reward

            except Exception as e:
                print(f"Error calculating reward for arm {arm_id}: {e}")
                real_rewards[arm_id] = 0.0

            # 恢复模型状态，确保每个批次的评估是独立的
            global_model.load_state_dict(current_model_state)

    else:
        # 对于第一个session或没有前一个模型的情况，使用基于数据质量的启发式奖励
        for arm_id, batch in batch_arm_mapping.items():
            reward = _calculate_heuristic_reward(batch, session)
            real_rewards[arm_id] = reward

    # 保存当前模型状态供下一次使用
    prev_global_model = global_model.state_dict().copy()

    return real_rewards

def _calculate_heuristic_reward(batch, session):
    """计算基于数据质量的启发式奖励"""
    try:
        # 方法2: 基于数据多样性的奖励
        diversity_reward = _calculate_diversity_reward(batch)

        # 方法3: 基于数据置信度的奖励
        confidence_reward = _calculate_confidence_reward(batch)

        # 方法4: 基于数据新颖性的奖励 (与新domain的相关性)
        novelty_reward = _calculate_novelty_reward(batch, session)

        # 综合奖励
        total_reward = 0.4 * diversity_reward + 0.3 * confidence_reward + 0.3 * novelty_reward

        return max(0, min(1, total_reward))  # 限制在0-1范围内

    except Exception as e:
        print(f"Error calculating heuristic reward: {e}")
        return 0.5  # 默认奖励

def _calculate_diversity_reward(batch):
    """计算数据多样性奖励"""
    try:
        # 简化方法: 基于批次内样本的方差
        all_features = []
        for data in batch:
            if isinstance(data, (list, tuple)) and len(data) > 0:
                inputs = data[0]
            else:
                inputs = data

            # 提取特征 (使用模型的中间层或简单统计)
            with torch.no_grad():
                if hasattr(global_model, 'features'):
                    features = global_model.features(inputs)
                    features = features.view(features.size(0), -1)
                else:
                    # 如果没有特征提取器，使用原始数据的统计
                    features = inputs.view(inputs.size(0), -1)

                all_features.append(features.mean(dim=0))

        if len(all_features) > 1:
            features_tensor = torch.stack(all_features)
            diversity = torch.var(features_tensor, dim=0).mean().item()
            return min(1.0, diversity * 10)  # 缩放
        else:
            return 0.5
    except:
        return 0.5

def _calculate_confidence_reward(batch):
    """计算数据置信度奖励"""
    try:
        total_confidence = 0
        count = 0

        for data in batch:
            if isinstance(data, (list, tuple)) and len(data) > 0:
                inputs = data[0]
            else:
                inputs = data

            confidence = _calculate_batch_confidence(inputs)
            total_confidence += confidence
            count += 1

        return total_confidence / count if count > 0 else 0.5
    except:
        return 0.5

def _calculate_batch_confidence(inputs):
    """计算单个批次的置信度"""
    global_model.eval()
    with torch.no_grad():
        outputs = global_model(inputs)
        if hasattr(outputs, 'logits'):
            outputs = outputs.logits
        probabilities = torch.softmax(outputs, dim=1)
        confidence = torch.max(probabilities, dim=1)[0].mean().item()
        return confidence

def _calculate_novelty_reward(batch, session):
    """计算数据新颖性奖励"""
    try:
        # 基于当前domain和新domain的差异
        current_domain = data_simulator.get_domain_for_session(session)

        # 简化方法: 如果数据来自新domain，给予更高奖励
        domain_changes = [20, 40, 60, 80]
        is_new_domain = any(session == change for change in domain_changes)

        if is_new_domain:
            return 0.8  # 新domain数据有更高价值
        else:
            return 0.3  # 旧domain数据价值较低
    except:
        return 0.5


if __name__ == "__main__":
    global_model = globalModel("office31")
    gold_model = GoldModel("office31")
    cache_manager = cacheManager()
    data_simulator = DomainIncrementalDataSimulator()
    evaluator = ModelEvaluator(gold_model)

    vehicle_env = VehicleEnvironment(global_model, gold_model, cache_manager, data_simulator)
    num_sessions = 10
    print("完整的联合优化模型")
    state_dim = 3 * config.NUM_VEHICLES  # 置信度、测试损失、质量评分
    action_dim = 2 * config.NUM_VEHICLES  # 上传批次、带宽分配
    drl_agent = DRLAgent(state_dim, action_dim)
    mab_selector = MABDataSelector(num_arms = config.MAX_LOCAL_BATCHES * config.NUM_VEHICLES)

    for session in range(num_sessions):
        data_simulator.update_session(session)
        print(f"Session {session + 1}/{num_sessions}")
        # 获取环境状态
        state = vehicle_env.get_environment_state()
        print(f"state： {state}")
        # DRL智能体选择动作
        action = drl_agent.select_action(state)
        total_upload_size = 0
        upload_batches = action[::2].astype(int)
        print(f"upload_batches: {upload_batches}")
        bandwidth_allocations = action[1::2]
        print(f"bandwidth: {bandwidth_allocations}")

        # 存储每个车辆的数据批次和对应的MAB臂
        vehicle_batches_arms = {}

        for i, vehicle in enumerate(vehicle_env.vehicles):
            # 根据DRL决策上传数据
            num_batches = upload_batches[i]
            if num_batches > 0:
                new_data = data_simulator.generate_vehicle_data(
                    vehicle.id, num_batches=num_batches
                )
                # 为每个批次分配唯一的一个MAB臂ID
                batch_arms = []
                for j, batch in enumerate(new_data):
                    arm_id = vehicle.id * 5 + j  # 为每个数据批次分配一个id
                    batch_arms.append(arm_id)

                vehicle_batches_arms[vehicle.id] = {
                    'batches': new_data,
                    'arms': batch_arms
                }
                # 用初始质量评分更新缓存
                quality_scores = [0.5] * len(new_data)  # 初始质量评分
                cache_manager.update_cache(vehicle.id, new_data, quality_scores)
                total_upload_size += len(new_data) * config.BATCH_SIZE * 3 * 224 * 224

        # 训练全局模型并计算真实的MAB奖励
        all_data = []
        batch_arm_mapping = {}
        # 构建训练数据集和臂映射
        for vehicle_id, info in vehicle_batches_arms.items():
            for j, (batch, arm_id) in enumerate(zip(info['batches'], info['arms'])):
                all_data.append(batch)
                batch_arm_mapping[arm_id] = batch

        if all_data:
            # 在训练过程中计算真实MAB奖励
            real_rewards = _calculate_real_mab_rewards(
                all_data, batch_arm_mapping, mab_selector, session
            )

            # 使用真实奖励更新MAB选择器
            for arm_id, reward in real_rewards.items():
                mab_selector.update_arm(arm_id, reward)

            # 使用真实奖励更新缓存质量评分
            for vehicle_id, info in vehicle_batches_arms.items():
                vehicle_quality_scores = []
                for arm_id in info['arms']:
                    if arm_id in real_rewards:
                        # 将奖励转换为质量评分 (0-1范围)
                        quality_score = max(0, min(1, real_rewards[arm_id]))
                        vehicle_quality_scores.append(quality_score)

                if vehicle_quality_scores:
                    # 重新更新缓存，使用真实质量评分
                    cache_manager.update_cache(
                        vehicle_id,
                        info['batches'],
                        vehicle_quality_scores
                    )

        # 评估性能并计算奖励
        test_data = data_simulator.generate_vehicle_data(0, num_batches=2)
        accuracy, loss = evaluator.evaluate_model(global_model, test_data[0] if test_data else [])
        print(f"accuracy: {accuracy}, loss: {loss}")





        exit()
