import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.parameters import config
from environment.dataSimu_env import DomainIncrementalDataSimulator
from learning.cache_manager import cacheManager
from environment.vehicle_env import VehicleEnvironment
from models.drl_agent import DRLAgent
from models.global_model import globalModel
from models.gold_model import GoldModel
from models.mab_selector import MABDataSelector


if __name__ == "__main__":
    global_model = globalModel("office31")
    gold_model = GoldModel("office31")
    cache_manager = cacheManager()
    data_simulator = DomainIncrementalDataSimulator()

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
                # 为每个数据批次分配MAB臂
                batch_arms = []
                for j, batch in enumerate(new_data):
                    print(f"j:{j}")
                    arm_id = vehicle.id * 5 + j  # 为每个数据批次分配一个id
                    batch_arms.append(arm_id)

                vehicle_batches_arms[vehicle.id] = {
                    'batches': new_data,
                    'arms': batch_arms
                }
                # 暂时先用模拟质量评分更新缓存
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




        exit()
