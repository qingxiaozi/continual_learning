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
        for i, vehicle in enumerate(vehicle_env.vehicles):
                # 根据DRL决策上传数据
                num_batches = upload_batches[i]
                if num_batches > 0:
                    new_data = data_simulator.generate_vehicle_data(
                        vehicle.id, num_batches=num_batches
                    )
                    # MAB数据质量评估


        exit()
