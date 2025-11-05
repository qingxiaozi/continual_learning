
from environment.vehicle_env import VehicleEnvironment

if __name__ == "__main__":
    vehicle_env = VehicleEnvironment()
    num_sessions = 10
    print("完整的联合优化模型")
    state_dim = 3 * config.NUM_VEHICLES  # 置信度、测试损失、质量评分
    action_dim = 2 * config.NUM_VEHICLES  # 上传批次、带宽分配
    drl_agent = DRLAgent(state_dim, action_dim)

    # for session in range(num_sessions):
    #     print(f"Session {session + 1}/{num_sessions}")
    #     # 获取环境状态
    #     state = vehicle_env.get_environment_state()
