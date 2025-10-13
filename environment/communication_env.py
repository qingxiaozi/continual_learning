import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.parameters import config
from environment.vehicle_env import VehicleEnvironment


class CommunicationSystem:
    """
    通信系统类
    负责计算车路协同系统中的各种通信时延，包括：
    - 数据传输时延 (t_trans)
    - 数据标注时延 (t_label)
    - 模型重训练时延 (t_retrain)
    - 模型广播时延 (t_broadcast)
    """
    def __init__(self, vehicle_env = None):
        # 环境引用
        self.vehicle_env = vehicle_env

        # 通信基本参数
        self.base_bandwidth = config.BASE_BANDWIDTH  # B_base，基础带宽 20 MHz
        self.noise_power = config.NOISE_POWER  # δ²，噪声功率 (W)，1e-10
        self.I_v = config.INTERFERENCE_POWER  # 干扰功率 (W)，常数
        self.base_station_power = config.BS_TRANSMIT_POWER  # p_b，基站发射功率 (W)
        self.vehicle_transmit_power = config.VEHICLE_TRANSMIT_POWER  # 车辆发射功率 (W)

        # 信道模型参数
        self.reference_gain = config.REFERENCE_GAIN  # G_0，参考距离（1m)处的路径增益
        self.path_loss_exponent = config.PATH_LOSS_EXPONENT  # α: 路径损耗指数 2.7
        self.shadowing_std = config.SHADOWING_STD  # 阴影衰落标准（8 dB）

        # 数据参数
        self.sample_size = 1024  # b0，单个样本的大小（bits），1
        self.samples_of_batch = config.SAMPLES_OF_BATCH  # |b_v^s|，灭个批次包含的样本数

        # 计算参数
        self.golden_model_computation = 1e6  # 黄金模型处理一个样本的计算周期数，1
        self.global_model_computation = 2e6  # 全局模型处理一个样本的计算周期数，1
        self.edge_server_compution = 10e9  # C，边缘服务器计算能力（Cycles/s），1

        # 训练参数
        self.training_epochs = config.NUM_EPOCH  # E，训练轮次
        self.cache_samples_per_vehicle = config.MAX_LOCAL_BATCHES * config.SAMPLES_OF_BATCH  # |D_v|，每车缓存样本数

        # 模型参数
        self.model_parameter_size = 50e6  # P_m，模型参数量的大小（bit），1

        # 缓存阴影衰落值，避免重复计算
        self._shadowing_cache = {}

    def calculate_channel_gain(self, vehicle, base_station, session_id):
        """
        计算信道增益 g_v^s
        公式：g_v^s = G_0 * (d_v^s)^{-α} * ξ
        参数：
            vehicle：车辆对象
            base_station：基站字典
            session_id：训练会话ID
        返回：
            float：信道增益
        """
        if vehicle is None or base_station is None:
            return 0.0
        # 计算距离
        print(f"vehicle的位置为{vehicle.position}")
        print(f"基站的位置为{base_station['position']}")
        distance = np.linalg.norm(vehicle.position - base_station['position'])
        # 避免除零
        distance = max(distance, 1.0)
        # 计算路径损耗部分: G_0 * (d_v^s)^{-α}
        path_loss_component = self.reference_gain * (distance ** (-self.path_loss_exponent))
        # 生成阴影衰落 ξ (对数正态分布)
        shadowing_key = f"{vehicle.id}_{session_id}"
        if shadowing_key in self._shadowing_cache:
            shadowing_db = self._shadowing_cache[shadowing_key]
        else:
            shadowing_db = np.random.normal(0, self.shadowing_std)
            self._shadowing_cache[shadowing_key] = shadowing_db

        # 转换为线性值
        shadowing_linear = 10 ** (shadowing_db / 10)
        print(shadowing_linear)

        return path_loss_component * shadowing_linear



if __name__ == "__main__":
    env = VehicleEnvironment()
    comm_system = CommunicationSystem(env)
    vehicle = env.vehicles[0]
    base_station = env.base_stations[0]
    channel_gain = comm_system.calculate_channel_gain(vehicle, base_station, session_id = 1)




