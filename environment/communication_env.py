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

    def __init__(self, vehicle_env=None):
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
        self.samples_of_per_batch = config.SAMPLES_OF_BATCH  # |b_v^s|，每个批次包含的样本数

        # 计算参数
        self.golden_model_computation = 1e6  # 黄金模型处理一个样本的计算周期数，1
        self.global_model_computation = 2e6  # 全局模型处理一个样本的计算周期数，1
        self.edge_server_computation = 10e9  # C，边缘服务器计算能力（Cycles/s），1

        # 训练参数
        self.training_epochs = config.NUM_EPOCH  # E，训练轮次
        self.cache_samples_per_vehicle = (
            config.MAX_LOCAL_BATCHES * config.SAMPLES_OF_BATCH
        )  # |D_v|，每车缓存样本数

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
        distance = np.linalg.norm(vehicle.position - base_station["position"])
        # 避免除零
        distance = max(distance, 1.0)
        # 计算路径损耗部分: G_0 * (d_v^s)^{-α}
        path_loss_component = self.reference_gain * (
            distance ** (-self.path_loss_exponent)
        )
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

    def calculate_uplink_rate(self, vehicle, base_station, bandwidth_ratio, session_id):
        """
        计算上行链路通信速率 R_{uplink}^s
        公式: R_{uplink}^s = β_v^s * B_base * log₂(1 + (P_v * g_v^s) / (δ² + I_v))
        输入：
            bandwidth_ratio：车辆v的带宽分配比例
        """
        if vehicle is None or base_station is None:
            return 0.0

        channel_gain = self.calculate_channel_gain(vehicle, base_station, session_id)
        signal_power = self.vehicle_transmit_power * channel_gain
        noise_plus_interference = self.noise_power + self.I_v
        snr = signal_power / max(noise_plus_interference, 1e-20)  # 防止(δ² + I_v)为0

        allocated_bandwidth = bandwidth_ratio * self.base_bandwidth
        uplink_rate = allocated_bandwidth * np.log2(1 + snr)

        return max(uplink_rate, 0)

    def calculate_downlink_rate(self, session_id):
        """
        计算下行链路广播速率 R_{downlink}^s
        公式: R_{downlink}^s = B_base * log₂(1 + (P_b * min_{v∈V} g_v^s) / (δ² + I_v))
        """
        if self.vehicle_env is None:
            return 0.0

        min_channel_gain = float('inf')

        for vehicle in self.vehicle_env.vehicles:
            if vehicle.bs_connection is not None:
                base_station = self.vehicle_env._get_base_station_by_id(vehicle.bs_connection)
                if base_station:
                    channel_gain = self.calculate_channel_gain(vehicle, base_station, session_id)
                    min_channel_gain = min(min_channel_gain, channel_gain)

        if min_channel_gain == float('inf'):
            min_channel_gain = 1e-6

        signal_power = self.base_station_power * min_channel_gain
        noise_plus_interference = self.noise_power + self.I_v
        snr = signal_power / max(noise_plus_interference, 1e-20)
        downlink_rate = self.base_bandwidth * np.log2(1 + snr)

        return max(downlink_rate, 0)

    def _calculate_total_upload_data(self, upload_decisions):
        """
        计算总上传数据量
        输入：
            upload_decision：内部元素为（vehicle_id, upload_batches）,其中upload_batches 是该车辆在当前训练阶段决定上传的数据批次数量。
        """
        total_data_bits = 0
        for vehicle_id, upload_batches in upload_decisions:
            total_data_bits += upload_batches * self.samples_of_per_batch * self.sample_size

        return total_data_bits

    def calculate_transmission_delay(self, upload_decisions, bandwidth_allocations, session_id):
        """
        计算数据传输时延 t_trans

        公式: t_trans = max_{v∈V} [ (m_v^s * |b_v^s| * b0) / R_{uplink}^s ]

        参数:
            upload_decisions: 上传决策列表，每个元素为 (vehicle_id, upload_batches)
            bandwidth_allocations: 带宽分配字典 {vehicle_id: bandwidth_ratio}
            session_id: 训练会话ID

        返回:
            float: 数据传输时延 (秒)
        """
        max_delay = 0.0
        for vehicle_id, upload_batches in upload_decisions:
            if upload_batches == 0:
                continue
            # 获取车辆对象
            vehicle = self.vehicle_env._get_vehicle_by_id(vehicle_id)
            if not vehicle or vehicle.bs_connection is None:
                continue
            # 获取基站
            base_station = self.vehicle_env._get_base_station_by_id(vehicle.bs_connection)
            if not base_station:
                continue
            # 获取带宽分配比例
            bandwidth_ratio = bandwidth_allocations.get(vehicle_id, 0.0)  # 从bandwidth_allocations字典中获取键为vehicle_id的值，如果这个键不存在，则返回默认0.0
            if bandwidth_ratio <= 0:
                continue
            # 计算上行速率
            uplink_rate = self.calculate_uplink_rate(vehicle, base_station, bandwidth_ratio, session_id)
            if uplink_rate <= 0:
                continue
            # 计算单个车辆上传数据量：m_v^s * |b_v^s| * b0
            data_size_bits = upload_batches * self.samples_of_per_batch * self.sample_size
            # 计算传输时延
            transmission_delay = data_size_bits / uplink_rate
            # 取最大值
            max_delay = max(max_delay, transmission_delay)

        return max_delay






if __name__ == "__main__":
    env = VehicleEnvironment()
    comm_system = CommunicationSystem(env)
    vehicle = env.vehicles[0]
    base_station = env.base_stations[0]
    a = comm_system.calculate_uplink_rate(
        vehicle, base_station, 0.2, session_id = 1
    )
    b = comm_system.calculate_downlink_rate(
        session_id = 1
    )
    print(f"上行链路通信速率 a:{a}，下行链路通信速率 b:{b}")

