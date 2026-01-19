import numpy as np
from config.parameters import Config
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
        self.base_bandwidth = Config.BASE_BANDWIDTH  # B_base，基础带宽 20 MHz
        self.noise_power = Config.NOISE_POWER  # δ²，噪声功率 (W)，1e-10
        self.I_v = Config.INTERFERENCE_POWER  # 干扰功率 (W)，常数
        self.base_station_power = Config.BS_TRANSMIT_POWER  # p_b，基站发射功率 (W)
        self.vehicle_transmit_power = Config.VEHICLE_TRANSMIT_POWER  # 车辆发射功率 (W)

        # 信道模型参数
        self.reference_gain = Config.REFERENCE_GAIN  # G_0，参考距离（1m)处的路径增益
        self.path_loss_exponent = Config.PATH_LOSS_EXPONENT  # α: 路径损耗指数 2.7
        self.shadowing_std = Config.SHADOWING_STD  # 阴影衰落标准（8 dB）

        # 数据参数
        self.sample_size = (
            Config.IMAGE_SIZE * Config.IMAGE_SIZE * 3 * 8
        )  # b0，单个样本的大小（bits），RGB 图像每个通道8位
        self.samples_of_per_batch = Config.BATCH_SIZE  # |b_v^s|，每个批次包含的样本数

        # 计算参数
        self.golden_model_computation = 2e6  # 黄金模型处理一个样本的计算周期数，1
        self.global_model_computation = 5e6  # 全局模型处理一个样本的计算周期数，1
        self.edge_server_computation = 2e10  # C，边缘服务器计算能力（Cycles/s），20GHZ

        # 训练参数
        self.training_epochs = Config.NUM_EPOCH  # E，训练轮次

        # 模型参数
        self.model_parameter_size = 3.578e8 / 4  # P_m，模型参数量的大小（bit），传输量化后的resnet18

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

        min_channel_gain = float("inf")

        for vehicle in self.vehicle_env.vehicles:
            if vehicle.bs_connection is not None:
                base_station = self.vehicle_env._get_base_station_by_id(
                    vehicle.bs_connection
                )
                if base_station:
                    channel_gain = self.calculate_channel_gain(
                        vehicle, base_station, session_id
                    )
                    min_channel_gain = min(min_channel_gain, channel_gain)

        if min_channel_gain == float("inf"):
            min_channel_gain = 1e-6

        signal_power = self.base_station_power * min_channel_gain
        noise_plus_interference = self.noise_power + self.I_v
        snr = signal_power / max(noise_plus_interference, 1e-20)
        downlink_rate = self.base_bandwidth * np.log2(1 + snr)

        return max(downlink_rate, 0)

    def calculate_transmission_delay(
        self, upload_decisions, bandwidth_allocations, session_id
    ):
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
            base_station = self.vehicle_env._get_base_station_by_id(
                vehicle.bs_connection
            )
            if not base_station:
                continue
            # 获取带宽分配比例
            bandwidth_ratio = bandwidth_allocations.get(
                vehicle_id, 0.0
            )  # 从bandwidth_allocations字典中获取键为vehicle_id的值，如果这个键不存在，则返回默认0.0
            if bandwidth_ratio <= 0:
                continue
            # 计算上行速率
            uplink_rate = self.calculate_uplink_rate(
                vehicle, base_station, bandwidth_ratio, session_id
            )
            if uplink_rate <= 0:
                continue
            # 计算单个车辆上传数据量：m_v^s * |b_v^s| * b0
            data_size_bits = (
                upload_batches * self.samples_of_per_batch * self.sample_size
            )
            # 计算传输时延
            transmission_delay = data_size_bits / uplink_rate
            # 取最大值
            max_delay = max(max_delay, transmission_delay)

        return max_delay

    def calculate_labeling_delay(self, upload_decisions):
        """
        计算数据标注时延 t_label
        公式：t_label = (c_golden * Σ_{v=1}^{|V|} (m_v^s * |b_v^s|)) / C
        参数:
            upload_decisions: 上传决策列表，每个元素为 (vehicle_id, upload_batches)

        返回:
            float: 数据标注时延 (秒)
        """
        total_samples = 0

        for vehicle_id, upload_batches in upload_decisions:
            total_samples += upload_batches * self.samples_of_per_batch
            # 计算总计算需求
        total_computation = total_samples * self.golden_model_computation
        # 计算标注时延
        labeling_delay = total_computation / self.edge_server_computation

        return labeling_delay

    def calculate_retraining_delay(self, total_samples, actual_epochs=None):
        """
        计算模型重训练时延 t_retrain
        公式: t_retrain = (E * |D_v| * |V| * c_global) / C
        参数:
            actual_epochs: 实际训练epoch数，默认为None，表示使用配置的训练轮次
            total_samples: 所有车辆缓存数据样本量 |D_v| * |V|
        返回:
            float: 模型重训练时延 (秒)
        """
        if total_samples <= 0:
            return 0.0

        if actual_epochs is None:
            actual_epochs = self.training_epochs

        total_computation = (
            actual_epochs * total_samples * self.global_model_computation
        )
        # 计算重训练时延
        retraining_delay = total_computation / self.edge_server_computation

        return retraining_delay

    def calculate_broadcast_delay(self, session_id):
        """
        计算模型广播时延 t_broadcast
        公式： t_broadcast = P_m / R_{downlink}^s
        参数:
            session_id: 训练会话ID
        返回:
            float: 模型广播时延 (秒)
        """
        # 计算下行广播速率
        downlink_rate = self.calculate_downlink_rate(session_id)
        if downlink_rate <= 0:
            return float("inf")
        # 计算广播时延
        broadcast_delay = self.model_parameter_size / downlink_rate

        return broadcast_delay

    def calculate_total_training_delay(
        self, upload_decisions, bandwidth_allocations, session_id, total_samples, actual_epochs=None
    ):
        """
        计算训练阶段总时延 T_s
        公式: T_s = t_trans + t_label + t_retrain + t_broadcast
        参数:
            upload_decisions: 上传决策列表，每个元素为 (vehicle_id, upload_batches)
            bandwidth_allocations: 带宽分配字典
            session_id: 训练会话ID
            total_samples: 实际训练样本总数
            actual_epochs: 实际训练epoch数（考虑早停）
        返回:
            dict: 包含各项时延和总时延的字典
        """
        # 计算各项时延
        t_trans = self.calculate_transmission_delay(
            upload_decisions, bandwidth_allocations, session_id
        )
        t_label = self.calculate_labeling_delay(upload_decisions)
        t_retrain = self.calculate_retraining_delay(total_samples, actual_epochs)
        t_broadcast = self.calculate_broadcast_delay(session_id)
        total_delay = t_trans + t_label + t_retrain

        # total_delay = t_trans + t_label + t_retrain + t_broadcast

        delay_breakdown = {
            "transmission_delay": t_trans,
            "labeling_delay": t_label,
            "retraining_delay": t_retrain,
            "broadcast_delay": t_broadcast,
            "total_delay": total_delay,
            "upload_data_size": self._calculate_total_upload_data(upload_decisions),
            "effective_uplink_rate": self._calculate_effective_uplink_rate(
                upload_decisions, bandwidth_allocations, session_id
            ),
            "effective_downlink_rate": self.calculate_downlink_rate(session_id),
        }

        return delay_breakdown

    def get_communication_statistics(self, session_id):
        """获取通信统计信息"""
        stats = {
            "session_id": session_id,
            "num_vehicles": len(self.vehicle_env.vehicles),
            "num_base_stations": len(self.vehicle_env.base_stations),
            "connected_vehicles": sum(
                1 for v in self.vehicle_env.vehicles if v.bs_connection is not None
            ),
            "average_channel_gain": self._calculate_average_channel_gain(session_id),
            "min_channel_gain": self._calculate_min_channel_gain(session_id),
            "max_channel_gain": self._calculate_max_channel_gain(session_id),
            # 'effective_uplink_rate': self._calculate_effective_uplink_rate(upload_decisions, bandwidth_allocations, session_id),
            "effective_downlink_rate": self.calculate_downlink_rate(session_id),
        }
        return stats

    def _calculate_total_upload_data(self, upload_decisions):
        """
        计算总上传数据量
        输入：
            upload_decision：内部元素为（vehicle_id, upload_batches）,其中upload_batches 是该车辆在当前训练阶段决定上传的数据批次数量。
        """
        total_data_bits = 0
        for vehicle_id, upload_batches in upload_decisions:
            total_data_bits += (
                upload_batches * self.samples_of_per_batch * self.sample_size
            )

        return total_data_bits

    def _calculate_effective_uplink_rate(
        self, upload_decisions, bandwidth_allocations, session_id
    ):
        """
        计算有效上行速率
        有效上传速率 = 计算总上传数据量 / 最大传输时间
        """
        total_data = self._calculate_total_upload_data(upload_decisions)
        t_trans = self.calculate_transmission_delay(
            upload_decisions, bandwidth_allocations, session_id
        )

        if t_trans > 0:
            return total_data / t_trans
        else:
            return 0.0

    def _calculate_average_channel_gain(self, session_id):
        """计算平均信道增益"""
        total_gain = 0.0
        count = 0

        for vehicle in self.vehicle_env.vehicles:
            if vehicle.bs_connection is not None:
                base_station = self.vehicle_env._get_base_station_by_id(
                    vehicle.bs_connection
                )
                if base_station:
                    gain = self.calculate_channel_gain(
                        vehicle, base_station, session_id
                    )
                    total_gain += gain
                    count += 1

        return total_gain / count if count > 0 else 0.0

    def _calculate_min_channel_gain(self, session_id):
        """计算最小信道增益"""
        min_gain = float("inf")

        for vehicle in self.vehicle_env.vehicles:
            if vehicle.bs_connection is not None:
                base_station = self.vehicle_env._get_base_station_by_id(
                    vehicle.bs_connection
                )
                if base_station:
                    gain = self.calculate_channel_gain(
                        vehicle, base_station, session_id
                    )
                    min_gain = min(min_gain, gain)

        return min_gain if min_gain != float("inf") else 0.0

    def _calculate_max_channel_gain(self, session_id):
        """计算最大信道增益"""
        max_gain = 0.0

        for vehicle in self.vehicle_env.vehicles:
            if vehicle.bs_connection is not None:
                base_station = self.vehicle_env._get_base_station_by_id(
                    vehicle.bs_connection
                )
                if base_station:
                    gain = self.calculate_channel_gain(
                        vehicle, base_station, session_id
                    )
                    max_gain = max(max_gain, gain)

        return max_gain


if __name__ == "__main__":
    print(Config.BASE_BANDWIDTH)
    # 创建环境和通信系统
    vehicle_env = VehicleEnvironment(None, None, None, None)
    comm_system = CommunicationSystem(vehicle_env)

    # 模拟一个训练阶段
    session_id = 1
    num_vehicles = len(vehicle_env.vehicles)

    # 创建上传决策（示例：每辆车上传2个批次）
    upload_decisions = [(vehicle.id, 2) for vehicle in vehicle_env.vehicles]

    # 创建带宽分配（示例：均匀分配）
    total_bandwidth = 1.0
    bandwidth_per_vehicle = total_bandwidth / num_vehicles
    bandwidth_allocations = {
        vehicle.id: bandwidth_per_vehicle for vehicle in vehicle_env.vehicles
    }
    total_samples = 20 * 3 * 32
    # 计算总时延
    delay_breakdown = comm_system.calculate_total_training_delay(
        upload_decisions, bandwidth_allocations, session_id, total_samples
    )

    # 输出结果
    print("=== 训练阶段时延分析 ===")
    print(f"数据传输时延: {delay_breakdown['transmission_delay']:.2f} 秒")
    print(f"数据标注时延: {delay_breakdown['labeling_delay']:.2f} 秒")
    print(f"模型重训练时延: {delay_breakdown['retraining_delay']:.2f} 秒")
    print(f"模型广播时延: {delay_breakdown['broadcast_delay']:.2f} 秒")
    print(f"总时延: {delay_breakdown['total_delay']:.2f} 秒")
    print(f"总上传数据: {delay_breakdown['upload_data_size'] / 1e6:.2f} Mbit")
    print(f"有效上行速率: {delay_breakdown['effective_uplink_rate'] / 1e6:.2f} Mbps")
    print(f"下行广播速率: {delay_breakdown['effective_downlink_rate'] / 1e6:.2f} Mbps")
