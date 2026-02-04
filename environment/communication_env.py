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

    def __init__(self, vehicle_env=None, global_model=None):
        # 环境引用
        self.vehicle_env = vehicle_env

        # 通信基本参数
        self.base_bandwidth = Config.BASE_BANDWIDTH  # B_base，基础带宽 20 MHz
        self.noise_power = Config.NOISE_POWER  # δ²，噪声功率 (W)，1e-10
        self.interference_power = Config.INTERFERENCE_POWER  # 干扰功率 (W)，常数
        self.bs_power = Config.BS_TRANSMIT_POWER  # p_b，基站发射功率 (W)
        self.vehicle_transmit_power = Config.VEHICLE_TRANSMIT_POWER  # 车辆发射功率 (W)

        # 信道模型参数
        self.g0 = Config.REFERENCE_GAIN  # G_0，参考距离（1m)处的路径增益
        self.alpha = Config.PATH_LOSS_EXPONENT  # α: 路径损耗指数 2.7
        self.shadowing_std = Config.SHADOWING_STD  # 阴影衰落标准（8 dB）

        # 数据参数
        self.sample_bits = (
            Config.IMAGE_SIZE * Config.IMAGE_SIZE * 3 * 8
        )  # b0，单个样本的大小（bits），RGB 图像每个通道8位
        self.batch_size = Config.BATCH_SIZE  # samples per batch

        # 计算参数
        self.c_golden = 2e6  # 黄金模型处理一个样本的计算周期数，1
        self.c_global = 5e6  # 全局模型处理一个样本的计算周期数，1
        self.C_edge = 2e10  # C，边缘服务器计算能力（Cycles/s），20GHZ

        # 训练参数
        self.training_epochs = Config.NUM_EPOCH  # E，训练轮次

        # 模型参数
        self.model = global_model
        self.model_param_bits = sum(p.numel() for p in self.model.parameters()) * 8  # 单位：bit

    def _get_vehicle(self, vid):
        return self.vehicle_env._get_vehicle_by_id(vid) if self.vehicle_env else None

    def _get_bs(self, bs_id):
        return self.vehicle_env._get_base_station_by_id(bs_id) if self.vehicle_env else None
    
    def calculate_channel_gain(self, vehicle, base_station):
        """
        计算信道增益 g_v^s
        公式：g_v^s = G_0 * (d_v^s)^{-α} * ξ
        参数：
            vehicle：车辆对象
            base_station：基站字典
        返回：
            float：信道增益
        """
        if not vehicle or not base_station:
            return 0.0
        # 计算距离
        d = max(np.linalg.norm(vehicle.position - base_station["utm_position"]), 1.0)
        # 计算路径损耗部分: G_0 * (d_v^s)^{-α}
        path_loss = self.g0 * (d ** (-self.alpha))
        # 生成阴影衰落 ξ (对数正态分布)
        shadowing_db = np.random.normal(0, self.shadowing_std)
        shadowing_linear = 10 ** (shadowing_db / 10)

        return path_loss * shadowing_linear

    def calculate_uplink_rate(self, vehicle, base_station, beta):
        """
        计算上行链路通信速率 R_{uplink}^s
        公式: R_{uplink}^s = β_v^s * B_base * log₂(1 + (P_v * g_v^s) / (δ² + I_v))
        输入：
            bandwidth_ratio：车辆v的带宽分配比例
        """
        g = self.calculate_channel_gain(vehicle, base_station)
        snr = (self.vehicle_transmit_power * g) / max(self.noise_power + self.interference_power, 1e-20)
        return max(beta * self.base_bandwidth * np.log2(1 + snr), 0.0)

    def calculate_downlink_rate(self):
        """
        Docstring for calculate_downlink_rate
        
        :param self: Description
        """
        if not self.vehicle_env:
            return 0.0
        min_g = float('inf')
        for v in self.vehicle_env.vehicles:
            if v.bs_connection is not None:
                bs = self._get_bs(v.bs_connection)
                if bs:
                    g = self.calculate_channel_gain(v, bs)
                    min_g = min(min_g, g)
        min_g = min_g if min_g != float('inf') else 1e-6
        snr = (self.bs_power * min_g) / max(self.noise_power + self.interference_power, 1e-20)
        return max(self.base_bandwidth * np.log2(1 + snr), 0.0)

    def _total_samples_from_decisions(self, upload_decisions):
        return sum(m * self.batch_size for _, m in upload_decisions)
    
    def calculate_transmission_delay(
        self, upload_decisions, bandwidth_allocations
    ):
        """
        计算数据传输时延 t_trans

        公式: t_trans = max_{v∈V} [ (m_v^s * |b_v^s| * b0) / R_{uplink}^s ]

        参数:
            upload_decisions: 上传决策列表，每个元素为 (vehicle_id, upload_batches)
            bandwidth_allocations: 带宽分配字典 {vehicle_id: bandwidth_ratio}

        返回:
            float: 数据传输时延 (秒)
        """
        max_delay = 0.0
        for vid, m_batches in upload_decisions:
            if m_batches <= 0:
                continue
            v = self._get_vehicle(vid)
            if not v or v.bs_connection is None:
                continue
            bs = self._get_bs(v.bs_connection)
            if not bs:
                continue
            beta = bandwidth_allocations.get(vid, 0.0)
            if beta <= 0:
                continue
            rate = self.calculate_uplink_rate(v, bs, beta)
            if rate <= 0:
                continue
            data_bits = m_batches * self.batch_size * self.sample_bits
            delay = data_bits / rate
            max_delay = max(max_delay, delay)
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
        total_samples = self._total_samples_from_decisions(upload_decisions)
        comp = total_samples * self.c_golden
        return comp / self.C_edge

    def calculate_retraining_delay(self, total_samples, actual_epochs=None):
        if total_samples <= 0:
            return 0.0
        epochs = actual_epochs if actual_epochs is not None else self.training_epochs
        comp = epochs * total_samples * self.c_global
        return comp / self.C_edge

    def calculate_broadcast_delay(self):
        """
        计算模型广播时延 t_broadcast
        公式： t_broadcast = P_m / R_{downlink}^s
        返回:
            float: 模型广播时延 (秒)
        """
        rate = self.calculate_downlink_rate()
        return float('inf') if rate <= 0 else self.model_param_bits / rate

    def calculate_total_training_delay(
            self,
            upload_decisions,
            bandwidth_allocations,
            total_samples,
            include_broadcast=True,
            actual_epochs=None
        ):
            """训练阶段总时延（可选是否包含广播）"""
            t_trans = self.calculate_transmission_delay(upload_decisions, bandwidth_allocations)
            t_label = self.calculate_labeling_delay(upload_decisions)
            t_retrain = self.calculate_retraining_delay(total_samples, actual_epochs)
            t_broadcast = self.calculate_broadcast_delay() if include_broadcast else 0.0

            total = t_trans + t_label + t_retrain + t_broadcast

            return {
                "transmission_delay": t_trans,
                "labeling_delay": t_label,
                "retraining_delay": t_retrain,
                "broadcast_delay": t_broadcast,
                "total_delay": total,
            }
