import numpy as np
import torch
from scipy.optimize import minimize
from typing import List, Optional, Dict, Tuple
from environment.communication_env import CommunicationSystem
from config.parameters import Config
from models.drl_agent import DRLAgent
from environment.communication_env import CommunicationSystem
from environment.vehicle_env import VehicleEnvironment


class BandwidthAllocator:
    """带宽分配优化器"""

    def __init__(self, batch_choices, communication_system=None, vehicle_env=None):
        self.batch_choices = batch_choices
        self.num_vehicles = len(batch_choices)
        self.comm_system = communication_system
        self.vehicle_env = vehicle_env
        # print("带宽分配")

    # 平均带宽分配
    def allocate_average_bandwidth(self) -> np.ndarray:
        """
        Returns:
            bandwidth_ratios: 每辆车的带宽比例列表
        """
        # 简单平均分配：每辆车分配 1/车辆数 的带宽
        return np.ones(self.num_vehicles) / self.num_vehicles

    # 按照批次比例进行带宽分配
    def allocate_proportional_bandwidth(self) -> np.ndarray:
        """
        按批次比例分配带宽
        根据每辆车的上传批次数量按比例分配
        """
        total_batches = sum(self.batch_choices)
        if total_batches == 0:
            return self.allocate_average_bandwidth()

        return np.array(self.batch_choices) / total_batches

    # 按照最小化最大传输时延分配带宽
    def allocate_minmaxdelay_bandwidth(self, session_id: int = 0) -> Tuple[np.ndarray, float]:
        """
        优化带宽分配（基于最小化最大传输时延）
        根据理论公式：min max (m_v * |b_v| * b_0) / (β_v * R_uplink)

        Args:
            session_id: 训练会话ID

        Returns:
            bandwidth_ratios: 每辆车的带宽比例列表
            min_max_delay: 最小化的最大传输时延
        """
        # 如果没有数据传输需求，返回平均分配
        if sum(self.batch_choices) == 0:
            return self.allocate_average_bandwidth(), 0.0

        # 计算每辆车的K_v值（基本传输需求）
        K_values = self._calculate_K_values(session_id)

        # 优化带宽分配
        bandwidth_ratios, min_max_delay = self._solve_optimization_problem(K_values)

        return bandwidth_ratios, min_max_delay

    def _calculate_K_values(self, session_id: int) -> np.ndarray:
        """
        计算每辆车的K_v值
        K_v = (m_v * |b_v| * b_0) / (B_base * log₂(1+SNR_v))
        """
        K_values = np.zeros(self.num_vehicles)

        for v in range(self.num_vehicles):
            if self.batch_choices[v] == 0:
                continue

            # 获取车辆和基站信息
            vehicle = self.vehicle_env._get_vehicle_by_id(v)
            if not vehicle or vehicle.bs_connection is None:
                K_values[v] = 0
                continue

            base_station = self.vehicle_env._get_base_station_by_id(vehicle.bs_connection)
            if not base_station:
                K_values[v] = 0
                continue

            # 使用通信系统计算上行速率系数
            # 这里需要通信系统的相关函数来计算信道增益和SNR
            if hasattr(self.comm_system, 'calculate_channel_gain'):
                channel_gain = self.comm_system.calculate_channel_gain(
                    vehicle, base_station, session_id
                )

                # 计算信噪比
                signal_power = self.comm_system.vehicle_transmit_power * channel_gain
                noise_interference = self.comm_system.noise_power + self.comm_system.I_v
                snr = signal_power / max(noise_interference, 1e-20)

                # 计算对数项，避免log2(1)为0
                log_term = max(np.log2(1 + snr), 0.01)

                # 计算数据量
                data_per_batch = self.comm_system.samples_of_per_batch * self.comm_system.sample_size
                total_data = self.batch_choices[v] * data_per_batch

                # 计算K_v
                K_values[v] = total_data / (self.comm_system.base_bandwidth * log_term)
            else:
                # 如果没有通信系统，使用简单比例
                K_values[v] = self.batch_choices[v]

        return K_values

    def _solve_optimization_problem(self, K_values: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        求解带宽分配优化问题

        原始问题：min max (K_v / β_v)
        s.t. ∑β_v ≤ 1, β_v ≥ 0, β_v ≤ m_v/M_max

        转换为线性规划：
        min t
        s.t. K_v ≤ t * β_v
             ∑β_v ≤ 1
             β_v ≥ 0
             β_v ≤ m_v/M_max
        """
        num_vehicles = len(K_values)

        # 如果没有有效的K值，使用简单分配
        if np.sum(K_values) == 0:
            return self.allocate_average_bandwidth(), 0.0

        try:
            # 方法1：使用闭式解（无约束时）
            if not hasattr(Config, 'MAX_UPLOAD_BATCHES'):
                # 无上界约束的最优解
                total_K = np.sum(K_values)
                bandwidth_ratios = K_values / total_K
                min_max_delay = total_K
            else:
                # 方法2：使用SciPy优化（有约束时）
                bandwidth_ratios, min_max_delay = self._solve_with_scipy(K_values)

            return bandwidth_ratios, min_max_delay

        except Exception as e:
            # 如果优化失败，回退到按K值比例分配
            print(f"优化失败: {e}, 使用按比例分配")
            total_K = np.sum(K_values)
            if total_K > 0:
                bandwidth_ratios = K_values / total_K
                # 应用约束
                bandwidth_ratios = self._apply_constraints(bandwidth_ratios)
                min_max_delay = max(K_values[i]/bandwidth_ratios[i]
                                   for i in range(num_vehicles) if bandwidth_ratios[i] > 0)
            else:
                bandwidth_ratios = self.allocate_average_bandwidth()
                min_max_delay = 0.0

            return bandwidth_ratios, min_max_delay

    def _solve_with_scipy(self, K_values: np.ndarray) -> Tuple[np.ndarray, float]:
        """使用SciPy求解优化问题"""
        num_vehicles = len(K_values)
        M_max = Config.MAX_UPLOAD_BATCHES

        # 目标函数：最小化最大传输时间t
        def objective(x):
            t = x[0]  # 第一个变量是t
            return t

        # 约束：K_v ≤ t * β_v
        constraints = []
        for v in range(num_vehicles):
            if K_values[v] > 0:
                def constraint_func(x, v=v):
                    t = x[0]
                    beta = x[1+v]
                    return t * beta - K_values[v]
                constraints.append({'type': 'ineq', 'fun': constraint_func})

        # 约束：∑β_v ≤ 1
        def sum_constraint(x):
            betas = x[1:1+num_vehicles]
            return 1 - np.sum(betas)
        constraints.append({'type': 'ineq', 'fun': sum_constraint})

        # 约束：β_v ≥ 0
        for v in range(num_vehicles):
            def nonneg_constraint(x, v=v):
                return x[1+v]
            constraints.append({'type': 'ineq', 'fun': nonneg_constraint})

        # 约束：β_v ≤ m_v/M_max
        if M_max > 0:
            for v in range(num_vehicles):
                max_ratio = self.batch_choices[v] / M_max
                def upper_constraint(x, v=v, max_ratio=max_ratio):
                    return max_ratio - x[1+v]
                constraints.append({'type': 'ineq', 'fun': upper_constraint})

        # 初始值
        x0 = np.ones(num_vehicles + 1)  # [t, β_1, β_2, ..., β_n]
        x0[0] = np.sum(K_values)  # 初始t设为总K值
        x0[1:] = K_values / np.sum(K_values)  # 初始带宽按K值比例分配

        # 变量边界
        bounds = [(0, None)] + [(0, 1)] * num_vehicles

        # 求解
        result = minimize(objective, x0, method='SLSQP',
                         constraints=constraints, bounds=bounds,
                         options={'maxiter': 1000, 'ftol': 1e-6})

        if result.success:
            t_opt = result.x[0]
            bandwidth_ratios = result.x[1:1+num_vehicles]

            # # 确保数值稳定性
            # bandwidth_ratios = np.clip(bandwidth_ratios, 0, 1)
            # total = np.sum(bandwidth_ratios)
            # if total > 0:
            #     bandwidth_ratios = bandwidth_ratios / total
            # else:
            #     bandwidth_ratios = self.allocate_average_bandwidth()
            bandwidth_ratios = np.clip(result.x[1:1+num_vehicles], 0, 1)
            # 显式强制满足个体上限（防止SLSQP轻微越界）
            if M_max != float('inf') and M_max > 0:
                for v in range(num_vehicles):
                    max_ratio = self.batch_choices[v] / M_max
                    bandwidth_ratios[v] = min(bandwidth_ratios[v], max_ratio)

            return bandwidth_ratios, t_opt
        else:
            raise ValueError(f"优化失败: {result.message}")

    def _apply_constraints(self, bandwidth_ratios: np.ndarray) -> np.ndarray:
        """应用约束到带宽分配"""
        if hasattr(Config, 'MAX_UPLOAD_BATCHES'):
            M_max = Config.MAX_UPLOAD_BATCHES
            for v in range(self.num_vehicles):
                max_ratio = self.batch_choices[v] / M_max if M_max > 0 else 1.0
                bandwidth_ratios[v] = min(bandwidth_ratios[v], max_ratio)

        # 重新归一化
        total = np.sum(bandwidth_ratios)
        if total > 0:
            bandwidth_ratios = bandwidth_ratios / total
        else:
            bandwidth_ratios = self.allocate_average_bandwidth()

        return bandwidth_ratios

    def calculate_transmission_delay(self, bandwidth_ratios: np.ndarray,
                                   session_id: int = 0) -> float:
        """
        计算传输时延（直接使用CommunicationSystem的函数）
        """
        if self.comm_system is None or self.vehicle_env is None:
            return 0.0

        # 构建上传决策列表
        upload_decisions = []
        for v in range(self.num_vehicles):
            if self.batch_choices[v] > 0:
                upload_decisions.append((v, self.batch_choices[v]))

        # 构建带宽分配字典
        bandwidth_allocations = {}
        for v in range(self.num_vehicles):
            if bandwidth_ratios[v] > 0:
                bandwidth_allocations[v] = bandwidth_ratios[v]

        # 使用通信系统的calculate_transmission_delay函数
        return self.comm_system.calculate_transmission_delay(
            upload_decisions, bandwidth_allocations, session_id
        )
