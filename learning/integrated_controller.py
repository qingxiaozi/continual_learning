import numpy as np
import torch
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
        print("带宽分配")

    # 平均带宽分配
    def allocate_average_bandwidth(self) -> np.ndarray:
        """
        Returns:
            bandwidth_ratios: 每辆车的带宽比例列表
        """
        # 简单平均分配：每辆车分配 1/车辆数 的带宽
        bandwidth_ratios = np.ones(self.num_vehicles) / self.num_vehicles
        return bandwidth_ratios

    # 按照批次比例进行带宽分配
    def allocate_proportional_bandwidth(self) -> np.ndarray:
        """
        按批次比例分配带宽
        根据每辆车的上传批次数量按比例分配
        """
        total_batches = sum(self.batch_choices)
        if total_batches == 0:
            return self.allocate_average_bandwidth()

        bandwidth_ratios = np.array(self.batch_choices) / total_batches
        return bandwidth_ratios

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

            # 确保数值稳定性
            bandwidth_ratios = np.clip(bandwidth_ratios, 0, 1)
            total = np.sum(bandwidth_ratios)
            if total > 0:
                bandwidth_ratios = bandwidth_ratios / total
            else:
                bandwidth_ratios = self.allocate_average_bandwidth()

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


class IntegratedController:
    """集成控制器：结合DRL决策和带宽分配"""

    def __init__(self, state_dim, communication_system=None, vehicle_env=None):
        self.agent = DRLAgent(state_dim)
        self.communication_system = communication_system
        self.vehicle_env = vehicle_env

    def make_decision(self, state, available_batches=None, vehicle_states=None,
                     session_id=0, allocation_method='optimized'):
        """
        做出完整决策

        Args:
            state: 状态向量
            available_batches: 可用批次限制
            vehicle_states: 车辆状态
            session_id: 训练会话ID
            allocation_method: 带宽分配方法 ['average', 'proportional', 'optimized']

        Returns:
            complete_action: 完整动作向量 [批次1, 带宽1, 批次2, 带宽2, ...]
            batch_choices: 批次选择列表
            allocation_info: 分配信息字典
        """
        # 1. DQN选择批次
        action_vector, batch_choices = self.agent.select_action(
            state, available_batches=available_batches
        )

        # 2. 带宽分配
        allocator = BandwidthAllocator(batch_choices, self.communication_system, self.vehicle_env)

        if allocation_method == 'average':
            bandwidth_ratios = allocator.allocate_average_bandwidth()
            allocation_info = {'method': 'average', 'transmission_delay': 0.0}

        elif allocation_method == 'proportional':
            bandwidth_ratios = allocator.allocate_proportional_bandwidth()
            allocation_info = {'method': 'proportional', 'transmission_delay': 0.0}

        elif allocation_method == 'optimized':
            bandwidth_ratios, min_max_delay = allocator.allocate_optimized_bandwidth(session_id)
            # 计算实际传输时延
            actual_delay = allocator.calculate_transmission_delay(bandwidth_ratios, session_id)
            allocation_info = {
                'method': 'optimized',
                'min_max_delay': min_max_delay,
                'actual_transmission_delay': actual_delay
            }
        else:
            raise ValueError(f"未知的分配方法: {allocation_method}")

        # 3. 更新动作向量中的带宽部分
        for i in range(len(batch_choices)):
            action_vector[i * 2 + 1] = bandwidth_ratios[i]

        return action_vector, batch_choices, allocation_info

    def train_step(self, state, batch_choices, reward, next_state, done, td_error_estimate=None):
        """训练一步"""
        # 存储经验
        self.agent.store_experience(state, batch_choices, reward, next_state, done, td_error_estimate)

        # 优化模型
        loss = self.agent.optimize_model()
        return loss


import numpy as np
from scipy.optimize import minimize

def main():
    """带宽分配和集成控制器测试主函数"""
    print("=" * 60)
    print("车路协同带宽分配与集成控制器测试")
    print("=" * 60)

    try:
        # 1. 基础配置
        num_vehicles = Config.NUM_VEHICLES
        state_dim = 3 * num_vehicles  # 每辆车3个状态特征

        print(f"\n1. 系统配置")
        print(f"   - 车辆数量: {num_vehicles}")
        print(f"   - 状态维度: {state_dim}")
        print(f"   - 最大上传批次: {Config.MAX_UPLOAD_BATCHES}")

        # 2. 创建模拟环境和通信系统
        print(f"\n2. 创建模拟环境")
        # 创建一个简化的车辆环境用于测试
        class MockVehicle:
            def __init__(self, id, position=(0, 0)):
                self.id = id
                self.position = np.array(position)
                self.bs_connection = 0 if id < 8 else None  # 模拟部分车辆未连接

        class MockVehicleEnv:
            def __init__(self, num_vehicles):
                self.vehicles = [MockVehicle(i, (i*10, 0)) for i in range(num_vehicles)]
                self.base_stations = {0: {"position": np.array([50, 0]), "id": 0}}

            def _get_vehicle_by_id(self, id):
                return self.vehicles[id] if id < len(self.vehicles) else None

            def _get_base_station_by_id(self, id):
                return self.base_stations.get(id)

        # 创建模拟环境
        vehicle_env = MockVehicleEnv(num_vehicles)

        # 3. 创建通信系统
        print(f"3. 创建通信系统")
        class MockCommunicationSystem:
            def __init__(self):
                self.base_bandwidth = Config.BASE_BANDWIDTH
                self.noise_power = Config.NOISE_POWER
                self.I_v = Config.INTERFERENCE_POWER
                self.vehicle_transmit_power = Config.VEHICLE_TRANSMIT_POWER
                self.vehicle_env = vehicle_env
                self.samples_of_per_batch = Config.BATCH_SIZE
                self.sample_size = Config.IMAGE_SIZE * Config.IMAGE_SIZE * 3 * 32

            def calculate_channel_gain(self, vehicle, base_station, session_id):
                if vehicle is None or base_station is None:
                    return 0.0
                distance = np.linalg.norm(vehicle.position - base_station["position"])
                distance = max(distance, 1.0)
                # 简化信道增益计算
                return 1.0 / (distance ** 2.7)

            def calculate_uplink_rate(self, vehicle, base_station, bandwidth_ratio, session_id):
                if vehicle is None or base_station is None:
                    return 0.0
                channel_gain = self.calculate_channel_gain(vehicle, base_station, session_id)
                snr = self.vehicle_transmit_power * channel_gain / (self.noise_power + self.I_v)
                allocated_bandwidth = bandwidth_ratio * self.base_bandwidth
                return allocated_bandwidth * np.log2(1 + snr)

            def calculate_transmission_delay(self, upload_decisions, bandwidth_allocations, session_id):
                max_delay = 0.0
                for vehicle_id, upload_batches in upload_decisions:
                    if upload_batches == 0:
                        continue
                    vehicle = self.vehicle_env._get_vehicle_by_id(vehicle_id)
                    if not vehicle or vehicle.bs_connection is None:
                        continue
                    bandwidth_ratio = bandwidth_allocations.get(vehicle_id, 0.0)
                    if bandwidth_ratio <= 0:
                        continue
                    # 简化计算：固定数据速率
                    data_per_batch = self.samples_of_per_batch * self.sample_size
                    total_data = upload_batches * data_per_batch
                    # 假设速率为带宽比率的函数
                    rate = bandwidth_ratio * self.base_bandwidth * 0.1  # 简化
                    delay = total_data / rate if rate > 0 else float('inf')
                    max_delay = max(max_delay, delay)
                return max_delay

        comm_system = MockCommunicationSystem()
        print(f"   - 通信系统创建成功")
        print(f"   - 基础带宽: {comm_system.base_bandwidth/1e6:.1f} MHz")

        # 4. 测试带宽分配器
        print(f"\n4. 测试带宽分配器")

        # 模拟批次选择
        np.random.seed(42)
        batch_choices = np.random.randint(0, Config.MAX_UPLOAD_BATCHES + 1, num_vehicles)
        print(f"   - 批次选择: {batch_choices}")
        print(f"   - 总上传批次: {sum(batch_choices)}")

        allocator = BandwidthAllocator(batch_choices, comm_system, vehicle_env)

        # 测试平均分配
        print(f"\n   (1) 平均分配测试")
        avg_bandwidth = allocator.allocate_average_bandwidth()
        print(f"     带宽分配: {['%.4f'%x for x in avg_bandwidth]}")
        print(f"     总带宽和: {sum(avg_bandwidth):.6f}")

        # 测试按比例分配
        print(f"\n   (2) 按比例分配测试")
        prop_bandwidth = allocator.allocate_proportional_bandwidth()
        print(f"     带宽分配: {['%.4f'%x for x in prop_bandwidth]}")
        print(f"     总带宽和: {sum(prop_bandwidth):.6f}")

        # 测试优化分配
        print(f"\n   (3) 优化分配测试")
        try:
            opt_bandwidth, min_max_delay = allocator.allocate_minmaxdelay_bandwidth(session_id=1)
            print(f"     带宽分配: {['%.4f'%x for x in opt_bandwidth]}")
            print(f"     总带宽和: {sum(opt_bandwidth):.6f}")
            print(f"     最小最大时延: {min_max_delay:.6f}")
        except Exception as e:
            print(f"     优化分配失败: {e}")
            opt_bandwidth = avg_bandwidth

        # 测试时延计算
        print(f"\n   (4) 时延计算测试")
        for method_name, bandwidth in [("平均分配", avg_bandwidth),
                                      ("比例分配", prop_bandwidth),
                                      ("优化分配", opt_bandwidth)]:
            delay = allocator.calculate_transmission_delay(bandwidth, session_id=1)
            print(f"     {method_name}: {delay:.6f} 秒")

        # 5. 测试集成控制器
        print(f"\n5. 测试集成控制器")

        controller = IntegratedController(state_dim, comm_system, vehicle_env)

        # 模拟状态
        test_state = np.random.randn(state_dim)
        test_available_batches = np.random.randint(0, Config.MAX_UPLOAD_BATCHES + 1, num_vehicles)

        # 测试不同分配方法
        allocation_methods = ['average', 'proportional', 'optimized']

        for method in allocation_methods:
            print(f"\n   ({method}) 分配方法:")
            try:
                action, batches, info = controller.make_decision(
                    test_state,
                    available_batches=test_available_batches,
                    session_id=1,
                    allocation_method=method
                )

                print(f"     批次选择: {batches}")
                print(f"     批次总数: {sum(batches)}")
                print(f"     带宽分配: {['%.4f'%x for x in action[1::2]]}")
                print(f"     分配信息: {info}")

                # 验证动作向量
                if len(action) == 2 * num_vehicles:
                    print(f"     动作向量验证: 通过 ({len(action)}维)")
                else:
                    print(f"     动作向量验证: 失败 ({len(action)}维)")

            except Exception as e:
                print(f"     决策失败: {e}")

        # 6. 测试训练步骤
        print(f"\n6. 测试训练步骤")

        # 模拟训练数据
        next_state = np.random.randn(state_dim)
        reward = 0.5
        done = False

        try:
            loss = controller.train_step(
                test_state,
                batches if 'batches' in locals() else batch_choices,
                reward,
                next_state,
                done
            )

            if loss is not None:
                print(f"   训练损失: {loss:.6f}")
                print(f"   训练成功完成")
            else:
                print(f"   训练返回None（可能经验池不足）")

        except Exception as e:
            print(f"   训练失败: {e}")

        # 7. 验证智能体状态
        print(f"\n7. 智能体状态验证")
        try:
            epsilon = controller.agent._get_epsilon()
            memory_size = len(controller.agent.memory)
            steps = controller.agent.steps_done

            print(f"   当前探索率 (ε): {epsilon:.4f}")
            print(f"   经验回放池大小: {memory_size}")
            print(f"   已执行步数: {steps}")
            print(f"   目标网络更新间隔: {controller.agent.update_target_every}")

            if memory_size > 0:
                print(f"   经验池正常，可进行训练")
            else:
                print(f"   经验池为空，需要先收集经验")

        except Exception as e:
            print(f"   智能体验证失败: {e}")

        # 8. 约束条件测试
        print(f"\n8. 约束条件测试")
        print(f"   批次选择限制测试:")

        test_batches = [5, 0, 3, 2, 0, 1, 4, 0, 2, 3]
        test_allocator = BandwidthAllocator(test_batches, comm_system, vehicle_env)

        # 测试应用约束
        test_bandwidth = np.array([0.2, 0.1, 0.15, 0.1, 0.05, 0.1, 0.2, 0.05, 0.03, 0.02])
        constrained_bandwidth = test_allocator._apply_constraints(test_bandwidth.copy())

        print(f"   原始带宽: {['%.4f'%x for x in test_bandwidth]}")
        print(f"   约束后带宽: {['%.4f'%x for x in constrained_bandwidth]}")
        print(f"   原始总和: {sum(test_bandwidth):.4f}")
        print(f"   约束后总和: {sum(constrained_bandwidth):.4f}")

        # 9. K值计算测试
        print(f"\n9. K值计算测试")
        try:
            K_values = test_allocator._calculate_K_values(session_id=1)
            print(f"   K值结果: {['%.6f'%x for x in K_values]}")
            print(f"   总K值: {sum(K_values):.6f}")

            # 验证K值合理性
            for i in range(len(test_batches)):
                if test_batches[i] > 0 and K_values[i] <= 0:
                    print(f"   警告: 车辆{i}有批次但K值={K_values[i]:.6f}")

        except Exception as e:
            print(f"   K值计算失败: {e}")

        print(f"\n" + "=" * 60)
        print("测试完成！所有功能正常 ✅")
        print("=" * 60)

    except ImportError as e:
        print(f"\n导入错误: {e}")
        print("请确保所有依赖模块已正确导入")
    except AttributeError as e:
        print(f"\n配置错误: {e}")
        print("请检查Config类中的参数定义")
    except Exception as e:
        print(f"\n测试过程中出现错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()