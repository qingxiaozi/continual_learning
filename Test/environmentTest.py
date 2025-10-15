import pytest
import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environment.vehicle_env import Vehicle, VehicleEnvironment
from environment.communication_env import CommunicationSystem
from config.parameters import config


class TestVehicleEnvironment:
    def test_vehicle_initialization(self):
        """
        测试车辆初始化
        """
        vehicle = Vehicle(0, np.array([100, 0]))
        assert vehicle.id == 0
        assert np.array_equal(vehicle.position, np.array([100, 0]))
        assert len(vehicle.data_batches) == 0
        assert vehicle.bs_connection is None

    def test_environment_initialization(self):
        """
        测试环境初始化
        """
        env = VehicleEnvironment()
        assert len(env.vehicles) == config.NUM_VEHICLES
        assert len(env.base_stations) <= 5

        # 检查每辆车都连接到了基站
        for vehicle in env.vehicles:
            assert vehicle.bs_connection is not None

    def test_vehicle_movement(self):
        """
        测试车辆移动
        """
        env = VehicleEnvironment()
        initial_positions = [v.position.copy() for v in env.vehicles]
        env.update_vehicle_positions()
        for i, vehicle in enumerate(env.vehicles):
            # 检查位置是否更新
            assert not np.array_equal(vehicle.position, initial_positions[i])
            # 检查基站连接是否更新
            assert vehicle.bs_connection is not None


class TestCommunicationSystem:
    def test_communication_system_initialization(self):
        """测试通信系统初始化"""
        print("开始测试通信系统初始化...")

        try:
            # 创建车辆环境
            env = VehicleEnvironment()
            print("✓ VehicleEnvironment 创建成功")
            assert env is not None, "环境创建失败"

            # 创建通信系统
            comm_system = CommunicationSystem(env)
            print("✓ CommunicationSystem 创建成功")
            assert comm_system is not None, "通信系统创建失败"

            # 检查基本参数
            assert comm_system.base_bandwidth > 0, "基础带宽应该大于0"
            assert comm_system.noise_power > 0, "噪声功率应该大于0"
            assert comm_system.edge_server_computation > 0, "边缘服务器计算能力应该大于0"

            print("✓ 通信系统参数初始化正确")

        except Exception as e:
            print(f"❌ 通信系统初始化测试失败: {e}")
            import traceback
            traceback.print_exc()
            raise

    def test_channel_gain_calculation(self):
        """测试信道增益计算"""
        print("开始测试信道增益计算...")

        try:
            env = VehicleEnvironment()
            comm_system = CommunicationSystem(env)

            vehicle = env.vehicles[0]
            base_station = env.base_stations[0]

            # 测试信道增益计算
            channel_gain = comm_system.calculate_channel_gain(
                vehicle, base_station, session_id=1
            )
            print(f"✓ 信道增益计算: {channel_gain:.6f}")

            # 验证信道增益合理性
            assert channel_gain > 0, f"信道增益应该大于0，实际得到: {channel_gain}"
            assert channel_gain < 1, f"信道增益通常小于1，实际得到: {channel_gain}"

            # 测试多次计算的确定性（相同的输入应该得到相同的结果）
            channel_gain2 = comm_system.calculate_channel_gain(
                vehicle, base_station, session_id=1
            )
            assert channel_gain == channel_gain2, "相同输入的信道增益计算应该一致"

            # 测试不同会话ID得到不同的阴影衰落
            channel_gain3 = comm_system.calculate_channel_gain(
                vehicle, base_station, session_id=2
            )
            # 由于阴影衰落随机，可能相同也可能不同，但不应该报错

            print("✓ 信道增益计算测试通过")

        except Exception as e:
            print(f"❌ 信道增益计算测试失败: {e}")
            import traceback
            traceback.print_exc()
            raise

    def test_uplink_rate_calculation(self):
        """测试上行链路速率计算"""
        print("开始测试上行链路速率计算...")

        try:
            env = VehicleEnvironment()
            comm_system = CommunicationSystem(env)

            vehicle = env.vehicles[0]
            base_station = env.base_stations[0]

            # 测试不同带宽分配的上行速率
            bandwidth_ratios = [0.1, 0.5, 1.0]
            rates = []

            for ratio in bandwidth_ratios:
                uplink_rate = comm_system.calculate_uplink_rate(
                    vehicle, base_station, bandwidth_ratio=ratio, session_id=1
                )
                rates.append(uplink_rate)
                print(f"✓ 带宽分配 {ratio:.1f} 的上行速率: {uplink_rate:.2f} bit/s")

                # 验证速率非负
                assert uplink_rate >= 0, f"上行速率应该非负，实际得到: {uplink_rate}"

            # 验证带宽分配越大速率越高（在相同信道条件下）
            for i in range(1, len(rates)):
                assert rates[i] >= rates[i-1], "带宽分配增加时上行速率应该增加或保持不变"

            # 测试零带宽分配
            zero_rate = comm_system.calculate_uplink_rate(
                vehicle, base_station, bandwidth_ratio=0.0, session_id=1
            )
            assert zero_rate == 0, "零带宽分配时上行速率应该为0"

            print("✓ 上行链路速率计算测试通过")

        except Exception as e:
            print(f"❌ 上行链路速率计算测试失败: {e}")
            import traceback
            traceback.print_exc()
            raise

    def test_downlink_rate_calculation(self):
        """测试下行链路速率计算"""
        print("开始测试下行链路速率计算...")

        try:
            env = VehicleEnvironment()
            comm_system = CommunicationSystem(env)

            # 测试下行速率计算
            downlink_rate = comm_system.calculate_downlink_rate(session_id=1)
            print(f"✓ 下行速率计算: {downlink_rate:.2f} bit/s")

            # 验证下行速率合理性
            assert downlink_rate >= 0, f"下行速率应该非负，实际得到: {downlink_rate}"
            assert downlink_rate > 0, f"下行速率应该大于0，实际得到: {downlink_rate}"

            # 测试不同会话的下行速率
            downlink_rate2 = comm_system.calculate_downlink_rate(session_id=2)
            # 由于信道条件可能变化，速率可能不同

            print("✓ 下行链路速率计算测试通过")

        except Exception as e:
            print(f"❌ 下行链路速率计算测试失败: {e}")
            import traceback
            traceback.print_exc()
            raise

    def test_transmission_delay_calculation(self):
        """测试数据传输时延计算"""
        print("开始测试数据传输时延计算...")

        try:
            env = VehicleEnvironment()
            comm_system = CommunicationSystem(env)

            # 创建上传决策和带宽分配
            upload_decisions = []
            bandwidth_allocations = {}

            # 为前3辆车创建上传决策
            for i in range(min(3, len(env.vehicles))):
                upload_decisions.append((env.vehicles[i].id, 2))  # 每辆车上传2个批次
                bandwidth_allocations[env.vehicles[i].id] = 0.1  # 每辆车分配10%带宽

            # 计算传输时延
            transmission_delay = comm_system.calculate_transmission_delay(
                upload_decisions, bandwidth_allocations, session_id=1
            )
            print(f"✓ 数据传输时延: {transmission_delay:.4f} 秒")

            # 验证时延非负
            assert transmission_delay >= 0, f"传输时延应该非负，实际得到: {transmission_delay}"

            # 测试无上传数据的情况
            zero_upload_decisions = [(env.vehicles[0].id, 0)]
            zero_delay = comm_system.calculate_transmission_delay(
                zero_upload_decisions, bandwidth_allocations, session_id=1
            )
            assert zero_delay == 0, "无上传数据时传输时延应该为0"

            print("✓ 数据传输时延计算测试通过")

        except Exception as e:
            print(f"❌ 数据传输时延计算测试失败: {e}")
            import traceback
            traceback.print_exc()
            raise

    def test_labeling_delay_calculation(self):
        """测试数据标注时延计算"""
        print("开始测试数据标注时延计算...")

        try:
            env = VehicleEnvironment()
            comm_system = CommunicationSystem(env)

            # 创建上传决策
            upload_decisions = []
            total_samples = 0

            # 为前3辆车创建上传决策
            for i in range(min(3, len(env.vehicles))):
                batches = 2
                upload_decisions.append((env.vehicles[i].id, batches))
                total_samples += batches * comm_system.samples_of_per_batch

            # 计算标注时延
            labeling_delay = comm_system.calculate_labeling_delay(upload_decisions)
            print(f"✓ 数据标注时延: {labeling_delay:.6f} 秒")

            # 验证时延计算正确性
            expected_delay = (total_samples * comm_system.golden_model_computation) / comm_system.edge_server_computation
            assert abs(labeling_delay - expected_delay) < 1e-10, "标注时延计算不正确"

            # 验证时延非负
            assert labeling_delay >= 0, f"标注时延应该非负，实际得到: {labeling_delay}"

            # 测试无上传数据的情况
            zero_upload_decisions = []
            zero_delay = comm_system.calculate_labeling_delay(zero_upload_decisions)
            assert zero_delay == 0, "无上传数据时标注时延应该为0"

            print("✓ 数据标注时延计算测试通过")

        except Exception as e:
            print(f"❌ 数据标注时延计算测试失败: {e}")
            import traceback
            traceback.print_exc()
            raise

    def test_retraining_delay_calculation(self):
        """测试模型重训练时延计算"""
        print("开始测试模型重训练时延计算...")

        try:
            env = VehicleEnvironment()
            comm_system = CommunicationSystem(env)

            num_vehicles = len(env.vehicles)

            # 计算重训练时延
            retraining_delay = comm_system.calculate_retraining_delay(num_vehicles)
            print(f"✓ 模型重训练时延: {retraining_delay:.4f} 秒")

            # 验证时延计算正确性
            total_samples = comm_system.cache_samples_per_vehicle * num_vehicles
            computation_per_epoch = total_samples * comm_system.global_model_computation
            total_computation = comm_system.training_epochs * computation_per_epoch
            expected_delay = total_computation / comm_system.edge_server_computation

            assert abs(retraining_delay - expected_delay) < 1e-10, "重训练时延计算不正确"

            # 验证时延非负
            assert retraining_delay >= 0, f"重训练时延应该非负，实际得到: {retraining_delay}"

            print("✓ 模型重训练时延计算测试通过")

        except Exception as e:
            print(f"❌ 模型重训练时延计算测试失败: {e}")
            import traceback
            traceback.print_exc()
            raise



if __name__ == "__main__":
    pytest.main([__file__, "-v"])
