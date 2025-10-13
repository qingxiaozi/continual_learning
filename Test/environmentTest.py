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
    def test_communication_system(self):
        """测试通信系统功能"""
        print("开始测试通信系统...")

        try:
            # 创建车辆环境
            env = VehicleEnvironment()
            print("✓ VehicleEnvironment 创建成功")
            assert env is not None, "环境创建失败"

            # 创建通信系统
            comm_system = CommunicationSystem(env)
            print("✓ CommunicationSystem 创建成功")
            assert comm_system is not None, "通信系统创建失败"

            # 检查是否有车辆和基站
            assert len(env.vehicles) > 0, "没有创建车辆"
            assert len(env.base_stations) > 0, "没有创建基站"
            print(
                f"✓ 创建了 {len(env.vehicles)} 辆车和 {len(env.base_stations)} 个基站"
            )

            # 测试信道增益计算
            vehicle = env.vehicles[0]
            base_station = env.base_stations[0]
            channel_gain = comm_system.calculate_channel_gain(
                vehicle, base_station, session_id=1
            )
            print(f"✓ 信道增益计算: {channel_gain:.6f}")

            # 验证信道增益合理性
            assert channel_gain > 0, f"信道增益应该大于0，实际得到: {channel_gain}"
            assert channel_gain < 1, f"信道增益通常小于1，实际得到: {channel_gain}"

            # 测试上行速率计算
            uplink_rate = comm_system.calculate_uplink_rate(vehicle, base_station, bandwidth_ratio=0.1, session_id=1)
            print(f"✓ 上行速率计算: {uplink_rate:.2f} bit/s")

            # 测试下行速率计算
            downlink_rate = comm_system.calculate_downlink_rate(session_id=1)
            print(f"✓ 下行速率计算: {downlink_rate:.2f} bit/s")

        except Exception as e:
            print(f"❌ 测试失败: {e}")
            import traceback

            traceback.print_exc()
            raise  # 重新抛出异常，让测试失败


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
