import pytest
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environment.vehicle_env import Vehicle, VehicleEnvironment
from config.parameters import config

class TestVehicleEnvironment:
    def test_vehicle_initialization(self):
        '''
        测试车辆初始化
        '''
        vehicle = Vehicle(0, np.array([100, 0]))
        assert vehicle.id == 0
        assert np.array_equal(vehicle.position, np.array([100, 0]))
        assert len(vehicle.data_batches) == 0
        assert vehicle.bs_connection is None

    def test_environment_initialization(self):
        '''
        测试环境初始化
        '''
        env = VehicleEnvironment()
        assert len(env.vehicles) == config.NUM_VEHICLES
        assert len(env.base_stations) <= 5

        # 检查每辆车都连接到了基站
        for vehicle in env.vehicles:
            assert vehicle.bs_connection is not None

    def test_vehicle_movement(self):
        '''
        测试车辆移动
        '''
        env = VehicleEnvironment()
        initial_positions = [v.position.copy() for v in env.vehicles]
        env.update_vehicle_positions()
        for i, vehicle in enumerate(env.vehicles):
            # 检查位置是否更新
            assert not np.array_equal(vehicle.position, initial_positions[i])
            # 检查基站连接是否更新
            assert vehicle.bs_connection is not None

if __name__ == "__main__":
    pytest.main([__file__, "-v"])