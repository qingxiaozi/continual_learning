import pytest
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environment.vehicle_env import Vehicle
from config.parameters import config

class TestVehicleEnvironment:
    def test_vehicle_initialization(self):
        """测试车辆初始化"""
        vehicle = Vehicle(0, np.array([100, 0]))
        assert vehicle.id == 0
        assert np.array_equal(vehicle.position, np.array([100, 0]))
        assert len(vehicle.data_batches) == 0
        assert vehicle.bs_connection is None

if __name__ == "__main__":
    pytest.main([__file__, "-v"])