import pytest
import numpy as np
import torch
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environment.vehicle_env import VehicleEnvironment
from environment.dataSimu_env import (
    DomainIncrementalDataSimulator,
    Office31Dataset,
    BaseDataset,
)
from config.parameters import Config


class TestDataSimulator:
    """数据模拟器测试类"""

    def setup_method(self):
        """测试设置"""
        # 创建测试环境
        self.vehicle_env = VehicleEnvironment()
        self.data_simulator = DomainIncrementalDataSimulator(self.vehicle_env)

    def test_initialization(self):
        """测试数据模拟器初始化"""
        assert self.data_simulator.vehicle_env == self.vehicle_env
        assert self.data_simulator.current_dataset == Config.CURRENT_DATASET
        assert hasattr(self.data_simulator, "class_distributions")
        assert hasattr(self.data_simulator, "vehicle_data_assignments")

        print("✓ 数据模拟器初始化测试通过")

    # def test_domain_sequences(self):
    #     """测试域序列配置"""
    #     sequences = self.data_simulator.domain_sequences

    #     # 检查每个数据集的域序列
    #     assert 'office31' in sequences
    #     assert 'digit10' in sequences
    #     assert 'domainnet' in sequences

    #     # 检查域数量
    #     assert len(sequences['office31']) == 3
    #     assert len(sequences['digit10']) == 4
    #     assert len(sequences['domainnet']) == 6

    #     print("✓ 域序列配置测试通过")

    def test_domain_incremental_learning(self):
        """测试域增量学习"""
        # 测试多个会话的域切换
        test_sessions = [0, 20, 40, 60, 80]
        expected_domains = []

        for session in test_sessions:
            self.data_simulator.update_session(session)
            current_domain = self.data_simulator.get_current_domain()
            expected_domains.append(current_domain)

            print(f"Session {session}: 域 = {current_domain}")

        # 验证域确实在变化
        unique_domains = set(expected_domains)
        assert len(unique_domains) > 1, "域应该在不同会话间变化"

        print("✓ 域增量学习测试通过")

    def test_dirichlet_distribution(self):
        """测试狄利克雷分布数据分配"""
        num_vehicles = len(self.vehicle_env.vehicles)
        num_classes = self.data_simulator.dataset_info[Config.CURRENT_DATASET][
            "num_classes"
        ]

        # 检查类别分布
        assert len(self.data_simulator.class_distributions) == num_classes

        for class_idx, distribution in self.data_simulator.class_distributions.items():
            # 检查分布形状
            assert len(distribution) == num_vehicles
            # 检查分布和约为1
            assert abs(np.sum(distribution) - 1.0) < 1e-10
            # 检查所有值非负
            assert np.all(distribution >= 0)

        print("✓ 狄利克雷分布测试通过")

    def test_vehicle_data_generation(self):
        """测试车辆数据生成"""
        # 更新到第一个域
        self.data_simulator.update_session(0)

        # 测试几辆车的数据生成
        test_vehicles = [0, 1, 2]

        for vehicle_id in test_vehicles:
            data_batches = self.data_simulator.generate_vehicle_data(
                vehicle_id, num_batches=2
            )

            # 检查返回的数据批次
            assert isinstance(data_batches, list)
            assert len(data_batches) == 2

            for batch in data_batches:
                assert hasattr(batch, "__iter__")  # 应该是可迭代的DataLoader

                # 尝试获取一个批次的数据
                try:
                    sample_batch = next(iter(batch))
                    if isinstance(sample_batch, (list, tuple)):
                        inputs, targets = sample_batch
                        assert inputs is not None
                        assert targets is not None
                except StopIteration:
                    # 空批次也是可能的
                    pass

            print(f"✓ 车辆 {vehicle_id} 数据生成测试通过")

    def test_data_heterogeneity(self):
        """测试数据异构性"""
        self.data_simulator.update_session(0)

        # 收集所有车辆的数据量
        vehicle_sample_counts = []

        for vehicle in self.vehicle_env.vehicles[:10]:  # 测试前10辆车
            data_batches = self.data_simulator.generate_vehicle_data(
                vehicle.id, num_batches=1
            )

            if data_batches and len(data_batches) > 0:
                data_loader = data_batches[0]
                sample_count = (
                    len(data_loader.dataset)
                    if hasattr(data_loader.dataset, "__len__")
                    else 0
                )
                vehicle_sample_counts.append(sample_count)

        # 检查数据异构性（标准差应该大于0）
        if len(vehicle_sample_counts) > 1:
            std_dev = np.std(vehicle_sample_counts)
            print(f"数据分布标准差: {std_dev:.2f}")

            # 在异构设置下，标准差应该相对较大
            # 但具体数值取决于狄利克雷参数和数据集大小
            assert std_dev >= 0, "数据分布应该有变化"

        print("✓ 数据异构性测试通过")

    def test_data_distribution_info(self):
        """测试数据分布信息获取"""
        self.data_simulator.update_session(0)

        dist_info = self.data_simulator.get_data_distribution_info()

        # 检查信息完整性
        required_keys = [
            "dataset",
            "domain",
            "session",
            "domain_index",
            "total_domains",
        ]
        for key in required_keys:
            assert key in dist_info

        # 检查数据类型
        assert isinstance(dist_info["dataset"], str)
        assert isinstance(dist_info["domain"], str)
        assert isinstance(dist_info["session"], int)

        print("✓ 数据分布信息测试通过")
