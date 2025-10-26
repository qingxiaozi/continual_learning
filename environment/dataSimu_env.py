import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
from PIL import Image
import scipy.io as sio
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.parameters import config
from environment.vehicle_env import VehicleEnvironment

class DomainIncrementalDataSimulator:
    """
    域增量数据模拟器

    功能：
    1. 支持多个数据集的域增量学习
    2. 使用狄利克雷分布实现车辆数据的异构分配
    3. 动态切换域以模拟真实环境中的分布漂移
    """
    def __init__(self, vehicle_env):
        self.vehicle_env = vehicle_env
        self.current_dataset = config.CURRENT_DATASET
        self.current_domain_idx = 0
        self.current_session = 0  # = self.vehicle_env.current_session

        # 域序列
        self.domain_sequences = config.DOMAIN_SEQUENCES
        self.current_domains = self.domain_sequences[self.current_dataset]

        # 数据集信息
        self.dataset_info = {
            'office31': {
                'num_classes': config.OFFICE31_CLASSES,
                'domains': ['amazon', 'webcam', 'dslr'],
                'data_loader': self._load_office31_data
            },
            'digit10': {
                'num_classes': config.DIGIT10_CLASSES,
                'domains': ['mnist', 'emnist', 'usps', 'svhn'],
                'data_loader': self._load_digit10_data
            },
            'domainnet': {
                'num_classes': config.DOMAINNET_CLASSES,
                'domains': ['clipart', 'infograph', 'painting', 'quickdraw', 'real', 'sketch'],
                'data_loader': self._load_domainnet_data
            }
        }

        # 数据缓存
        self.data_cache = {}  # 缓存不同域的数据，key为域名或索引，值是数据
        self.vehicle_data_assignment = {}  # 每辆车的数据，key为数据集_域名（office31_amazon），值为一个字典，其中键为vehicle_id,值为该车辆在该域中分配到的索引列表

        # 数据变换
        self.transform = transforms.Compose([
            transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # 初始化数据分配
        self._initialize_data_distribution()


    def _initialize_data_distribution(self):
        """
        使用狄利克雷分布初始化车辆数据分配
        """
        num_vehicles = len(self.vehicle_env.vehicles)
        num_classes = self.dataset_info[self.current_dataset['num_classes']]
        # 为每个类别生成狄利克雷分布，存储每个类别在不同车辆上的分配比例
        self.class_distributions = {}

        for class_idx in range(num_classes):
            # 使用狄利克雷分布生成每个类别的车辆分配比例
            alpha = np.full(num_vehicles, config.DIRICHLET_ALPHA)
            distribution = np.random.dirichlet(alpha)
            self.class_distributions[class_idx] = distribution
        print(f"使用狄利克雷分布(α={config.DIRICHLET_ALPHA})初始化{num_vehicles}辆车的数据分配")

    def get_current_domain(self):
        """
        获取当前域
        """
        domain_idx = (self.current_session // config.DOMAIN_CHANGE_INTERVAL) % len(self.current_domains)
        return self.current_domains[domain_idx]

    def update_session(self, session_id):
        """
        更新训练会话
        """
        self.current_session = session_id
        # 检查是否需要切换域
        old_domain_idx = self.current_domain_idx
        self.current_domain_idx = (session_id // config.DOMAIN_CHANGE_INTERVAL) % len(self.current_domains)
        if old_domain_idx != self.current_domain_idx:
            print(f"session {session_id}：域切换 {self.current_domains[old_domain_idx]} -> {self.current_domains[self.current_domain_idx]}")
            # 清除缓存，强制重新加载新域数据
            domain_key = f"{self.current_dataset}_{self.get_current_domain()}"
            if domain_key in self.data_cache:
                del self.data_cache[domain_key]

    def generate_vehicle_data(self, vehicle_id, num_batches = 5):
        """
        为指定车辆生成数据批次
        输入：
            vehicle_id：车辆id
            num_batches：车辆的数据批次数量
        输出：
            batches：指定车辆的数据批次
        """
        current_domain = self.get_current_domain()
        domain_key = f"{self.current_dataset}_{current_domain}"
        # 加载或获取缓存的数据
        if domain_key not in self.data_cache:
            self.data_cache[domain_key] = self._load_domain_data(current_domain)

        full_dataset = self.data_cache[domain_key]
        # 获取该车辆的数据子集
        vehicle_indices = self._get_vehicle_data_indices(vehicle_id, full_dataset)
        if not vehicle_indices:
            print(f"警告：车辆 {vehicle_id} 在当前域 {current_domain} 中没有分配到数据")
            return []
        # 创建车辆特定的数据集
        vehicle_dataset = Subset(full_dataset, vehicle_indices)
        # 创建数据批次
        batches = []
        for _ in range(num_batches):
            dataloader = DataLoader(
                vehicle_dataset,
                batch_size = config.BATCH_SIZE,
                shuffle = True,
                drop_last = True
            )
            batches.append(dataloader)
        return batches

    def _get_vehicle_data_indices(self, vehicle_id, full_dataset):
        """
        获取指定车辆在当前域中的数据索引
        输入：
            vehicle_id：车辆id
            full_dataset：某个域的所有数据，即域数据集
        输出：

        """
        num_vehicles = len(self.vehicle_env.vehicles)
        vehicle_idx = vehicle_id
        # 如果还没有为该域分配数据，则进行分配
        domain_key = f"{self.current_dataset}_{self.get_current_domain()}"
        if domain_key not in self.vehicle_data_assignment:
            self._assign_domain_data_to_vehicles(full_dataset)

        return self.vehicle_data_assignment(full_dataset)

    def _assign_domain_data_to_vehicles(self, full_dataset):
        """
        使用狄利克雷分布将域数据分配给各个车辆
        """


    def _print_allocation_statistics(self, assignments, domain_key):
        """打印数据分配统计信息"""