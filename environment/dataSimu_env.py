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
import matplotlib.pyplot as plt


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
            "office31": {
                "num_classes": config.OFFICE31_CLASSES,
                "domains": ["amazon", "webcam", "dslr"],
                "data_loader": self._load_office31_data,
            },
            # "digit10": {
            #     "num_classes": config.DIGIT10_CLASSES,
            #     "domains": ["mnist", "emnist", "usps", "svhn"],
            #     "data_loader": self._load_digit10_data,
            # },
            # "domainnet": {
            #     "num_classes": config.DOMAINNET_CLASSES,
            #     "domains": [
            #         "clipart",
            #         "infograph",
            #         "painting",
            #         "quickdraw",
            #         "real",
            #         "sketch",
            #     ],
            #      "data_loader": self._load_domainnet_data,
            # },
        }

        # 数据缓存
        self.data_cache = {}  # 缓存不同域的数据，key为域名或索引，值是数据
        self.vehicle_data_assignments = (
            {}
        )  # 每辆车的数据，key为数据集_域名（office31_amazon），值为一个字典，其中键为vehicle_id,值为该车辆在该域中分配到的索引列表

        # 数据变换
        self.transform = transforms.Compose(
            [
                transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

        # 初始化数据分配
        self._initialize_data_distribution()

    def _initialize_data_distribution(self):
        """
        使用狄利克雷分布初始化车辆数据分配
        """
        num_vehicles = len(self.vehicle_env.vehicles)
        num_classes = self.dataset_info[self.current_dataset]["num_classes"]
        # 为每个类别生成狄利克雷分布，存储每个类别在不同车辆上的分配比例
        self.class_distributions = {}

        for class_idx in range(num_classes):
            # 使用狄利克雷分布生成每个类别的车辆分配比例
            alpha = np.full(num_vehicles, config.DIRICHLET_ALPHA)
            distribution = np.random.dirichlet(alpha)
            self.class_distributions[class_idx] = distribution
        print(
            f"使用狄利克雷分布(α={config.DIRICHLET_ALPHA})初始化{num_vehicles}辆车的数据分配"
        )

    def get_current_domain(self):
        """
        获取当前域
        """
        domain_idx = (self.current_session // config.DOMAIN_CHANGE_INTERVAL) % len(
            self.current_domains
        )
        return self.current_domains[domain_idx]

    def update_session(self, session_id):
        """
        更新训练会话
        """
        self.current_session = session_id
        # 检查是否需要切换域
        old_domain_idx = self.current_domain_idx
        self.current_domain_idx = (session_id // config.DOMAIN_CHANGE_INTERVAL) % len(
            self.current_domains
        )
        if old_domain_idx != self.current_domain_idx:
            print(
                f"session {session_id}：域切换 {self.current_domains[old_domain_idx]} -> {self.current_domains[self.current_domain_idx]}"
            )
            # 清除缓存，强制重新加载新域数据
            domain_key = f"{self.current_dataset}_{self.get_current_domain()}"
            if domain_key in self.data_cache:
                del self.data_cache[domain_key]

    def generate_vehicle_data(self, vehicle_id, num_batches=5):
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
                batch_size=config.BATCH_SIZE,
                shuffle=True,
                drop_last=True,
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
        if domain_key not in self.vehicle_data_assignments:
            self._assign_domain_data_to_vehicles(full_dataset)

        return self.vehicle_data_assignments[domain_key].get(vehicle_idx, [])

    def _assign_domain_data_to_vehicles(self, full_dataset):
        """
        使用狄利克雷分布将域数据分配给各个车辆
        """
        num_vehicles = len(self.vehicle_env.vehicles)
        domain_key = f"{self.current_dataset}_{self.get_current_domain()}"
        # 按类别组织数据索引，在加载数据集的时候必须要知道label
        class_indices = {}
        for idx, (_, label) in enumerate(full_dataset):
            if label not in class_indices:
                class_indices[label] = []
            class_indices[label].append(idx)
        # 为每个车辆分配数据索引
        vehicle_assignments = {i: [] for i in range(num_vehicles)}

        for class_idx, indices in class_indices.items():
            if class_idx not in self.class_distributions:
                # 如果类别不在初始分布中，创建新的分布
                alpha = np.full(num_vehicles, config.DIRICHLET_ALPHA)
                self.class_distributions[class_idx] = np.random.dirichlet(alpha)
            distribution = self.class_distributions[class_idx]
            num_samples = len(indices)

            # 根据分布分配样本
            sample_counts = (distribution * num_samples).astype(int)
            # 处理可能的舍入误差
            total_assigned = sum(sample_counts)
            if total_assigned < num_samples:
                # 将剩余样本随机分配给车辆
                remaining = num_samples - total_assigned
                extra_assignments = np.random.choice(num_vehicles, remaining)
                for vehicle_idx in extra_assignments:
                    sample_counts[vehicle_idx] += 1

            # 随机打乱索引并分配
            np.random.shuffle(indices)
            start_idx = 0
            for vehicle_idx, count in enumerate(sample_counts):
                if count > 0:
                    vehicle_assignments[vehicle_idx].extend(
                        indices[start_idx : start_idx + count]
                    )
                    start_idx += count

        self.vehicle_data_assignments[domain_key] = vehicle_assignments

        # 打印分配统计
        self._print_allocation_statistics(vehicle_assignments, domain_key)

    def _print_allocation_statistics(self, assignments, domain_key):
        """打印数据分配统计信息"""
        vehicle_counts = [len(indices) for indices in assignments.values()]
        total_samples = sum(vehicle_counts)

        print(f"\n=== {domain_key} 数据分配统计 ===")
        print(f"总样本数: {total_samples}")
        print(
            f"车辆数据量: 平均{np.mean(vehicle_counts):.1f}, "
            f"最小{min(vehicle_counts)}, 最大{max(vehicle_counts)}"
        )
        print(f"数据异构程度: 标准差{np.std(vehicle_counts):.1f}")

        # 计算每个车辆的数据分布差异
        distribution_differences = []
        for vehicle_idx in range(len(self.vehicle_env.vehicles)):
            if vehicle_idx in assignments and assignments[vehicle_idx]:
                # 这里可以添加更详细的分析
                pass
        print("==============================\n")

    def _load_domain_data(self, domain):
        """加载指定域的数据"""
        loader_func = self.dataset_info[self.current_dataset]["data_loader"]
        return loader_func(domain)

    def _load_office31_data(self, domain):
        """
        加载office-31数据集
        """
        dataset_path = os.path.join(config.DATA_BASE_PATH, "office-31", domain)
        if not os.path.exists(dataset_path):
            print(f"警告：Office-31 {domain}路径不存在：{dataset_path}")
            # return self._create_simulated_dataset(config.OFFICE31_CLASSES, 1000)
        return Office31Dataset(dataset_path, transform=self.transform)

    # def _load_digit10_data(self, domain):
    #     """加载Digit10数据集"""
    #     if domain == 'mnist':
    #         return MNISTDataset(transform=self.transform)
    #     elif domain == 'emnist':
    #         return EMNISTDataset(transform=self.transform)
    #     elif domain == 'usps':
    #         return USPSDataset(transform=self.transform)
    #     elif domain == 'svhn':
    #         return SVHNDataset(transform=self.transform)
    #     else:
    #         print(f"警告: Digit10 域 {domain} 不支持")
    #         return self._create_simulated_dataset(config.DIGIT10_CLASSES, 1000)

    def get_data_distribution_info(self):
        """获取当前数据分布信息"""
        current_domain = self.get_current_domain()
        domain_key = f"{self.current_dataset}_{current_domain}"

        info = {
            "dataset": self.current_dataset,
            "domain": current_domain,
            "session": self.current_session,
            "domain_index": self.current_domain_idx,
            "total_domains": len(self.current_domains),
            "has_data_assignment": domain_key in self.vehicle_data_assignments,
        }

        if domain_key in self.vehicle_data_assignments:
            assignments = self.vehicle_data_assignments[domain_key]
            sample_counts = [len(indices) for indices in assignments.values()]
            info.update(
                {
                    "total_samples": sum(sample_counts),
                    "avg_samples_per_vehicle": np.mean(sample_counts),
                    "std_samples_per_vehicle": np.std(sample_counts),
                    "min_samples": min(sample_counts),
                    "max_samples": max(sample_counts),
                }
            )

        return info


# 基础数据集类
class BaseDataset(Dataset):
    def __init__(self, transform=None):
        self.transform = transform
        self.data = []
        self.labels = []

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = self.data[idx]
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label


class Office31Dataset(BaseDataset):
    def __init__(self, data_path, transform=None):
        super().__init__(transform)
        self.data_path = data_path
        self._load_data()

    def _load_data(self):
        """
        加载office-31数据
        """
        classes = [
            d
            for d in os.listdir(self.data_path)
            if os.path.isdir(os.path.join(self.data_path, d))
        ]
        classes.sort()
        # 为每个类别分配一个唯一的数字索引{'back_pack':0, 'bike':1, ...}
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        self.idx_to_class = {i: cls_name for cls_name, i in self.class_to_idx.items()}

        for class_name in classes:
            class_path = os.path.join(self.data_path, class_name)
            class_idx = self.class_to_idx[class_name]
            for img_file in os.listdir(class_path):
                if img_file.lower().endswith((".jpg", ".jpeg", ".png")):
                    img_path = os.path.join(class_path, img_file)
                    self.data.append(img_path)
                    self.labels.append(class_idx)

    def __getitem__(self, idx):
        image_path = self.data[idx]
        label = self.labels[idx]
        # 加载图像
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        return image, label

    def get_class_name(self, label_idx):
        """
        根据标签索引获取类别名称
        """
        return self.idx_to_class.get(label_idx, "Unknown")


def display_dataset_statistics(dataset):
    """
    显示数据集的统计信息
    """
    print("\n=== 数据集统计信息 ===")
    print(f"总样本数: {len(dataset)}")
    print(f"类别数: {len(dataset.class_to_idx)}")

    # 统计每个类别的样本数
    class_counts = {}
    for label in dataset.labels:
        class_name = dataset.get_class_name(label)
        class_counts[class_name] = class_counts.get(class_name, 0) + 1

    print("\n各类别样本分布:")
    for class_name, count in sorted(class_counts.items()):
        print(f"  {class_name}: {count} 个样本")

    print("=====================\n")


def display_sample(dataset, index=None):
    """
    显示数据集中的一个样本

    参数:
        dataset: Office31Dataset实例
        index: 要显示的样本索引，如果为None则随机选择
    """
    if len(dataset) == 0:
        print("数据集为空!")
        return

    # 如果未指定索引，随机选择一个
    if index is None:
        index = np.random.randint(0, len(dataset))

    # 获取样本
    image, label = dataset[index]

    # 获取类别名称
    class_name = dataset.get_class_name(label)

    # 显示图像和标签信息
    plt.figure(figsize=(8, 6))
    plt.imshow(image)
    plt.title(f"sample #{index}\nclass_name: {class_name} (label: {label})")
    plt.axis("off")
    plt.tight_layout()
    plt.show()

    # 打印详细信息
    print(f"样本索引: {index}")
    print(f"图像路径: {dataset.data[index]}")
    print(f"类别名称: {class_name}")
    print(f"标签索引: {label}")
    print(f"数据集大小: {len(dataset)} 个样本")
    print(f"类别数量: {len(dataset.class_to_idx)}")


if __name__ == "__main__":
    env = VehicleEnvironment()
    dataSimu = DomainIncrementalDataSimulator(env)
    print(dataSimu.vehicle_env)
    print(dataSimu.current_dataset)
    dataSimu.update_session(1)
    print(f"当前域为{dataSimu.get_current_domain()}")
    num_vehicles = len(dataSimu.vehicle_env.vehicles)
    num_classes = dataSimu.dataset_info[config.CURRENT_DATASET]["num_classes"]
    for class_idx, distribution in dataSimu.class_distributions.items():
        print(f"类别{class_idx}")
        print(len(distribution))
        print(abs(np.sum(distribution)))
    test_vehicles = [
        0,
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
        11,
        12,
        13,
        14,
        15,
        16,
        17,
        18,
        19,
    ]
    # dataSimu._assign_domain_data_to_vehicles()
    for vehicle_id in test_vehicles:
        print(f"车辆{vehicle_id}")
        data_batches = dataSimu.generate_vehicle_data(vehicle_id, num_batches=2)
        for batch in data_batches:
            sample_batch = next(iter(batch))
            if isinstance(sample_batch, (list, tuple)):
                inputs, targets = sample_batch
                print(f"targets:{targets}")
