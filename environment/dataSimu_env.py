import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset, random_split
from torchvision import transforms
from PIL import Image
import scipy.io as sio
import logging
from collections import defaultdict
from config.parameters import Config
from learning.evaluator import ModelEvaluator
from utils.metrics import IncrementalMetricsCalculator
import matplotlib.pyplot as plt
logger = logging.getLogger(__name__)


class DomainIncrementalDataSimulator:
    """
    域增量数据模拟器

    功能：
    1. 支持多个数据集的域增量学习
    2. 每个域划分为训练集和测试集
    3. 使用狄利克雷分布实现车辆数据的异构分配，只分配训练集
    4. 动态切换域以模拟真实环境中的分布漂移
    5. 计算测试损失（待定）
    """

    def __init__(self):
        self.num_vehicles = Config.NUM_VEHICLES
        self.current_dataset = Config.CURRENT_DATASET
        self.current_session = 0

        # 域序列
        self.domain_sequences = Config.DOMAIN_SEQUENCES
        self.current_domains = self.domain_sequences[self.current_dataset]

        # 已见域记录
        self.seen_domains = []  # 记录已经出现过的域
        self.accuracy_history = {}      # {domain: [acc₁, acc₂, ...]}
        self.aa_history = []   # [AA_1, AA_2, ..., AA_k]

        # 数据集信息
        self.dataset_info = {
            "office31": {
                "num_classes": Config.OFFICE31_CLASSES,
                "domains": ["amazon", "webcam", "dslr"],
                "base_path": os.path.join(Config.DATA_BASE_PATH, "office-31"),
                "dataset_class": Office31Dataset,
            },
            "digit10": {
                "num_classes": Config.DIGIT10_CLASSES,
                "domains": ["mnist", "emnist", "usps", "svhn"],
                "base_path": os.path.join(Config.DATA_BASE_PATH, "digit10"),
                "dataset_class": Digit10Dataset,
            },
            "domainnet": {
                "num_classes": Config.DOMAINNET_CLASSES,
                "domains": [
                    "clipart",
                    "infograph",
                    "painting",
                    "quickdraw",
                    "real",
                    "sketch",
                ],
                "base_path": os.path.join(Config.DATA_BASE_PATH, "domainnet"),
                "dataset_class": DomainNetDataset,
            },
        }

        self.domain_label_cache = {}
        self.train_indices_cache = {}  # {domain_key: [orig_idx1, orig_idx2, ...]}
        # 数据缓存（现在分别缓存训练集、测试集和验证集）
        self.train_data_cache = {}  # {domain_key: train_dataset}
        self.test_data_cache = {}  # {domain_key: test_dataset}
        self.val_data_cache = {}

        self.vehicle_data_assignments = (
            {}
        )  # 每辆车的数据，key为数据集_域名（office31_amazon），值为一个字典，其中键为vehicle_id，值为该车辆在该域中分配到的索引列表

        # 1. 基础预处理（用于缓存原始数据）
        self.base_transform = transforms.Compose(
            [
                transforms.Resize((Config.IMAGE_SIZE, Config.IMAGE_SIZE)),
                transforms.Lambda(lambda x: x.convert("RGB") if x.mode != "RGB" else x),
                transforms.ToTensor(),  # 只转换到Tensor，不归一化
            ]
        )

        # 2. 训练时的增强+归一化
        self.train_transform = transforms.Compose(
            [
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                )
            ]
        )

        # 3. 验证/测试时的归一化
        self.test_transform = transforms.Compose(
            [
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                )
            ]
        )

        self.transform = self.base_transform
        # 初始化数据分配
        self._initialize_data_distribution()

    def _initialize_data_distribution(self):
        """
        使用狄利克雷分布初始化车辆数据分配
        """
        num_vehicles = self.num_vehicles
        num_classes = self.dataset_info[self.current_dataset]["num_classes"]
        # 为每个类别生成狄利克雷分布，存储每个类别在不同车辆上的分配比例
        self.class_distributions = {}

        for class_idx in range(num_classes):
            # 使用狄利克雷分布生成每个类别的车辆分配比例
            alpha = np.full(num_vehicles, Config.DIRICHLET_ALPHA)
            distribution = np.random.dirichlet(alpha)
            self.class_distributions[class_idx] = distribution

        print(f"初始化{num_vehicles}辆车的数据分配")

    def get_current_domain(self):
        """
        获取当前域
        """
        domain_idx = (self.current_session // Config.DOMAIN_CHANGE_INTERVAL) % len(
            self.current_domains
        )
        return self.current_domains[domain_idx]

    def update_session_dataset(self, session_id):
        """
        更新训练会话
        """
        old_domain = self.get_current_domain()  # 在更新前获取旧域
        self.current_session = session_id
        new_domain = self.get_current_domain()  # 在更新后获取新域

        # 自动管理seen_domains
        if new_domain not in self.seen_domains:
            self.seen_domains.append(new_domain)
            print(f"新域 {new_domain} 已加入已见域")

        # 域是否发生变化
        domain_changed = (old_domain != new_domain)

        # 预加载新域的数据集
        domain_key = f"{self.current_dataset}_{new_domain}"
        if domain_key not in self.test_data_cache:
            self._preload_domain_dataset(new_domain)

        return domain_changed, old_domain, new_domain

    def _preload_domain_dataset(self, domain):
        """预加载域的训练集、验证集和测试集"""
        domain_key = f"{self.current_dataset}_{domain}"

        if domain_key not in self.test_data_cache:
            # 加载完整数据集并分割
            original_dataset = self._load_domain_data(domain)
            if len(original_dataset) == 0:
                logger.warning(f"域 {domain} 的数据集为空")
                return

            # 预缓存标签 ===
            all_labels = [original_dataset[i][1] for i in range(len(original_dataset))]
            self.domain_label_cache[domain_key] = all_labels

            total_size = len(original_dataset)
            test_size = int(Config.TEST_RATIO * total_size)
            val_size = int(Config.VAL_RATIO * total_size)
            train_size = total_size - test_size - val_size

            # 设置随机种子以确保可重复性
            generator = torch.Generator()
            seed = hash(domain) % (2**32)
            generator.manual_seed(seed)

            indices = list(range(len(original_dataset)))
            train_indices, val_indices, test_indices = random_split(
                indices,
                [train_size, val_size, test_size],
                generator=generator
            )

            # 缓存训练索引
            self.train_indices_cache[domain_key] = train_indices.indices

            # 创建应用变换的子集
            train_dataset = self._create_subset_with_transform(
                original_dataset, train_indices, self.train_transform
            )
            val_dataset = self._create_subset_with_transform(
                original_dataset, val_indices, self.test_transform
            )
            test_dataset = self._create_subset_with_transform(
                original_dataset, test_indices, self.test_transform
            )

            # 将数据保存到缓存
            self.train_data_cache[domain_key] = train_dataset
            self.test_data_cache[domain_key] = test_dataset
            self.val_data_cache[domain_key] = val_dataset

    def _create_subset_with_transform(self, original_dataset, indices, transform):
        """创建应用了特定变换的子集"""

        class TransformSubset(Dataset):
            def __init__(self, dataset, indices, transform):
                self.dataset = dataset
                self.indices = indices
                self.transform = transform

            def __len__(self):
                return len(self.indices)

            def __getitem__(self, idx):
                # 从原始数据集获取数据
                image, label = self.dataset[self.indices[idx]]
                # 应用新的变换
                if self.transform:
                    image = self.transform(image)
                return image, label

        return TransformSubset(original_dataset, indices, transform)

    def generate_vehicle_data(self, vehicle_id):
        """
        为指定车辆生成数据批次
        输入：
            vehicle_id：车辆id
        输出：
            batches：指定车辆的数据批次
        """
        current_domain = self.get_current_domain()
        domain_key = f"{self.current_dataset}_{current_domain}"

        train_dataset = self.train_data_cache[domain_key]
        vehicle_indices = self._get_vehicle_data_indices(vehicle_id)

        if not vehicle_indices:
            logger.warning(f"车辆 {vehicle_id} 在域 {current_domain} 中无训练数据")
            return []

        # 创建车辆特定的训练数据集，Subset(original_dataset, indices)
        vehicle_dataset = Subset(train_dataset, vehicle_indices)

        # 创建数据批次
        dataloader = DataLoader(
            vehicle_dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            drop_last=True
        )

        # 收集所有批次
        batches = list(dataloader)
        logger.info(f"车辆 {vehicle_id} 在域 {current_domain} 中共有 {len(batches)} 个数据批次")
        return batches

    def _get_vehicle_data_indices(self, vehicle_id):
        """
        获取指定车辆在当前域中的数据索引
        输入：
            vehicle_id：车辆id
        输出：
            List[int]: 该车辆分配到的样本索引列表
        """
        vehicle_idx = vehicle_id

        # 如果还没有为该域分配数据，则进行分配
        domain_key = f"{self.current_dataset}_{self.get_current_domain()}"
        if domain_key not in self.vehicle_data_assignments:
            self._assign_domain_data_to_vehicles()

        return self.vehicle_data_assignments[domain_key].get(vehicle_idx, [])

    def _assign_domain_data_to_vehicles(self):
        """
        使用狄利克雷分布将域训练数据分配给各个车辆
        """
        domain_key = f"{self.current_dataset}_{self.get_current_domain()}"
        orig_train_indices = self.train_indices_cache[domain_key]
        train_labels = [self.domain_label_cache[domain_key][i] for i in orig_train_indices]

        # 按类别组织训练数据索引
        class_indices = defaultdict(list)
        for idx, label in enumerate(train_labels):
            class_indices[label].append(idx)

        # 为每个车辆分配训练数据索引
        vehicle_assignments = {i: [] for i in range(self.num_vehicles)}

        for class_idx, indices in class_indices.items():

            #为每个类别生成一个狄利克雷分布
            if class_idx not in self.class_distributions:

                # 如果类别不在初始分布中，创建新的分布
                alpha = np.full(self.num_vehicles, Config.DIRICHLET_ALPHA)
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
                extra_assignments = np.random.choice(self.num_vehicles, remaining)
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

    def _load_domain_data(self, domain):
        """加载指定域的所有数据"""
        info = self.dataset_info[self.current_dataset]
        path = os.path.join(info["base_path"], domain)
        if not os.path.exists(path):
            logger.warning(f"Path not found: {path}")
        return info["dataset_class"](path, transform=self.base_transform)

    def evaluate_model(self, model, strategy=None):
        """
        评估模型在增量学习设置下的性能（支持 current / cumulative 策略）
        参数:
            model: 要评估的模型
            strategy: 测试策略 ('current', 'cumulative')，如果为None则使用配置的策略

        返回:
            dict: 包含各种评估指标的字典
        """
        if strategy is None:
            strategy = Config.TEST_STRATEGY

        # 计算当前域的性能
        current_domain = self.get_current_domain()

        # 保存当前评估结果到历史
        if not hasattr(self, 'accuracy_history'):
            self.accuracy_history = {}  # 存储格式: {domain: [a1, a2, ..., ak]}
        if not hasattr(self, 'aa_history'):
            self.aa_history = []

        ## Step 1: 评估所有已见域（包括当前域）
        cumulative_results = {}
        for domain in self.seen_domains:
            cumulative_results[domain] = self._evaluate_on_domain(model, domain)

        # Step 2: 更新 accuracy_history
        for domain, result in cumulative_results.items():
            if domain not in self.accuracy_history:
                self.accuracy_history[domain] = []
            self.accuracy_history[domain].append(result["accuracy"])

        # Step 3: 计算指标
        metrics = IncrementalMetricsCalculator.compute_metrics(
            self.seen_domains, self.accuracy_history
        )

        # Step 4: 计算并更新 AIA
        if "AA" in metrics:
            self.aa_history.append(metrics["AA"])
            aia = IncrementalMetricsCalculator.compute_aia(self.aa_history)
            metrics["AIA"] = aia

        # Step 5: 构建返回结果
        current_result = cumulative_results.get(current_domain, {"accuracy": 0.0, "loss": 0.0})
        accuracies = [r["accuracy"] for r in cumulative_results.values()]
        losses = [r["loss"] for r in cumulative_results.values()]

        return {
            "current_domain": current_result,
            "current_domain_name": current_domain,
            "cumulative": {
                "average_accuracy": np.mean(accuracies),
                "average_loss": np.mean(losses),
                "domain_results": cumulative_results,
            },
            "metrics": metrics
        }

    def _evaluate_on_domain(self, model, domain):
        """在指定域上评估模型"""
        domain_key = f"{self.current_dataset}_{domain}"

        if domain_key not in self.test_data_cache:
            logger.warning(f"{domain_key} 域的测试数据未加载:")
            return {"accuracy": 0.0, "loss": 0.0}

        test_loader = DataLoader(
            self.test_data_cache[domain_key],
            batch_size=Config.BATCH_SIZE,
            shuffle=False
        )

        evalator = ModelEvaluator()
        accuracy, avg_loss = evalator.evaluate_model(model, test_loader)

        return {"accuracy": accuracy, "loss": avg_loss}

    def get_test_dataset(self, domain=None):
        """获取指定域的测试数据集"""
        if domain is None:
            domain = self.get_current_domain()

        domain_key = f"{self.current_dataset}_{domain}"
        return self.test_data_cache.get(domain_key, None)

    def get_val_dataset(self, domain=None):
        """获取指定域的验证数据集"""
        if domain is None:
            domain = self.get_current_domain()

        domain_key = f"{self.current_dataset}_{domain}"
        return self.val_data_cache.get(domain_key, None)

    def get_cumulative_test_datasets(self):
        """获取所有已见域的测试数据集"""
        return {
            domain: self.test_data_cache[f"{self.current_dataset}_{domain}"]
            for domain in self.seen_domains
            if f"{self.current_dataset}_{domain}" in self.test_data_cache
        }

    def reset(self):
        """
        重置数据模拟器状态，用于新 episode 开始
        1. 重置会话计数器
        2. 清空已见域列表
        3. 清空数据缓存
        4. 重新初始化数据分配
        """
        self.current_session = 0
        self.seen_domains = []
        self.train_data_cache.clear()
        self.test_data_cache.clear()
        self.val_data_cache.clear()
        self.vehicle_data_assignments.clear()
        self._initialize_data_distribution()
        logger.info("数据重置完成")

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


class Digit10Dataset(BaseDataset):
    def __init__(self, data_path, transform=None):
        super().__init__(transform)
        self.data_path = data_path
        self._load_data()

    def _load_data(self):
        """
        加载digit10数据
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


class DomainNetDataset(BaseDataset):
    def __init__(self, data_path, transform=None):
        super().__init__(transform)
        self.data_path = data_path
        self._load_data()

    def _load_data(self):
        print("DomainNetDataset待补充")

    def __getitem__(self, idx):
        image_path = self.data[idx]
        label = self.labels[idx]
        # 加载图像
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        return image, label


if __name__ == "__main__":
    # 1. 初始化车辆环境
    print("1. 初始化车辆环境...")
    data_simulator = DomainIncrementalDataSimulator()
    print("\n2. 域增量切换演示...")
    sessions_to_demo = [0, 1, 2, 3, 4]  # 演示的会话点

    for session in sessions_to_demo:
        print(f"\n--- Session {session} ---")
        # 更新会话
        data_simulator.update_session_dataset(session)
        current_domain = data_simulator.get_current_domain()
        # 为前3辆车分配数据并显示信息
        print("\n车辆数据分配示例:")
        for vehicle_id in range(data_simulator.num_vehicles):
            # 生成车辆数据
            vehicle_data = data_simulator.generate_vehicle_data(vehicle_id)

            if vehicle_data:
                total_samples = len(vehicle_data) * Config.BATCH_SIZE
                print(
                    f"  车辆 {vehicle_id}: {total_samples} 个训练样本, {len(vehicle_data)} 个批次"
                )
            else:
                print(f"  车辆 {vehicle_id}: 无训练数据")