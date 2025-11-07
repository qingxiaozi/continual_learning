import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset, random_split
from torchvision import transforms
from PIL import Image
import scipy.io as sio
from collections import defaultdict
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.parameters import config
import matplotlib.pyplot as plt


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
        self.num_vehicles = config.NUM_VEHICLES
        self.current_dataset = config.CURRENT_DATASET
        self.current_domain_idx = 0
        self.current_session = 0

        # # 初始化黄金模型
        # self.golden_model = GoldenModelManager(self.current_dataset)

        # 域序列
        self.domain_sequences = config.DOMAIN_SEQUENCES
        self.current_domains = self.domain_sequences[self.current_dataset]

        # 已见域记录
        self.seen_domains = []  # 记录已经出现过的域
        self.seen_domains_test_sets = {}  # 存储已见域的测试集

        # 数据集信息
        self.dataset_info = {
            "office31": {
                "num_classes": config.OFFICE31_CLASSES,
                "domains": ["amazon", "webcam", "dslr"],
                "data_loader": self._load_office31_data,
            },
            "digit10": {
                "num_classes": config.DIGIT10_CLASSES,
                "domains": ["mnist", "emnist", "usps", "svhn"],
                "data_loader": self._load_digit10_data,
            },
            "domainnet": {
                "num_classes": config.DOMAINNET_CLASSES,
                "domains": [
                    "clipart",
                    "infograph",
                    "painting",
                    "quickdraw",
                    "real",
                    "sketch",
                ],
                "data_loader": self._load_domainnet_data,
            },
        }

        # 数据缓存（现在分别缓存训练集和测试集）
        self.train_data_cache = {}  # {domain_key: train_dataset}
        self.test_data_cache = {}  # {domain_key: test_dataset}

        self.vehicle_data_assignments = (
            {}
        )  # 每辆车的数据，key为数据集_域名（office31_amazon），值为一个字典，其中键为vehicle_id，值为该车辆在该域中分配到的索引列表
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
        num_vehicles = self.num_vehicles
        num_classes = self.dataset_info[self.current_dataset]["num_classes"]
        # 为每个类别生成狄利克雷分布，存储每个类别在不同车辆上的分配比例
        self.class_distributions = {}

        for class_idx in range(num_classes):
            # 使用狄利克雷分布生成每个类别的车辆分配比例
            alpha = np.full(num_vehicles, config.DIRICHLET_ALPHA)
            distribution = np.random.dirichlet(alpha)
            self.class_distributions[class_idx] = distribution
        print("\n=== 数据环境初始化完成 ===")
        print(f"狄利克雷分布系数 α = {config.DIRICHLET_ALPHA}")
        print(f"初始化{num_vehicles}辆车的数据分配")
        print("=====================\n")

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
        old_session = self.current_session
        old_domain = self.get_current_domain()  # 在更新前获取旧域

        self.current_session = session_id
        current_domain = self.get_current_domain()  # 在更新后获取新域

        # 确保当前域被记录到已见域中
        if current_domain not in self.seen_domains:
            self.seen_domains.append(current_domain)
            print(f"新域 {current_domain} 已加入已见域")

        # 只有在域实际发生变化时才打印切换信息
        if old_domain != current_domain:
            old_domain_display = old_domain if old_session > 0 else "初始域"
            print(
                f"Session {session_id}: 域切换 {old_domain_display} -> {current_domain}"
            )

        # 预加载当前域的数据集
        self._preload_domain_test_set(current_domain)

    def _preload_domain_test_set(self, domain):
        """预加载域的测试集并保存到已见域测试集"""
        domain_key = f"{self.current_dataset}_{domain}"

        if domain_key not in self.test_data_cache:
            # 加载完整数据集并分割
            full_dataset = self._load_domain_data(domain)
            if len(full_dataset) == 0:
                print(f"警告: 域 {domain} 的数据集为空")
                return
            train_size = int(config.TRAIN_TEST_SPLIT * len(full_dataset))
            test_size = len(full_dataset) - train_size
            if test_size <= 0:
                print(f"警告: 域 {domain} 的测试集大小为0")
                return
            train_dataset, test_dataset = random_split(
                full_dataset, [train_size, test_size]
            )

            # 缓存
            self.train_data_cache[domain_key] = train_dataset
            self.test_data_cache[domain_key] = test_dataset

        # 保存到已见域测试集
        if domain_key not in self.seen_domains_test_sets:
            self.seen_domains_test_sets[domain_key] = self.test_data_cache[domain_key]
            print(
                f"已加载 {domain} 域的测试集，样本数: {len(self.test_data_cache[domain_key])}"
            )

    def generate_vehicle_data(self, vehicle_id, num_batches=None):
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
        # 由于在切换域后，训练集和测试集已经分别放至train_data_cache和test_data_cache，因此此处可直接取用
        train_dataset = self.train_data_cache[domain_key]
        # 获取该车辆的训练数据子集索引
        vehicle_indices = self._get_vehicle_data_indices(vehicle_id, train_dataset)

        if not vehicle_indices:
            print(
                f"警告: 车辆 {vehicle_id} 在当前域 {current_domain} 中没有分配到训练数据"
            )
            return []

        # 创建车辆特定的训练数据集，Subset(original_dataset, indices)
        vehicle_dataset = Subset(train_dataset, vehicle_indices)
        # 计算实际可用的最大批次数量
        max_batches = len(vehicle_indices) // config.BATCH_SIZE
        if num_batches is None:
            num_batches = max_batches
        else:
            num_batches = min(num_batches, max_batches)
        # 创建数据批次
        dataloader = DataLoader(
            vehicle_dataset, batch_size=config.BATCH_SIZE, shuffle=True, drop_last=True
        )

        # 收集所有批次
        batches = []
        for i, batch in enumerate(dataloader):
            if i >= num_batches:
                break
            batches.append(batch)

        print(
            f"车辆 {vehicle_id} 在域 {current_domain} 中共有 {len(batches)} 个批次，每批 {config.BATCH_SIZE} 个样本"
        )
        return batches

    def _get_vehicle_data_indices(self, vehicle_id, train_dataset):
        """
        获取指定车辆在当前域中的数据索引
        输入：
            vehicle_id：车辆id
            train_dataset：某个域的训练集
        输出：

        """
        num_vehicles = self.num_vehicles
        vehicle_idx = vehicle_id

        # 如果还没有为该域分配数据，则进行分配
        domain_key = f"{self.current_dataset}_{self.get_current_domain()}"
        if domain_key not in self.vehicle_data_assignments:
            self._assign_domain_data_to_vehicles(train_dataset)

        return self.vehicle_data_assignments[domain_key].get(vehicle_idx, [])

    def _assign_domain_data_to_vehicles(self, train_dataset):
        """
        使用狄利克雷分布将域训练数据分配给各个车辆
        """
        num_vehicles = self.num_vehicles
        domain_key = f"{self.current_dataset}_{self.get_current_domain()}"

        # 按类别组织训练数据索引
        class_indices = defaultdict(list)
        for idx in range(len(train_dataset)):
            _, label = train_dataset[idx]
            class_indices[label].append(idx)

        # 为每个车辆分配训练数据索引
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

        print(f"\n=== {domain_key} 训练数据分配统计 ===")
        print(f"总训练样本数: {total_samples}")
        print(
            f"车辆训练数据量: 平均{np.mean(vehicle_counts):.1f}, "
            f"最小{min(vehicle_counts)}, 最大{max(vehicle_counts)}"
        )
        print(f"训练数据异构程度: 标准差{np.std(vehicle_counts):.1f}")
        print("==============================\n")

    def _load_domain_data(self, domain):
        """加载指定域的数据(未分割)"""
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

    def _load_digit10_data(self, domain):
        """加载Digit10数据集"""
        dataset_path = os.path.join(config.DATA_BASE_PATH, "digit10", domain)
        if not os.path.exists(dataset_path):
            print(f"警告：digit10 {domain}路径不存在：{dataset_path}")
            # return self._create_simulated_dataset(config.OFFICE31_CLASSES, 1000)
        return Digit10Dataset(dataset_path, transform=self.transform)

    def _load_domainnet_data(self, domain):
        """加载DomainNet数据集"""
        dataset_path = os.path.join(config.DATA_BASE_PATH, "domainnet", domain)
        if not os.path.exists(dataset_path):
            print(f"警告：domainnet {domain}路径不存在：{dataset_path}")
            # return self._create_simulated_dataset(config.OFFICE31_CLASSES, 1000)
        return DomainNetDataset(dataset_path, transform=self.transform)

    def evaluate_model(self, model, strategy=None):
        """
        评估模型性能

        参数:
            model: 要评估的模型
            strategy: 测试策略 ('current', 'cumulative')，如果为None则使用配置的策略

        返回:
            dict: 包含各种评估指标的字典
        """
        if strategy is None:
            strategy = config.TEST_STRATEGY

        results = {}

        if strategy == "current":
            # 仅评估当前域
            current_domain = self.get_current_domain()
            current_results = self._evaluate_on_domain(model, current_domain)
            results["current_domain"] = current_results
            results["current_domain_name"] = current_domain

        elif strategy == "cumulative":
            # 评估所有已见域
            cumulative_results = {}

            for domain in self.seen_domains:
                domain_key = f"{self.current_dataset}_{domain}"
                if domain_key in self.seen_domains_test_sets:
                    domain_results = self._evaluate_on_domain(model, domain)
                    cumulative_results[domain] = domain_results

            # 计算平均性能
            if cumulative_results:
                accuracies = [
                    result["accuracy"] for result in cumulative_results.values()
                ]
                losses = [result["loss"] for result in cumulative_results.values()]

                results["cumulative"] = {
                    "average_accuracy": np.mean(accuracies),
                    "average_loss": np.mean(losses),
                    "domain_results": cumulative_results,
                }

        return results

    def _evaluate_on_domain(self, model, domain):
        """在指定域上评估模型"""
        domain_key = f"{self.current_dataset}_{domain}"

        if domain_key not in self.test_data_cache:
            print(f"警告: 域 {domain} 的测试数据未加载")
            return {"accuracy": 0.0, "loss": 0.0}

        test_dataset = self.test_data_cache[domain_key]
        test_loader = DataLoader(
            test_dataset, batch_size=config.BATCH_SIZE, shuffle=False
        )

        # 计算准确率和损失
        model.eval()
        correct = 0
        total = 0
        total_loss = 0.0
        criterion = torch.nn.CrossEntropyLoss()

        with torch.no_grad():
            for inputs, targets in test_loader:
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                total_loss += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()

        accuracy = correct / total if total > 0 else 0.0
        avg_loss = total_loss / len(test_loader) if len(test_loader) > 0 else 0.0

        return {"accuracy": accuracy, "loss": avg_loss, "samples": total}

    def get_test_dataset(self, domain=None):
        """获取指定域的测试数据集"""
        if domain is None:
            domain = self.get_current_domain()

        domain_key = f"{self.current_dataset}_{domain}"
        return self.test_data_cache.get(domain_key, None)

    def get_cumulative_test_datasets(self):
        """获取所有已见域的测试数据集"""
        cumulative_datasets = {}
        for domain in self.seen_domains:
            domain_key = f"{self.current_dataset}_{domain}"
            if domain_key in self.seen_domains_test_sets:
                cumulative_datasets[domain] = self.seen_domains_test_sets[domain_key]

        return cumulative_datasets

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
            "seen_domains": self.seen_domains.copy(),
            "has_data_assignment": domain_key in self.vehicle_data_assignments,
        }

        if domain_key in self.vehicle_data_assignments:
            assignments = self.vehicle_data_assignments[domain_key]
            sample_counts = [len(indices) for indices in assignments.values()]
            info.update(
                {
                    "total_train_samples": sum(sample_counts),
                    "avg_train_samples_per_vehicle": np.mean(sample_counts),
                    "std_train_samples_per_vehicle": np.std(sample_counts),
                    "min_train_samples": min(sample_counts),
                    "max_train_samples": max(sample_counts),
                }
            )
        else:
            # 即使分配失败，也提供默认值
            info.update(
                {
                    "total_train_samples": 0,
                    "avg_train_samples_per_vehicle": 0,
                    "std_train_samples_per_vehicle": 0,
                    "min_train_samples": 0,
                    "max_train_samples": 0,
                }
            )

        # 添加测试集信息
        if domain_key in self.test_data_cache:
            info["current_test_samples"] = len(self.test_data_cache[domain_key])

        info["total_test_samples"] = sum(
            len(dataset) for dataset in self.seen_domains_test_sets.values()
        )

        return info

    # def prepare_golden_model(self, train_all_domains=True):
    #     """
    #     准备黄金模型

    #     参数:
    #         train_all_domains: 是否使用所有域的数据训练黄金模型
    #     """
    #     print(f"准备黄金模型，数据集: {self.current_dataset}")

    #     # 检查黄金模型是否已经训练
    #     if os.path.exists(self.golden_model.model_path):
    #         print("黄金模型已存在，跳过训练")
    #         return

    #     # 收集训练数据
    #     if train_all_domains:
    #         # 使用所有域的数据训练
    #         train_datasets = []
    #         for domain in self.domain_sequences[self.current_dataset]:
    #             domain_data = self._load_domain_data(domain)
    #             train_datasets.append(domain_data)

    #         # 合并所有域的数据
    #         from torch.utils.data import ConcatDataset
    #         full_train_dataset = ConcatDataset(train_datasets)

    #         # 分割训练集和验证集
    #         train_size = int(0.8 * len(full_train_dataset))
    #         val_size = len(full_train_dataset) - train_size
    #         train_dataset, val_dataset = torch.utils.data.random_split(
    #             full_train_dataset, [train_size, val_size]
    #         )

    #     else:
    #         # 只使用当前域的数据训练
    #         full_dataset = self._load_domain_data(self.get_current_domain())

    #         # 分割训练集和验证集
    #         train_size = int(0.8 * len(full_dataset))
    #         val_size = len(full_dataset) - train_size
    #         train_dataset, val_dataset = torch.utils.data.random_split(
    #             full_dataset, [train_size, val_size]
    #         )

    #     # 微调黄金模型
    #     self.golden_model.fine_tune(train_dataset, val_dataset)

    #     # 评估黄金模型在所有域上的性能
    #     self._evaluate_golden_model_on_all_domains()

    # def _evaluate_golden_model_on_all_domains(self):
    #     """评估黄金模型在所有域上的性能"""
    #     print("\n=== 黄金模型域性能评估 ===")

    #     for domain in self.domain_sequences[self.current_dataset]:
    #         domain_data = self._load_domain_data(domain)
    #         data_loader = DataLoader(domain_data, batch_size=32, shuffle=False)

    #         accuracy = self.golden_model.evaluate(data_loader)
    #         print(f"域 {domain}: {accuracy:.2f}%")

    #     print("==========================\n")

    # def label_uploaded_data(self, uploaded_data):
    #     """
    #     为上传的数据生成标签

    #     参数:
    #         uploaded_data: 上传的数据批次列表

    #     返回:
    #         list: 标注后的数据批次列表
    #     """
    #     labeled_data = []

    #     for data_batch in uploaded_data:
    #         # 使用黄金模型生成标签
    #         batch_labeled_data = self.golden_model.label_data(data_batch)
    #         labeled_data.extend(batch_labeled_data)

    #     print(f"数据标注完成: {len(labeled_data)} 个批次已标注")
    #     return labeled_data


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
        print("Digit10Dataset待补充")


class DomainNetDataset(BaseDataset):
    def __init__(self, data_path, transform=None):
        super().__init__(transform)
        self.data_path = data_path
        self._load_data()

    def _load_data(self):
        print("DomainNetDataset待补充")


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


if __name__ == "__main__":
    # 1. 初始化车辆环境
    print("1. 初始化车辆环境...")
    data_simulator = DomainIncrementalDataSimulator()

    # 2. 显示初始信息
    print(f"车辆数量: {data_simulator.num_vehicles}")
    print(f"当前数据集: {data_simulator.current_dataset}")
    print(f"可用域: {data_simulator.current_domains}")
    print(f"狄利克雷参数 α: {config.DIRICHLET_ALPHA}")
    # 3. 演示域增量切换
    print("\n2. 域增量切换演示...")
    sessions_to_demo = [0, 1, 2, 3, 4]  # 演示的会话点

    for session in sessions_to_demo:
        print(f"\n--- Session {session} ---")
        # 更新会话
        data_simulator.update_session(session)
        current_domain = data_simulator.get_current_domain()
        print(f"当前域: {current_domain}")
        print(f"已见域: {data_simulator.seen_domains}")
        # 获取数据分布信息
        dist_info = data_simulator.get_data_distribution_info()
        print(f"训练数据总量: {dist_info['total_train_samples']}")
        print(f"测试数据总量: {dist_info['total_test_samples']}")
        # 为前3辆车分配数据并显示信息
        print("\n车辆数据分配示例:")
        for vehicle_id in range(data_simulator.num_vehicles):
            # 生成车辆数据
            vehicle_data = data_simulator.generate_vehicle_data(vehicle_id)

            if vehicle_data:
                total_samples = len(vehicle_data) * config.BATCH_SIZE
                print(
                    f"  车辆 {vehicle_id}: {total_samples} 个训练样本, {len(vehicle_data)} 个批次"
                )
            else:
                print(f"  车辆 {vehicle_id}: 无训练数据")
