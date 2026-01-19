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
        self.current_domain_idx = 0
        self.current_session = 0

        # 域序列
        self.domain_sequences = Config.DOMAIN_SEQUENCES
        self.current_domains = self.domain_sequences[self.current_dataset]

        # 已见域记录
        self.seen_domains = []  # 记录已经出现过的域
        self.seen_domains_test_sets = {}  # 存储已见域的测试集

        # 数据集信息
        self.dataset_info = {
            "office31": {
                "num_classes": Config.OFFICE31_CLASSES,
                "domains": ["amazon", "webcam", "dslr"],
                "data_loader": self._load_office31_data,
            },
            "digit10": {
                "num_classes": Config.DIGIT10_CLASSES,
                "domains": ["mnist", "emnist", "usps", "svhn"],
                "data_loader": self._load_digit10_data,
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
                "data_loader": self._load_domainnet_data,
            },
        }

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
        print("\n=== 数据环境初始化完成 ===")
        print(f"狄利克雷分布系数 α = {Config.DIRICHLET_ALPHA}")
        print(f"初始化{num_vehicles}辆车的数据分配")
        print("=====================\n")

    def get_current_domain(self):
        """
        获取当前域
        """
        domain_idx = (self.current_session // Config.DOMAIN_CHANGE_INTERVAL) % len(
            self.current_domains
        )
        return self.current_domains[domain_idx]

    def update_session(self, session_id):
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

        # 域发生变化时打印切换信息
        domain_changed = (old_domain != new_domain)
        if domain_changed:
            old_display = old_domain if self.current_session > 0 else "初始域"
            print(f"Session {session_id}: 域切换 {old_display} -> {new_domain}")

        # 预加载新域的数据集
        self._preload_domain_dataset(new_domain)

        return domain_changed, old_domain, new_domain

    def _preload_domain_dataset(self, domain):
        """预加载域的训练集、验证集和测试集"""
        domain_key = f"{self.current_dataset}_{domain}"

        if domain_key not in self.test_data_cache:
            # 加载完整数据集并分割
            original_dataset = self._load_domain_data(domain)
            if len(original_dataset) == 0:
                print(f"警告: 域 {domain} 的数据集为空")
                return

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

        # 保存到已见域测试集，仅首次
        if domain_key not in self.seen_domains_test_sets:
            self.seen_domains_test_sets[domain_key] = self.test_data_cache[domain_key]
            print(
                f"已加载 {domain} 域的数据集 - 训练集: {len(self.train_data_cache[domain_key])}, "
                f"验证集: {len(self.val_data_cache[domain_key])}, "
                f"测试集: {len(self.test_data_cache[domain_key])}"
            )

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
        vehicle_indices = self._get_vehicle_data_indices(vehicle_id, train_dataset)

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

    def _get_vehicle_data_indices(self, vehicle_id, train_dataset):
        """
        获取指定车辆在当前域中的数据索引
        输入：
            vehicle_id：车辆id
            train_dataset：某个域的训练集
        输出：

        """
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

            #为每个类别生成一个狄利克雷分布
            if class_idx not in self.class_distributions:

                # 如果类别不在初始分布中，创建新的分布
                alpha = np.full(num_vehicles, Config.DIRICHLET_ALPHA)
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
        print("=======================================\n")

    def _load_domain_data(self, domain):
        """加载指定域的数据(未分割)"""
        loader_func = self.dataset_info[self.current_dataset]["data_loader"]
        return loader_func(domain, transform=self.base_transform)

    def _load_office31_data(self, domain, transform=None):
        """
        加载office-31数据集
        """
        dataset_path = os.path.join(Config.DATA_BASE_PATH, "office-31", domain)
        if not os.path.exists(dataset_path):
            print(f"警告：Office-31 {domain}路径不存在：{dataset_path}")
        return Office31Dataset(dataset_path, transform=transform)

    def _load_digit10_data(self, domain, transform=None):
        """加载Digit10数据集"""
        dataset_path = os.path.join(Config.DATA_BASE_PATH, "digit10", domain)
        if not os.path.exists(dataset_path):
            print(f"警告：digit10 {domain}路径不存在：{dataset_path}")
        return Digit10Dataset(dataset_path, transform=transform)

    def _load_domainnet_data(self, domain, transform=None):
        """加载DomainNet数据集"""
        dataset_path = os.path.join(Config.DATA_BASE_PATH, "domainnet", domain)
        if not os.path.exists(dataset_path):
            print(f"警告：domainnet {domain}路径不存在：{dataset_path}")
        return DomainNetDataset(dataset_path, transform=transform)

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
            strategy = Config.TEST_STRATEGY

        results = {}

        # 计算当前域的性能
        current_domain = self.get_current_domain()
        current_results = self._evaluate_on_domain(model, current_domain)
        results["current_domain"] = current_results
        results["current_domain_name"] = current_domain

        # 保存当前评估结果到历史
        if not hasattr(self, 'accuracy_history'):
            self.accuracy_history = {}  # 存储格式: {domain: [a1, a2, ..., ak]}
        if not hasattr(self, 'aa_history'):
            self.aa_history = []

        # 获取当前任务/域索引
        current_task_idx = self.seen_domains.index(current_domain) if current_domain in self.seen_domains else -1
        k = current_task_idx + 1  # k: 当前已学习的任务数量

        # 评估所有已见域的性能
        cumulative_results = {}
        for domain in self.seen_domains:
            if domain == current_domain:
                cumulative_results[domain] = current_results
            else:
                domain_results = self._evaluate_on_domain(model, domain)
                cumulative_results[domain] = domain_results

        # 更新历史记录
        for domain, result in cumulative_results.items():
            if domain not in self.accuracy_history:
                self.accuracy_history[domain] = []
            self.accuracy_history[domain].append(result["accuracy"])

        # 提取准确率矩阵的当前行
        current_accuracies = {}
        for domain in self.seen_domains:
            if domain in self.accuracy_history:
                history = self.accuracy_history[domain]
                # 当前性能是历史记录的最后一个值
                current_accuracies[domain] = history[-1] if history else 0.0

        aa_k = 0.0
        aia_k = 0.0
        fm_k = 0.0
        bwt_k = 0.0

        # 计算四种指标
        if k > 0:  # 至少学习了一个任务
            # 1. 平均准确率 AA_k
            aa_k = np.mean(list(current_accuracies.values()))

            # 2. 平均增量准确率 AIA_k
            if not hasattr(self, 'aa_history'):
                self.aa_history = []
            self.aa_history.append(aa_k)
            aia_k = np.mean(self.aa_history)

            # 3. 遗忘度量 FM_k (需要修正)
            if k > 1:
                forgetting_values = []
                for domain in self.seen_domains[:k-1]:  # 遍历前k-1个任务
                    if domain in self.accuracy_history:
                        history = self.accuracy_history[domain]
                        if history:
                            # 历史最佳：该任务所有历史评估中的最大值
                            max_historical = max(history)
                            current_acc = current_accuracies.get(domain, 0.0)
                            forgetting = max_historical - current_acc
                            forgetting_values.append(forgetting)

                fm_k = np.mean(forgetting_values) if forgetting_values else 0.0

            # 4. 反向迁移 BWT_k (需要重大修正)
            if k > 1:
                bwt_values = []
                for j, domain in enumerate(self.seen_domains[:k-1], 1):  # j从1到k-1
                    if domain in self.accuracy_history:
                        history = self.accuracy_history[domain]
                        if len(history) > 0:
                            # 关键修正：a_{j,j}是任务j首次出现时的准确率
                            # 任务j在第j次评估时首次出现，所以应该是history[0]
                            # 因为每个任务的历史列表从它首次出现开始记录
                            initial_acc = history[0]  # 修正：用history[0]而不是history[j-1]
                            current_acc = current_accuracies.get(domain, 0.0)
                            bwt = current_acc - initial_acc
                            bwt_values.append(bwt)
                            # 调试输出
                            print(f"BWT计算: 任务{domain}(j={j}), history={history}, "
                                f"初始={initial_acc:.4f}, 当前={current_acc:.4f}, BWT={bwt:.4f}")

                bwt_k = np.mean(bwt_values) if bwt_values else 0.0

            # 将指标添加到结果中
            results["metrics"] = {
                "AA": aa_k,          # 平均准确率
                "AIA": aia_k,        # 平均增量准确率
                "FM": fm_k,          # 遗忘度量
                "BWT": bwt_k,        # 反向迁移
                "k": k               # 当前任务数
            }

        # 保持原有的累积评估结果
        if cumulative_results:
            accuracies = [result["accuracy"] for result in cumulative_results.values()]
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
            test_dataset, batch_size=Config.BATCH_SIZE, shuffle=False
        )

        # 计算准确率和损失
        model.eval()
        correct = 0
        total = 0
        total_loss = 0.0
        criterion = torch.nn.CrossEntropyLoss()

        with torch.no_grad():
            for inputs, targets in test_loader:
                device = next(model.parameters()).device
                inputs = inputs.to(device)
                targets = targets.to(device)
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

    def get_val_dataset(self, domain=None):
        """获取指定域的验证数据集"""
        if domain is None:
            domain = self.get_current_domain()

        domain_key = f"{self.current_dataset}_{domain}"
        return self.val_data_cache.get(domain_key, None)

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
            "total_domains": len(self.current_domains),
            "seen_domains": self.seen_domains.copy(),
        }

        if domain_key in self.train_data_cache:
            info.update(
                {
                    "dataset_sizes": {
                        "train": len(self.train_data_cache[domain_key]),
                        "val": len(self.val_data_cache[domain_key]),
                        "test": len(self.test_data_cache[domain_key]),
                    }
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


class Digit10Dataset(BaseDataset):
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
    print(f"狄利克雷参数 α: {Config.DIRICHLET_ALPHA}")
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
        print(f"训练数据总量: {dist_info['dataset_sizes']['train']}")
        print(f"测试数据总量: {dist_info['dataset_sizes']['test']}")
        print(f"验证数据总量: {dist_info['dataset_sizes']['val']}")
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
