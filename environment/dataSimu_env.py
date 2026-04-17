import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset, random_split
from torchvision import transforms
from PIL import Image
import logging
from collections import defaultdict
from config.parameters import Config

logger = logging.getLogger(__name__)


class TransformSubset(Dataset):
    def __init__(self, dataset, indices, transform):
        self.dataset = dataset
        self.indices = indices
        self.transform = transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        image, label = self.dataset[self.indices[idx]]
        if self.transform:
            image = self.transform(image)
        return image, label


class ImageFolderDataset(Dataset):
    def __init__(self, data_path, transform=None):
        self.transform = transform
        self.data_path = data_path
        self.data = []
        self.labels = []
        self._load_data()

    def _load_data(self):
        classes = [d for d in os.listdir(self.data_path) if os.path.isdir(os.path.join(self.data_path, d))]
        classes.sort()
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        self.idx_to_class = {i: cls_name for cls_name, i in self.class_to_idx.items()}

        for class_name in classes:
            class_path = os.path.join(self.data_path, class_name)
            class_idx = self.class_to_idx[class_name]
            for img_file in os.listdir(class_path):
                if img_file.lower().endswith((".jpg", ".jpeg", ".png")):
                    self.data.append(os.path.join(class_path, img_file))
                    self.labels.append(class_idx)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = Image.open(self.data[idx]).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, self.labels[idx]

    def get_class_name(self, label_idx):
        return self.idx_to_class.get(label_idx, "Unknown")


class DomainIncrementalDataSimulator:
    """
    域增量数据模拟器

    =========================================================
    数据结构
    =========================================================
    每个域的数据划分：
    - 50% 初始数据 (init_domain_cache): 用于初始模型训练和奖励计算
    - 5×10% 子数据集 (sub_domain_cache): 用于持续学习训练

    缓存结构：
    - full_domain_cache:     {domain_key: original_dataset}     原始域数据
    - init_domain_cache:    {domain_key: {train/val/test}}     50%初始数据
    - sub_domain_cache:     {(domain_key, sub_idx): {train/val/test}}  5×10%子集
    - vehicle_data_assignments: {(domain_key, sub_idx): {vid: [indices]}}  车辆分配
    - class_distributions:  {class_idx: [ratio_per_vehicle]}    狄利克雷分布

    =========================================================
    核心流程
    =========================================================
    reset() → 清空所有缓存，重置session=0
         ↓
    update_session_dataset(session_id) → 域/子集切换
         ↓
    _preload_all_sub_domain_datasets() → 加载50%+5×10%数据
         ↓
    generate_vehicle_data(vid) → 分配车辆数据，生成batches
         ↓
    evaluate_model() → 用50%初始数据评估，计算奖励
         ↓
    get_cumulative_val/test_datasets() → 累积验证/测试集用于训练

    =========================================================
    域切换规则
    =========================================================
    - DOMAIN_CHANGE_INTERVAL: 域切换间隔
    - 每个域有5个子集，sub_idx在域内循环 [0, 1, 2, 3, 4]
    """

    NUM_CLASSES = {"office31": 31, "digit10": 10, "domainnet": 345}
    DOMAINS = {
        "office31": ["amazon", "webcam", "dslr"],
        "digit10": ["mnist", "emnist", "usps", "svhn"],
        "domainnet": ["clipart", "infograph", "painting", "quickdraw", "real", "sketch"],
    }
    BASE_PATH = {
        "office31": os.path.join(Config.DATA_BASE_PATH, "office-31"),
        "digit10": Config.SAMPLED_DATA_PATH if Config.USE_SAMPLED_DATA else os.path.join(Config.DATA_BASE_PATH, "digit10"),
        "domainnet": Config.SAMPLED_DATA_PATH if Config.USE_SAMPLED_DATA else os.path.join(Config.DATA_BASE_PATH, "domainnet"),
    }

    TRAIN_TRANSFORM = transforms.Compose([
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    TEST_TRANSFORM = transforms.Compose([
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    BASE_TRANSFORM = transforms.Compose([
        transforms.Resize((Config.IMAGE_SIZE, Config.IMAGE_SIZE)),
        transforms.Lambda(lambda x: x.convert("RGB") if x.mode != "RGB" else x),
        transforms.ToTensor(),
    ])

    def __init__(self):
        self.num_vehicles = Config.NUM_VEHICLES
        self.current_dataset = Config.CURRENT_DATASET
        self.current_domains = self.DOMAINS[self.current_dataset]
        self.reset()

    def reset(self):
        self.current_session = 0
        self.domain_sub_idx = {}
        self.seen_domains = []
        self.accuracy_history = {}
        self.aa_history = []
        self.sub_domain_cache = {}
        self.init_domain_cache = {}
        self.full_domain_cache = {}
        self.vehicle_data_assignments = {}
        self.class_distributions = self._init_dirichlet()
        logger.info("数据重置完成")

    def _init_dirichlet(self):
        distributions = {}
        num_classes = self.NUM_CLASSES[self.current_dataset]
        for vehicle_id in range(self.num_vehicles):
            alpha = np.full(num_classes, Config.DIRICHLET_ALPHA)
            distributions[vehicle_id] = np.random.dirichlet(alpha)
        return distributions

    def get_current_domain(self):
        domain_idx = (self.current_session // Config.DOMAIN_CHANGE_INTERVAL) % len(self.current_domains)
        return self.current_domains[domain_idx]

    def update_session_dataset(self, session_id):
        old_domain = self.get_current_domain()
        self.current_session = session_id
        new_domain = self.get_current_domain()

        if new_domain not in self.seen_domains:
            self.seen_domains.append(new_domain)
            logger.info(f"新域 {new_domain} 已加入已见域")

        domain_changed = (old_domain != new_domain)
        is_first_domain = (len(self.full_domain_cache) == 0)

        if domain_changed or is_first_domain:
            self.domain_sub_idx[new_domain] = 0
            self._preload_domain_data(new_domain)
        else:
            self.domain_sub_idx[new_domain] = (self.domain_sub_idx.get(new_domain, -1) + 1) % 5

        return domain_changed, old_domain, new_domain

    def _preload_domain_data(self, domain):
        domain_key = f"{self.current_dataset}_{domain}"

        if domain_key not in self.full_domain_cache:
            original_dataset = ImageFolderDataset(self.BASE_PATH[self.current_dataset] + "/" + domain, transform=self.BASE_TRANSFORM)
            if len(original_dataset) == 0:
                logger.warning(f"域 {domain} 的数据集为空")
                return
            self.full_domain_cache[domain_key] = original_dataset

        original_dataset = self.full_domain_cache[domain_key]

        if domain_key not in self.init_domain_cache:
            all_indices = list(range(len(original_dataset)))
            np.random.seed(Config.RANDOM_SEED)
            np.random.shuffle(all_indices)
            init_size = int(0.5 * len(original_dataset))
            init_indices = all_indices[:init_size]
            self.init_domain_cache[domain_key] = self._split_indices(original_dataset, init_indices, Config.RANDOM_SEED, "50%初始数据")

        if (domain_key, 0) not in self.sub_domain_cache:
            self._create_non_overlapping_subsets(original_dataset, domain_key)

        logger.info(f"已预加载域 {domain} 的数据")

    def _create_non_overlapping_subsets(self, original_dataset, domain_key):
        total_size = len(original_dataset)
        init_size = int(0.5 * total_size)

        all_indices = list(range(total_size))
        np.random.seed(Config.RANDOM_SEED + 100)
        np.random.shuffle(all_indices)

        init_indices = set(all_indices[:init_size])
        remaining_indices = [i for i in all_indices if i not in init_indices]

        subset_size = len(remaining_indices) // 5

        for sub_idx in range(5):
            sub_indices = remaining_indices[sub_idx * subset_size:(sub_idx + 1) * subset_size]
            self.sub_domain_cache[(domain_key, sub_idx)] = self._split_indices(original_dataset, sub_indices, Config.RANDOM_SEED + 100 + sub_idx, f"子集{sub_idx}")

    def _split_indices(self, original_dataset, indices, seed, name):
        indices = list(indices)
        generator = torch.Generator()
        generator.manual_seed(seed)

        test_size = int(Config.TEST_RATIO * len(indices))
        val_size = int(Config.VAL_RATIO * len(indices))
        train_size = len(indices) - test_size - val_size

        train_idx, val_idx, test_idx = random_split(indices, [train_size, val_size, test_size], generator=generator)

        return {
            "train": TransformSubset(original_dataset, train_idx.indices, self.TRAIN_TRANSFORM),
            "val": TransformSubset(original_dataset, val_idx.indices, self.TEST_TRANSFORM),
            "test": TransformSubset(original_dataset, test_idx.indices, self.TEST_TRANSFORM),
        }

    def _get_current_sub_domain_idx(self):
        return self.domain_sub_idx.get(self.get_current_domain(), 0)

    def _get_domain_key(self, domain=None):
        if domain is None:
            domain = self.get_current_domain()
        return f"{self.current_dataset}_{domain}"

    def _get_sub_key(self, domain=None):
        return (self._get_domain_key(domain), self._get_current_sub_domain_idx())

    def generate_vehicle_data(self, vehicle_id):
        current_domain = self.get_current_domain()
        sub_key = self._get_sub_key()

        if sub_key not in self.sub_domain_cache:
            logger.warning(f"子数据集 {sub_key} 未加载")
            return []

        train_dataset = self.sub_domain_cache[sub_key]["train"]
        vehicle_indices = self._get_vehicle_data_indices(vehicle_id)

        if not vehicle_indices:
            logger.warning(f"车辆 {vehicle_id} 在域 {current_domain} 子集 {self._get_current_sub_domain_idx()} 中无训练数据")
            return []

        vehicle_dataset = Subset(train_dataset, vehicle_indices)
        dataloader = DataLoader(vehicle_dataset, batch_size=Config.BATCH_SIZE, shuffle=False, drop_last=True)
        return list(dataloader)

    def _get_vehicle_data_indices(self, vehicle_id):
        sub_key = self._get_sub_key()
        if sub_key not in self.vehicle_data_assignments:
            self._assign_data_to_vehicles()
        return self.vehicle_data_assignments.get(sub_key, {}).get(vehicle_id, [])

    def _assign_data_to_vehicles(self):
        sub_key = self._get_sub_key()
        if sub_key not in self.sub_domain_cache:
            logger.warning(f"子数据集 {sub_key} 未加载，无法分配数据")
            return

        train_dataset = self.sub_domain_cache[sub_key]["train"]
        train_labels = [train_dataset[i][1] for i in range(len(train_dataset))]

        class_indices = defaultdict(list)
        for idx, label in enumerate(train_labels):
            class_indices[label].append(idx)

        for indices in class_indices.values():
            np.random.shuffle(indices)

        class_pointers = {c: 0 for c in class_indices.keys()}
        class_available = {c: len(indices) for c, indices in class_indices.items()}

        total_train_samples = len(train_labels)
        samples_per_vehicle = total_train_samples // self.num_vehicles
        all_classes = sorted(class_indices.keys())
        num_classes = len(all_classes)

        vehicle_assignments = {i: [] for i in range(self.num_vehicles)}

        for vehicle_id in range(self.num_vehicles):
            if vehicle_id not in self.class_distributions:
                self.class_distributions[vehicle_id] = np.random.dirichlet(
                    np.full(num_classes, Config.DIRICHLET_ALPHA))

            distribution = self.class_distributions[vehicle_id].copy()
            assigned_count = 0

            while assigned_count < samples_per_vehicle:
                available_classes = [c for c in all_classes if class_available.get(c, 0) > 0]
                if not available_classes:
                    break

                probs = np.array([distribution[all_classes.index(c)] for c in available_classes])
                probs = probs / probs.sum()

                selected_class = np.random.choice(available_classes, p=probs)
                vehicle_assignments[vehicle_id].append(class_indices[selected_class][class_pointers[selected_class]])
                class_pointers[selected_class] += 1
                class_available[selected_class] -= 1
                assigned_count += 1

        self.vehicle_data_assignments[sub_key] = vehicle_assignments

    def get_cumulative_test_datasets(self):
        datasets = {}
        for domain in self.seen_domains:
            domain_key = self._get_domain_key(domain)
            if domain_key in self.init_domain_cache:
                datasets[(domain, "init")] = self.init_domain_cache[domain_key]["test"]
            for sub_idx in range(5):
                sub_key = (domain_key, sub_idx)
                if sub_key in self.sub_domain_cache:
                    datasets[(domain, sub_idx)] = self.sub_domain_cache[sub_key]["test"]
        return datasets

    def get_cumulative_val_datasets(self):
        datasets = {}
        for domain in self.seen_domains:
            domain_key = self._get_domain_key(domain)
            if domain_key in self.init_domain_cache:
                datasets[(domain, "init")] = self.init_domain_cache[domain_key]["val"]
            for sub_idx in range(5):
                sub_key = (domain_key, sub_idx)
                if sub_key in self.sub_domain_cache:
                    datasets[(domain, sub_idx)] = self.sub_domain_cache[sub_key]["val"]
        return datasets


if __name__ == "__main__":
    print("1. 初始化数据模拟器...")
    data_simulator = DomainIncrementalDataSimulator()

    print("\n2. 域增量切换演示...")
    for session in range(5):
        print(f"\n--- Session {session} ---")
        data_simulator.update_session_dataset(session)
        current_domain = data_simulator.get_current_domain()
        print(f"当前域: {current_domain}, 子集: {data_simulator._get_current_sub_domain_idx()}")

        print("\n车辆数据分配示例:")
        for vehicle_id in range(min(3, data_simulator.num_vehicles)):
            vehicle_data = data_simulator.generate_vehicle_data(vehicle_id)
            if vehicle_data:
                total_samples = len(vehicle_data) * Config.BATCH_SIZE
                print(f"  车辆 {vehicle_id}: {total_samples} 个训练样本, {len(vehicle_data)} 个批次")
            else:
                print(f"  车辆 {vehicle_id}: 无训练数据")
