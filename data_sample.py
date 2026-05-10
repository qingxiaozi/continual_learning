import os
import shutil
import random
import argparse
from config.parameters import Config

random.seed(Config.RANDOM_SEED)

DATASETS = {
    "digit10": {
        "source": "./data/digit10",
        "target": "./data/digit10_sample",
        "domains": ["mnist", "emnist", "usps", "svhn"],
    },
    "domainnet": {
        "source": "./data/domainnet",
        "target": "./data/domainnet_sample",
        "domains": ["clipart", "infograph", "painting", "quickdraw", "real", "sketch"],
    },
}

SAMPLE_RATIO = 0.1

def get_total_images(source_dir):
    if not os.path.exists(source_dir):
        return 0
    total = 0
    for class_name in os.listdir(source_dir):
        class_path = os.path.join(source_dir, class_name)
        if os.path.isdir(class_path):
            images = [f for f in os.listdir(class_path)
                      if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            total += len(images)
    return total

def sample_domain_data(source_base, target_base, domain_name):
    source_dir = os.path.join(source_base, domain_name)
    target_dir = os.path.join(target_base, domain_name)

    if not os.path.exists(source_dir):
        print(f"警告: 域 {domain_name} 的源目录不存在: {source_dir}")
        return 0

    total_original = get_total_images(source_dir)
    samples_to_take = int(total_original * SAMPLE_RATIO)

    print(f"\n处理域: {domain_name}")
    print(f"  原始图片数: {total_original}")
    print(f"  采样数量 (1/10): {samples_to_take}")

    os.makedirs(target_dir, exist_ok=True)

    classes = [d for d in os.listdir(source_dir) if os.path.isdir(os.path.join(source_dir, d))]
    classes.sort()

    print(f"  类别数: {len(classes)}")

    samples_per_class = samples_to_take // len(classes)
    remainder = samples_to_take % len(classes)

    total_sampled = 0

    for class_idx, class_name in enumerate(classes):
        class_source = os.path.join(source_dir, class_name)
        class_target = os.path.join(target_dir, class_name)

        images = [f for f in os.listdir(class_source)
                  if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

        n_sample = samples_per_class + (1 if class_idx < remainder else 0)

        if len(images) <= n_sample:
            sampled_images = images
        else:
            sampled_images = random.sample(images, n_sample)

        os.makedirs(class_target, exist_ok=True)

        for img in sampled_images:
            src = os.path.join(class_source, img)
            dst = os.path.join(class_target, img)
            shutil.copy2(src, dst)

        total_sampled += len(sampled_images)
        print(f"  类别 {class_name}: 采样 {len(sampled_images)} / {len(images)}")

    print(f"  总计: 采样 {total_sampled} 张图片")
    return total_sampled

def main():
    parser = argparse.ArgumentParser(description="数据集采样脚本")
    parser.add_argument("--dataset", type=str, required=True, choices=["digit10", "domainnet"],
                        help="选择数据集: digit10 或 domainnet")
    args = parser.parse_args()

    dataset_config = DATASETS[args.dataset]
    source_base = dataset_config["source"]
    target_base = dataset_config["target"]
    domains = dataset_config["domains"]

    print("=" * 50)
    print(f"{args.dataset.upper()} 数据集采样脚本 (1/10)")
    print("=" * 50)
    print(f"源目录: {source_base}")
    print(f"目标目录: {target_base}")
    print(f"采样比例: {SAMPLE_RATIO * 100}%")
    print(f"域: {domains}")

    if not os.path.exists(source_base):
        print(f"错误: 源目录不存在: {source_base}")
        return

    total_sampled = 0
    for domain in domains:
        total_sampled += sample_domain_data(source_base, target_base, domain)

    print("\n" + "=" * 50)
    print(f"采样完成! 总计采样: {total_sampled} 张图片")
    print("=" * 50)

if __name__ == "__main__":
    main()
