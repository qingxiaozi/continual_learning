import os
import shutil
import random
from config.parameters import Config

# 设置随机种子
random.seed(Config.RANDOM_SEED)

# 数据集路径
SOURCE_BASE = "./data/digit10"
TARGET_BASE = "./data/digit10_sample"

# 每个域采样数量
SAMPLES_PER_DOMAIN = 10350

# digit10 的域
# DOMAINS = [
#             "clipart",
#             "infograph",
#             "painting",
#             "quickdraw",
#             "real",
#             "sketch",
#         ]
DOMAINS = ["mnist", "emnist", "usps", "svhn"]

def sample_domain_data(domain_name):
    """从一个域中采样数据"""
    source_dir = os.path.join(SOURCE_BASE, domain_name)
    target_dir = os.path.join(TARGET_BASE, domain_name)
    
    if not os.path.exists(source_dir):
        print(f"警告: 域 {domain_name} 的源目录不存在: {source_dir}")
        return
    
    # 创建目标目录
    os.makedirs(target_dir, exist_ok=True)
    
    # 获取所有类别
    classes = [d for d in os.listdir(source_dir) if os.path.isdir(os.path.join(source_dir, d))]
    classes.sort()
    
    print(f"\n处理域: {domain_name}")
    print(f"  类别数: {len(classes)}")
    
    # 计算每个类别应采样的数量（均匀采样）
    samples_per_class = SAMPLES_PER_DOMAIN // len(classes)
    remainder = SAMPLES_PER_DOMAIN % len(classes)
    
    total_sampled = 0
    
    for class_idx, class_name in enumerate(classes):
        class_source = os.path.join(source_dir, class_name)
        class_target = os.path.join(target_dir, class_name)
        
        # 获取该类别下所有图片
        images = [f for f in os.listdir(class_source) 
                  if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        # 随机采样
        if len(images) <= samples_per_class + (1 if class_idx < remainder else 0):
            sampled_images = images
        else:
            # 前 remainder 个类别多采一个
            n_sample = samples_per_class + (1 if class_idx < remainder else 0)
            sampled_images = random.sample(images, n_sample)
        
        # 创建目标类别目录
        os.makedirs(class_target, exist_ok=True)
        
        # 复制图片
        for img in sampled_images:
            src = os.path.join(class_source, img)
            dst = os.path.join(class_target, img)
            shutil.copy2(src, dst)
        
        total_sampled += len(sampled_images)
        print(f"  类别 {class_name}: 采样 {len(sampled_images)} / {len(images)}")
    
    print(f"  总计: 采样 {total_sampled} 张图片")

def main():
    print("=" * 50)
    print("digit10 数据集采样脚本")
    print("=" * 50)
    print(f"源目录: {SOURCE_BASE}")
    print(f"目标目录: {TARGET_BASE}")
    print(f"每个域采样数: {SAMPLES_PER_DOMAIN}")
    print(f"域: {DOMAINS}")
    
    # 检查源目录
    if not os.path.exists(SOURCE_BASE):
        print(f"错误: 源目录不存在: {SOURCE_BASE}")
        print("请先下载 digit10 数据集")
        return
    
    # 为每个域采样
    for domain in DOMAINS:
        sample_domain_data(domain)
    
    print("\n" + "=" * 50)
    print("采样完成!")
    print("=" * 50)

if __name__ == "__main__":
    main()
