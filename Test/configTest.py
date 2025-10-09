import sys
import os
import torch

# 添加项目根目录到Python路径，以便能够导入config模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.parameters import Config
from config.paths import Paths

def test_config_basic():
    """测试配置类的基本功能"""
    print("=== 配置参数基础测试 ===")

    # 测试配置类实例化
    try:
        config = Config()
        print("✓ 配置类实例化成功")
    except Exception as e:
        print(f"✗ 配置类实例化失败: {e}")
        return

    # 测试参数访问
    try:
        print(f"✓ 车辆数: {config.NUM_VEHICLES}")
        print(f"✓ 设备类型: {config.DEVICE}")
        print(f"✓ 隐藏层大小: {config.DRL_HIDDEN_SIZE}")
        print(f"✓ 学习率: {config.DRL_LEARNING_RATE}")
        print("✓ 所有参数访问正常")
    except AttributeError as e:
        print(f"✗ 参数访问错误: {e}")
        return

    # 测试数据类型
    assert isinstance(config.NUM_VEHICLES, int), "NUM_VEHICLES应为整数"
    assert isinstance(config.DRL_LEARNING_RATE, float), "DRL_LEARNING_RATE应为浮点数"
    assert isinstance(config.DEVICE, torch.device), "DEVICE应为torch.device类型"
    print("✓ 数据类型检查通过")


def test_config_validation():
    """测试配置参数的合理性"""
    print("\n=== 配置参数验证测试 ===")
    config = Config()

    # 数值范围检查
    checks = [
        (config.NUM_VEHICLES > 0, "车辆数应大于0"),
        (0 < config.DRL_LEARNING_RATE < 1, "学习率应在0-1之间"),
        (0 < config.DRL_GAMMA <= 1, "折扣因子应在0-1之间"),
        (config.DRL_BATCH_SIZE > 0, "批次大小应大于0"),
        (config.DRL_BUFFER_SIZE > config.DRL_BATCH_SIZE, "缓冲区应大于批次大小"),
        (config.BASE_STATION_COVERAGE > 0, "基站覆盖范围应大于0"),
        (config.UPLINK_BANDWIDTH > 0, "上行带宽应大于0"),
        (config.DOWNLINK_BANDWIDTH > 0, "下行带宽应大于0"),
    ]

    all_passed = True
    for condition, message in checks:
        if condition:
            print(f"✓ {message}")
        else:
            print(f"✗ {message}")
            all_passed = False

    return all_passed


def test_device_compatibility():
    """测试设备兼容性"""
    print("\n=== 设备兼容性测试 ===")
    config = Config()

    print(f"当前设备: {config.DEVICE}")

    if config.DEVICE.type == "cuda":
        print("✓ 使用GPU加速")
        print(f"GPU设备: {torch.cuda.get_device_name()}")
        print(f"GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("ℹ 使用CPU，建议在支持CUDA的环境中使用GPU")

    # 测试PyTorch基本功能
    try:
        test_tensor = torch.randn(10, 10).to(config.DEVICE)
        result = test_tensor @ test_tensor.T
        print("✓ PyTorch张量运算正常")
    except Exception as e:
        print(f"✗ PyTorch运算测试失败: {e}")


def test_paths_basic():
    """测试路径类的基本功能"""
    print("=== 配置路径基础测试 ===")

    # 测试路径类实例化
    try:
        paths = Paths()
        print("✓ 路径类实例化成功")
    except Exception as e:
        print(f"✗ 路径类实例化失败: {e}")
        return

    # 测试参数访问
    try:
        print(f"✓ BASE_DIR: {paths.BASE_DIR}")
        print(f"✓ DATA_DIR: {paths.DATA_DIR}")
        print(f"✓ MODELS_DIR: {paths.MODELS_DIR}")
        print(f"✓ RESULTS_DIR: {paths.RESULTS_DIR}")
        print("✓ 所有参数访问正常")
    except AttributeError as e:
        print(f"✗ 参数访问错误: {e}")
        return

    # 测试数据类型
    assert isinstance(paths.BASE_DIR, str), "NUM_VEHICLES应为整数"
    assert isinstance(paths.DATA_DIR, str), "DRL_LEARNING_RATE应为浮点数"
    assert isinstance(paths.MODELS_DIR, str), "DEVICE应为torch.device类型"
    assert isinstance(paths.RESULTS_DIR, str), "DEVICE应为torch.device类型"
    print("✓ 数据类型检查通过")


def run_complete_test():
    """运行完整测试套件"""
    print("开始配置脚本测试...\n")

    # 运行所有测试
    test_config_basic()
    validation_passed = test_config_validation()
    test_device_compatibility()
    test_paths_basic()

    print("\n" + "="*50)
    print("测试总结:")
    if validation_passed:
        print("✓ 所有测试通过！配置脚本可以正常使用。")
    else:
        print("⚠ 部分测试未通过，请检查配置参数。")

    # 显示关键配置信息
    config = Config()
    print("\n关键配置信息:")
    print(f"- 训练设备: {config.DEVICE}")
    print(f"- 车辆数量: {config.NUM_VEHICLES}")
    print(f"- 训练轮次: {config.NUM_EPOCH}")
    print(f"- 批次大小: {config.BATCH_SIZE}")
    print(f"- 数据集: {config.CURRENT_DATASET}")


if __name__ == "__main__":
    run_complete_test()