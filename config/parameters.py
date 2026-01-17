import torch


class Config:
    # 实验参数
    NUM_EPISODES = 100
    NUM_TRAINING_SESSIONS = 60  # train session
    TARGET_UPDATE_INTERVAL = 10  # 每10个episode硬更新目标网络

    NUM_VEHICLES = 20  # 车辆数
    NUM_EPOCH = 60  # 训练epoch数
    BATCH_SIZE = 16  # batch size
    INIT_EPOCHS = 5  # MAB初始探索轮次

    # 通信参数
    BASE_STATION_COVERAGE = 8000  # 米，基站覆盖范围
    BASE_BANDWIDTH = 20e6  # Hz，基础带宽，限制数据上传的速度
    BS_TRANSMIT_POWER = 20  # W，基站发射功率，影响通信距离和质量（W）
    VEHICLE_TRANSMIT_POWER = 1  # 车辆发射功率（W）
    NOISE_POWER = 3.981e-14  # W，噪声功率，决定通信质量的下限
    PATH_LOSS_EXPONENT = 2.7  # 路径损耗指数
    SHADOWING_STD = 8  # 阴影衰落标准差
    INTERFERENCE_POWER = 1e-11  # 干扰功率
    REFERENCE_GAIN = 1e-3  # 参考距离（1m)处的路径增益

    TRAJECTORY_FILE = "trajectory_12213.csv"

    # 模型参数
    NUM_CLASSES = 10  # 类别的数量
    GOLD_MODEL_ACCURACY = 0.95  # 黄金模型准确率95%
    LEARNING_RATE = 0.001  # 学习率

    # MAB参数
    MAB_EXPLORATION_FACTOR = 2.0  # UCB探索因子
    MAB_MIN_SAMPLES = 5  # 最小样本数

    # DRL参数
    DRL_HIDDEN_SIZE = 128  # 神经网络隐藏层的大小
    DRL_LEARNING_RATE = 0.001  # 优化器的学习率
    DRL_GAMMA = 0.99  # 折扣因子
    DRL_BUFFER_SIZE = 10000  # 经验回放缓冲区大小
    DRL_BATCH_SIZE = 8  # batch size
    DRL_EPSILON_START = 0.9
    DRL_EPSILON_END = 0.05
    DRL_EPSILON_DECAY = 80  # epsilon衰减速率

    # 缓存参数
    MAX_CACHE_SIZE = 1000  # 边缘服务器的最大缓存批次
    MIN_CACHE_SIZE = 100  # 边缘服务器的最小缓存批次
    MAX_UPLOAD_BATCHES = 5  # 单阶段最大上传批次
    MAX_LOCAL_BATCHES = 5  # 单个智能车最大缓存批次

    # 数据分布参数
    DATASET_NAMES = ["digit10", "office31", "domainnet"]
    CURRENT_DATASET = "office31"
    # 域增量学习参数
    DOMAIN_SEQUENCES = {
        "office31": ["amazon", "webcam", "dslr"],
        "digit10": ["mnist", "emnist", "usps", "svhn"],
        "domainnet": [
            "clipart",
            "infograph",
            "painting",
            "quickdraw",
            "real",
            "sketch",
        ],
    }
    # 狄利克雷分布参数（控制数据异构程度）
    DIRICHLET_ALPHA = 0.5  # α越小，数据分布越异构
    # 数据路径
    DATA_BASE_PATH = "./data"
    # 数据集特定参数
    OFFICE31_CLASSES = 31
    DIGIT10_CLASSES = 10
    DOMAINNET_CLASSES = 345
    # 域切换参数
    DOMAIN_CHANGE_INTERVAL = 1  # 每1个session切换一个域
    # 数据加载参数
    IMAGE_SIZE = 32  # 统一调整到相同尺寸
    # 测试策略参数
    TEST_STRATEGY = "cumulative"  # 'current' 或 'cumulative'
    TEST_RATIO = 0.2  # 测试集比例
    VAL_RATIO = 0.2  # 验证集比例

    # 评估参数
    EVALUATION_FREQUENCY = 5  # 每5个session评估一次

    # 设备
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
