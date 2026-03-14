import torch
import random


class Config:
    RANDOM_SEED = 42  # 随机种子，用于可复现实验
    
    BANDWIDTH_STRATEGY = "MINMAX_DELAY"  # ["EQUAL", "GREEDY_CHANNEL", "MINMAX_DELAY"]
    UPLOAD_STRATEGY = "DRL"              # ["STATIC", "FIXED_RATIO", "LOSS_GREEDY", "DRL"]
    TRAINING_STRATEGY = "MAB"            # ["NEW_ONLY", "FIXED_RATIO", "MAB"]
    FIXED_RATIO = 0.5  # 固定比例策略中使用的新数据比例
    # 实验参数
    NUM_EPISODES = 837  # episode数，根据轨迹数量设置
    NUM_TRAINING_SESSIONS = 21  # train session
    TARGET_UPDATE_INTERVAL = 5  # 每5个episode硬更新目标网络
    NUM_TEST_EPISODES = 200  # 200
    NUM_TESTING_SESSIONS = 21

    NUM_VEHICLES = 20  # 车辆数
    # 模型参数
    NUM_EPOCH = 10  # 训练epoch数
    BATCH_SIZE = 32  # batch size
    LEARNING_RATE = 0.001  # 学习率

    # MAB参数
    INIT_EPOCHS = 3  # MAB初始探索轮次
    MAB_EXPLORATION_FACTOR = 2.0  # UCB探索因子
    
    # 通信参数
    BASE_STATION_COVERAGE = 800  # 米，基站覆盖范围（增加到10km以覆盖更多区域）
    BASE_BANDWIDTH = 20e6  # Hz，基础带宽，限制数据上传的速度
    BS_TRANSMIT_POWER = 20  # W，基站发射功率，影响通信距离和质量（W）
    VEHICLE_TRANSMIT_POWER = 1  # 车辆发射功率（W）
    NOISE_POWER = 3.981e-14  # W，噪声功率，决定通信质量的下限
    PATH_LOSS_EXPONENT = 2.7  # 路径损耗指数
    SHADOWING_STD = 8  # 阴影衰落标准差
    INTERFERENCE_POWER = 1e-11  # 干扰功率
    REFERENCE_GAIN = 1e-3  # 参考距离（1m)处的路径增益
    GOLDEN_MODEL_CYCLES = 2e6  # 黄金模型处理一个样本的计算周期数
    GLOBAL_MODEL_CYCLES = 5e6  # 全局模型处理一个样本的计算周期数
    EDGE_COMPUTE_CAPACITY = 2e10  # 边缘服务器计算能力（Cycles/s），20GHz

    # DRL参数
    DRL_HIDDEN_SIZE = 128  # 神经网络隐藏层的大小
    DRL_LEARNING_RATE = 0.0001  # 优化器的学习率
    DRL_GAMMA = 0.99  # 折扣因子
    DRL_BUFFER_SIZE = 10000  # 经验回放缓冲区大小
    DRL_BATCH_SIZE = 32  # batch size
    DRL_EPSILON_START = 0.9
    DRL_EPSILON_END = 0.05
    DRL_EPSILON_DECAY = 80  # epsilon衰减速率
    DRL_TARGET_UPDATE_EVERY = 100  # 目标网络更新频率
    DRL_TAU = 0.005  # 软更新参数

    # 缓存参数
    MAX_CACHE_SIZE = 1000  # 边缘服务器的最大缓存批次
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
    DOMAIN_CHANGE_INTERVAL = 7  # 每x个session切换一个域
    # 数据加载参数
    IMAGE_SIZE = 32  # 统一调整到相同尺寸
    # 测试策略参数
    TEST_STRATEGY = "cumulative"  # 'current' 或 'cumulative'
    TEST_RATIO = 0.2  # 测试集比例
    VAL_RATIO = 0.2  # 验证集比例

    # 设备
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 车辆环境参数
    PPP_RADIUS = 200  # PPP生成半径（米）
    PPP_LAMBDA = 0.001  # 单位面积车辆密度（辆/平方米）
    PPP_LAMBDA_BS = 3  # 单位面积基站密度（个/平方公里）
    MIN_BS_DISTANCE = 500.0  # 宏基站最小间距（米）
    VEHICLE_SPEED_FACTOR = 20.0  # 车辆速度因子（m/s），用于计算移动距离

    @staticmethod
    def set_seed(seed=None):
        """设置全局种子以保证实验可复现"""
        if seed is None:
            seed = Config.RANDOM_SEED
        random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
