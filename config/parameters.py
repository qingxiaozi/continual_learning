import torch


class Config:
    # 实验参数
    NUM_VEHICLES = 20  # 车辆数
    NUM_TRAINING_SESSIONS = 100  # train session
    NUM_EPOCH = 100  # 训练epoch数
    BATCH_SIZE = 32  # batch size
    INIT_EPOCHS = 10  # MAB初始探索轮次

    # 通信参数
    BASE_STATION_COVERAGE = 1000  # 米，基站覆盖范围
    BASE_BANDWIDTH = 20e6  # Hz，基础带宽，限制数据上传的速度
    BS_TRANSMIT_POWER = 10  # W，基站发射功率，影响通信距离和质量（W）
    VEHICLE_TRANSMIT_POWER = 0.1  # 车辆发射功率（W）
    NOISE_POWER = 1e-10  # W，噪声功率，决定通信质量的下限
    PATH_LOSS_EXPONENT = 2.7  # 路径损耗指数
    SHADOWING_STD = 8  # 阴影衰落标准差
    INTERFERENCE_POWER = 1e-11  # 干扰功率
    REFERENCE_GAIN = 1e-3  # 参考距离（1m)处的路径增益

    # 模型参数
    MODEL_INPUT_SIZE = 224  # 模型输入尺寸的大小
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
    DRL_BATCH_SIZE = 32  # batch size

    # 缓存参数
    MAX_CACHE_SIZE = 1000
    MIN_CACHE_SIZE = 100
    MAX_UPLOAD_BATCHES = 5  # 单阶段最大上传批次
    MAX_LOCAL_BATCHES = 100  # 单个智能车最大缓存批次
    MAX_CONFIDENCE_HISTORY = 7  # 单个智能车最大缓存置信度个数
    SAMPLES_OF_BATCH = 64  # 每个batch中的样本数

    # 数据集参数
    DATASET_NAMES = ["digit10", "office31", "domainnet"]
    CURRENT_DATASET = "digit10"

    # 设备
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


config = Config()
