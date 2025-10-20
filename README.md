## 一、项目结构
```
continual_learning/
├── main.py
├── config/
│   ├── __init__.py
│   ├── parameters.py
│   └── paths.py
├── environment/
│   ├── __init__.py
│   ├── vehicle_env.py
│   ├── communication_env.py
│   └── data_simulator.py
├── models/
│   ├── __init__.py
│   ├── neural_networks.py
│   ├── mab_selector.py
│   └── drl_agent.py
├── learning/
│   ├── __init__.py
│   ├── continual_learner.py
│   ├── cache_manager.py
│   └── evaluator.py
├── utils/
│   ├── __init__.py
│   ├── data_loader.py
│   └── metrics.py
└── experiments/
    ├── __init__.py
    ├── baseline_comparison.py
    └── ablation_study.py
```

## data
### ./mnist/MNIST/raw
train-images-idx3-ubyte: 训练集图像数据
train-labels-idx1-ubyte: 训练集标签数据
t10k-images-idx3-ubyte: 测试集图像数据（t10k代表10000个测试样本）
t10k-labels-idx1-ubyte: 测试集标签数据
xx.gz是上述文件的压缩版本
### ./emnist/EMNIST/raw
emnist-digits-train-images-idx3-ubyte: 训练图像
emnist-digits-train-labels-idx1-ubyte: 训练标签
emnist-digits-test-images-idx3-ubyte: 测试图像
emnist-digits-test-labels-idx1-ubyte: 测试标签
### ./svhn
每个图像都是32x32像素的RGB图像
### ./usps
16×16像素的灰度图像
总计 9,298 个样本 (训练集: 7,291，测试集: 2,007)
## Environment
### 车辆环境（VehicleEnvironment）：
负责模拟车辆和基站的初始化和动态更新。
车辆具有位置、缓存数据、本地模型、置信度历史等属性，并能连接到最近的基站。
环境会更新车辆的位置，并重新连接基站。
车辆碰撞问题尚未解决。

### 通信系统（CommunicationSystem）：
负责计算通信速率和传输时延。
根据车辆与基站的距离、带宽分配等参数计算上行和下行速率。
计算数据上传和模型广播的时延。

### 数据分布模拟器（DataDistributionSimulator）：
负责生成模拟数据，并模拟数据分布的漂移。
通过不同的域来模拟分布变化。为车辆生成数据批次，并计算模型在特定域上的测试损失。