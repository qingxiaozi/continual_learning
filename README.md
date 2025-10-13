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
## Environment
### 车辆环境（VehicleEnvironment）：
负责模拟车辆和基站的初始化和动态更新。
车辆（Vehicle）具有位置、缓存数据、本地模型、置信度历史等属性，并能连接到最近的基站。环境会更新车辆的位置，并重新连接基站。

### 通信系统（CommunicationSystem）：
负责计算通信速率和传输时延。
根据车辆与基站的距离、带宽分配等参数计算上行和下行速率。计算数据上传和模型广播的时延。

### 数据分布模拟器（DataDistributionSimulator）：
负责生成模拟数据，并模拟数据分布的漂移。
通过不同的域来模拟分布变化。为车辆生成数据批次，并计算模型在特定域上的测试损失。