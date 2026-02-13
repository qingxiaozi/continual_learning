## 一、项目结构
```
├── README.md
├── config
│   ├── parameters.py
│   └── paths.py
├── data
│   ├── PortoTaxi
│   ├── digit10
│   ├── domainnet
│   └── office-31
├── environment
│   ├── communication_env.py
│   ├── dataSimu_env.py
│   └── vehicle_env.py
├── experiment
│   ├── rl_env.py
│   ├── rl_test.py
│   └── rl_train.py
├── learning
│   ├── batch_selector.py
│   ├── cache_manager.py
│   ├── continualLearner.py
│   ├── evaluator.py
│   ├── losses.py
│   ├── mab_selector.py
│   └── trainer.py
├── models
│   ├── bandwidth_allocator.py
│   ├── drl_agent.py
│   ├── global_model.py
│   └── gold_model.py
├── requirements.txt
├── results
│   ├── global_model_digit10.pth
│   ├── global_model_office31.pth
│   ├── golden_model_digit10.pth
│   └── golden_model_office31.pth
└── utils
    ├── data_deal.py
    └── metrics.py
```

## 二、总体目标
通过强化学习（RL agent）学习智能决策策略 π(a|s)，使系统在动态域漂移环境下，自适应优化车辆数据批次上传策略。而后，通过给定的上层决策，即上传数据批次量，进行下层目标的优化，即带宽的分配，从而最大化模型性能提升效率（单位时间损失下降）。
并抵抗灾难性遗忘，得到一个训练好的 DRL 策略网络（Policy Network）。而后在测试阶段复现不同的 domain drift 场景，验证策略泛化性能。
系统分为三层
基础层：环境仿真，构建车联网物理与数据漂移模型，封装于rl_env.py
策略学习层：强化学习，进行数据上传决策，封装于rl_train.py
评估层：性能度量，测试策略在不同场景下的表现，封装于rl_test.py

整体框架如下：
```python

Initialize RL environment (VehicleEdgeEnv)
Initialize DRL agent (DQN/PPO etc.)
For episode in range(NUM_EPISODES):
    state = env.reset()
    for step in range(NUM_SESSIONS):
        action = agent.select_action(state)
        next_state, reward, done, info = env.step(action)
        agent.store(state, action, reward, next_state, done)
        agent.optimize()
        state = next_state
        if done:
            break
Save trained model

for episode in TEST_EPISODES:
    reset env
    load trained policy
    for session in NUM_TRAINING_SESSIONS:
        action = policy(state)
        next_state, reward, done, info = env.step(action)
        record comm metrics
    evaluate CL metrics (AA, BWT, FM, AIA)
```

关于episode

单个episode可以模拟一次完整的车辆持续学习流程，经历多个step，每个step/每多个step对应车辆感知到的不同域;

在每个episode中的env.reset()中，车辆位置、模型状态、数据状态和缓存都会重新初始化;

每个episode训练的目标是学会如何面对随时间变化的域漂移序列;

训练阶段的所有eposide都使用同一个数据集，每个step或者多个step对应不同的domain;

当域序列走完，或者step步达到上限，或者奖励变化小于阈值，则判定eposide结束。

关于step

每个step是一个完整的持续学习训练阶段，包括数据决策、带宽分配优化、数据上传、缓存更新、全局模型训练、奖励计算、状态更新，判断是否结束；

当一个训练阶段结束，则判定step结束。而判断“一个训练阶段结束”，可通过当前阶段达到固定的epoch数、当前step在episode中的索引达到上限等确定。

关于测试指标

持续学习相关的指标，需要在task/domain级别进行计算，即在domain结束时计算。

## 三、评价指标
1. 持续学习的质量
AA、AIA、FM、BWT
1. 系统与通信指标
每个episode下的reward，平均reward
每个episode下的通信时延，平均时延

## 四、数据集
|数据集|域|类别数|类别|样本数|图像大小|通道数|说明|
|------|------|------|------|------|------|------|------|
|office-31|Amazon、Webcam、DSLR|31|back_pack、bottle、desktop_computer、laptop_computer、mouse、phone、ring_binder、stapler、bike、calculator、file_cabinet、letter_tray、mug、printer、ruler、tape_dispenser、bike_helmet、desk_chair、headphones、mobile_phone、paper_notebook、projector、scissors、trash_can、bookcase、desk_lamp、keyboard、monitor、pen、punchers、speaker|Amazon:2817；dslr:498；webcam:795|图像大小|通道数|说明|
|digit10|MNIST、EMNIST、USPS、SVHN|10|0、1、2、3、4、5、6、7、8、9|MNIST：70000；EMNIST：280000；USPS：7291；SVHN：99289|图像大小|通道数|数字识别|
|DomainNet|Clipart、Infograph、Painting、Quickdraw、Real、Sketch|类别数|类别|Real: 约 172,000 张；Clipart: 约 48,000 张；Painting: 约 72,000 张；Quickdraw: 约 172,000 张；Sketch: 约 72,000 张；Infograph: 约 48,000 张|图像大小|通道数|说明|
<br>

## 五、结果
1. accuracy_matrices.npy：per-episode n*(1~n)，三角准确率估计
2. xx_steps.npy：[episodes,n]，过程分析
3. xx_all.npy：[episodes]，每个eposide的最终指标
4. episode_rewards.npy
5. episode_delays.npy
   
### environment
#### 车辆环境（vehicle_env）
1. 初始化车辆和基站环境。
2. 建立基站，确定其位置、覆盖范围、最大连接车辆数等属性；
3. 建立车辆集群，确定其初始位置，将车辆与距离其最近的基站连接；
4. 更新车辆位置，车辆以速度8-20 m/s(29-72 km/h)行驶，可确定其在任意时间点的位置；
5. 重置环境至初始状态。
6. 遗留问题：车辆行驶过程中可能碰撞

#### 通信环境（communication_env）
1. 计算上下行通信速率
2. 计算数据传输时延 (t_trans)
3. 计算数据标注时延 (t_label)
4. 计算模型重训练时延 (t_retrain)
5. 计算模型广播时延 (t_broadcast)
6. 遗留问题：传输时延计算单位需要确认，总带宽为MHz，样本大小为bit

#### 数据模拟（dataSimu_env）
功能：<br>
分别处理office-31、digit10、DomainNet三个数据集，将其按照Non-IID（狄利克雷分布）的方式划分给多个智能车辆，并支持增量学习。<br>
步骤：<br>
1. 支持三个数据集的读取，并进行数据预处理，使得样本的大小与通道数统一。大小：224x224；通道数：3。
2. 按照域顺序依次加载，第一个域作为初始数据，后续域作为增量数据。
3. 使用狄利克雷分布将每个域的数据非独立同分布的划分给多个车辆。
4. 每个车辆维护自己的数据缓存，并能够按照批次提供数据。

### models
#### 多臂老虎机（mab_selector）
功能：<br>
用于选择高质量的数据批次。<br>
1. 使用ucb算法选择所有臂（数据批次）中价值最高的臂。
2. 更新臂的统计，包括被选择的次数、累积奖励和平均奖励。
3. 计算一个数据批次在当前模型下的损失，并以该损失的负值作为该数据批次的奖励。注意：并未更新模型。
4. 计算所有批次的归一化质量评分。
5. MAB状态重置，即被选择的此时、累积奖励、平均奖励均为0。


#### 强化学习智能体（drl_agent）
1. 选择动作。以一定概率生成随机动作，否则利用策略网络选择动作。
2. 将数据标注量动作通过线性变换映射到[0, MAX_UPLOAD_BATCHES]的整数，将带宽分配进行归一化，使得总和为1。
3. 模型优化。从回放缓冲区随机采样一个批次的经验，使用策略网络计算当前状态-动作对的Q值，使用目标网络计算下一个状态最大的Q值，根据贝尔曼方程计算目标Q值。使用均方误差损失函数计算当前Q值和目标Q值之间的差值，并通过优化器更新策略网络。
4. 保存和加载模型，包括策略网络、目标网络和优化器的状态字典。


#### 接下来要做的
1. 通信参数找其他论文确认
2. 进行基线对比实验 10days
3. 消融实验 5days
4. 完善digit和dominNet数据集训练 10days


