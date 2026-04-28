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
│   ├── png
│   ├── npy
│   └── pth
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
Initialize DRL agent (DQN)
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
    for session in NUM_TESTING_SESSIONS:
        action = policy(state)
        next_state, reward, done, info = env.step(action)
        record comm metrics
    evaluate CL metrics (AA, BWT, FM, AIA)
```

1. 关于episode
单个episode可以模拟一次完整的车辆持续学习流程，经历多个step，每个step/每多个step对应车辆感知到的不同域;<br>
在每个episode中的env.reset()中，车辆位置、模型状态、数据状态和缓存都会重新初始化;<br>
每个episode训练的目标是学会如何面对随时间变化的域漂移序列;<br>
训练阶段的所有eposide都使用同一个数据集，每个step或者多个step对应不同的domain;<br>
当域序列走完，或者step步达到上限，或者奖励变化小于阈值，则判定eposide结束。<br>
2. 关于step
每个step是一个完整的持续学习训练阶段，包括数据决策、带宽分配优化、数据上传、缓存更新、全局模型训练、奖励计算、状态更新，判断是否结束；<br>
当一个训练阶段结束，则判定step结束。而判断“一个训练阶段结束”，可通过当前阶段达到固定的epoch数、当前step在episode中的索引达到上限等确定。<br>
3. 关于测试指标
持续学习相关的指标，需要在task/domain级别进行计算，即在domain结束时计算。<br>

## 三、评价指标

### 1. 持续学习的质量
- **AA (Average Accuracy)**：所有任务/域上的平均准确率
- **AIA (Average Incremental Accuracy)**：AA + BWT，反映整体持续学习性能
- **FM (Forward Transfer)**：正向迁移能力，衡量新任务对旧任务的帮助
- **BWT (Backward Transfer)**：反向迁移能力（灾难性遗忘），衡量旧任务对新任务的影响

### 2. 系统与通信指标
- **Reward**：每个episode下的奖励，反映模型性能提升与通信成本的权衡
- **Delay**：每个episode下的通信时延

### 3. MDP状态定义（论文）
根据论文MDP定义，状态s包含各车辆v的本地状态xs_v = [Cs_v, Ls_v, ds_v]：
- **Cs_v**：车辆v对新收集数据的平均推理置信度
- **Ls_v**：学习反馈分数，Ls_v = 1 / (1 + L(ωs-1g, Nsv))，L为损失函数
- **ds_v**：车辆到最近基站的归一化距离

### 4. 奖励函数设计
奖励函数公式：
```
reward = delta_acc - 2.0 * forgetting - 0.1 * delay
```
- **delta_acc**：准确率变化，当前测试集准确率与上一轮的差值
- **forgetting**：灾难性遗忘，衡量之前任务准确率的下降程度
- **delay**：通信时延

## 四、数据集
|数据集|域|类别数|类别|样本数|图像大小|通道数|说明|
|------|------|------|------|------|------|------|------|
|office-31|Amazon、Webcam、DSLR|31|back_pack、bottle、desktop_computer、laptop_computer、mouse、phone、ring_binder、stapler、bike、calculator、file_cabinet、letter_tray、mug、printer、ruler、tape_dispenser、bike_helmet、desk_chair、headphones、mobile_phone、paper_notebook、projector、scissors、trash_can、bookcase、desk_lamp、keyboard、monitor、pen、punchers、speaker|Amazon:2817；dslr:498；webcam:795 (总计:4,110)|224×224|3|域适应基准数据集|
|digit10|MNIST、EMNIST、USPS、SVHN|10|0、1、2、3、4、5、6、7、8、9|MNIST:70,000；EMNIST:280,000；USPS:7,291；SVHN:99,289 (总计:456,580)|224×224|3|数字识别数据集|
|DomainNet|Clipart、Infograph、Painting、Quickdraw、Real、Sketch|345|clipart、infograph、painting、quickdraw、real、sketch|Real:175,327；Quickdraw:172,500；Painting:75,759；Sketch:70,386；Infograph:53,201；Clipart:48,833 (总计:596,006)|224×224|3|大规模域适应数据集|
<br>

## 五、结果
1. accuracy_matrices.npy：per-episode n*(1~n)，三角准确率估计
2. xx_steps.npy：[episodes,n]，过程分析
3. xx_all.npy：[episodes]，每个eposide的最终指标
4. episode_rewards.npy
5. episode_delays.npy

## 六、对比实验
1. 带宽分配策略 
EQUAL (均匀分配)：所有上传车辆平分总带宽。<br>
GREEDY (贪婪信道分配)：将更多带宽分配给信道条件（SNR）更好的车辆。<br>
MINMAX (最小化最大传输时延)<br>

2. 上传决策策略
RATIO (固定比例)：每辆车上传其缓存容量的固定比例。<br>
DRL (深度强化学习)：通过DRL智能体，基于全局状态做出联合优化决策。<br>

3. 数据保留策略
FIFO (先进先出)：将新数据与旧数据按顺序混合，如果新旧数据超出缓存容量，则按照FIFO原则删除数据，使其满足缓存容量限制。训练时使用缓存中的所有数据。<br>
MAB (多臂老虎机)：在训练过程中动态评估每个数据批次的价值，下一阶段中，上传新数据，根据数据批次的价值去除超出缓存容量的旧数据。训练时使用缓存中的所有数据。<br>

1. 对比实验设计
Naive_Baseline：带宽平均分配，数据按照固定比例上传，训练方式采用固定比例训练。<br>
Proportional_BW：带宽按照比例分配，数据按照固定比例上传，训练方式采用固定比例训练。<br>
GREEDY_BW：带宽按照贪婪策略分配，数据按照固定比例上传，训练方式采用固定比例训练。<br>

### environment
#### 车辆环境（vehicle_env）
1. 初始化车辆和基站环境。
2. 建立基站，确定其位置、覆盖范围、最大连接车辆数等属性；
3. 建立车辆集群，确定其初始位置，将车辆与距离其最近的基站连接；
4. 更新车辆位置，车辆以速度20 m/s(72 km/h)行驶，可确定其在任意时间点的位置；
5. 重置环境至初始状态。

#### 通信环境（communication_env）
1. 计算上下行通信速率
2. 计算数据传输时延 (t_trans)
3. 计算数据标注时延 (t_label)
4. 计算模型重训练时延 (t_retrain)
5. 计算模型广播时延 (t_broadcast)

#### 数据模拟（dataSimu_env）
功能：<br>
分别处理office-31、digit10、DomainNet三个数据集，将其按照Non-IID（狄利克雷分布）的方式划分给多个智能车辆，并支持增量学习。<br>
数据分配逻辑：<br>
1. 每个域的数据划分为两部分：50%初始数据 + 5×10%子数据集<br>
2. 50%初始数据用于初始模型训练和奖励计算<br>
3. 5×10%子数据集用于持续学习训练，每个session使用一个子集<br>
4. 使用狄利克雷分布(Dirichlet)将每个域的数据非独立同分布的划分给多个车辆<br>
5. 每辆车获得相同数量的样本，但类别分布按狄利克雷概率随机分配<br>
6. 域按顺序依次切换，每个域有5个子集循环使用<br>
步骤：<br>
1. 支持三个数据集的读取，并进行数据预处理，使得样本的大小与通道数统一。大小：224x224；通道数：3。<br>
2. 按照域顺序依次加载，第一个域作为初始数据，后续域作为增量数据。<br>
3. 使用狄利克雷分布将每个域的数据非独立同分布的划分给多个车辆。<br>
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
采用DQN算法，包含以下功能：
1. **动作选择**：采用ε-greedy策略，以概率ε生成随机动作进行探索，否则利用策略网络（Q网络）选择动作
2. **动作空间**：动作维度为[NUM_VEHICLES + NUM_VEHICLES]，前NUM_VEHICLES维表示每辆车的上传批次数（0~MAX_UPLOAD_BATCHES），后NUM_VEHICLES维表示带宽分配权重（归一化后和为1）
3. **模型优化**：从回放缓冲区随机采样一个批次的经验，使用策略网络计算当前状态-动作对的Q值，使用目标网络计算下一个状态最大的Q值，根据贝尔曼方程计算目标Q值。使用均方误差损失函数计算当前Q值和目标Q值之间的差值，并通过优化器更新策略网络
4. **目标网络**：每隔一定步数将策略网络参数复制到目标网络，用于稳定训练
5. 保存和加载模型，包括策略网络、目标网络和优化器的状态字典

## 七、实验结果
### 损失变化/时间
rl模型：【奖励为损失变化/时间】
/home/ruiqinghe/workspace/continual_learning/results/pth/office31/trained_drl_model_office31_0.pth
/home/ruiqinghe/workspace/continual_learning/results/pth/digit10/trained_drl_model_digit10_0.pth
测试结果：【在验证集上测试】
/home/ruiqinghe/workspace/continual_learning/results/com_exp/office31/reward_0
/home/ruiqinghe/workspace/continual_learning/results/com_exp/digit10/reward_0
/home/ruiqinghe/workspace/continual_learning/results/npy/office31/npy_0
/home/ruiqinghe/workspace/continual_learning/results/npy/digit10/npy_0

### 损失变化-时间
rl模型：【奖励为损失变化-时间】
/home/ruiqinghe/workspace/continual_learning/results/pth/office31/trained_drl_model_office31_1.pth
/home/ruiqinghe/workspace/continual_learning/results/pth/digit10/trained_drl_model_digit10_1.pth

测试结果：【在验证集上测试】
/home/ruiqinghe/workspace/continual_learning/results/com_exp/office31/reward_1
/home/ruiqinghe/workspace/continual_learning/results/com_exp/digit10/reward_1
/home/ruiqinghe/workspace/continual_learning/results/npy/office31/npy_1
/home/ruiqinghe/workspace/continual_learning/results/npy/digit10/npy_1

测试结果：【在测试集上测试】
/home/ruiqinghe/workspace/continual_learning/results/com_exp/office31/reward_2
/home/ruiqinghe/workspace/continual_learning/results/com_exp/digit10/reward_2
/home/ruiqinghe/workspace/continual_learning/results/npy/office31/npy_2
/home/ruiqinghe/workspace/continual_learning/results/npy/digit10/npy_2