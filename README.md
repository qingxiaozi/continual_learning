## 一、项目结构
```
continual_learning/
├── main.py
├── Config/
│   ├── parameters.py
│   └── paths.py
├── environment/
│   ├── vehicle_env.py
│   ├── communication_env.py
│   └── data_simulator.py
├── models/
│   ├── global_model.py
│   ├── gold_model.py
│   ├── mab_selector.py
│   └── drl_agent.py
├── learning/
│   ├── continual_learner.py
│   ├── cache_manager.py
│   └── evaluator.py
├── utils/
│   ├── data_loader.py
│   └── metrics.py
└── experiments/
    ├── __init__.py
    ├── baseline_comparison.py
    └── ablation_study.py
```

## 二、联合优化模型实验步骤
当前版本代码运行流程：<br>
对于每个session:<br>
    更新会话和环境状态。更新当前的session，将对应域加入到已见域中，预加载域的数据集，并保存到train_data_cache和test_data_cache中；更新车辆位置：经过time_delta=1后车辆的位置；为车辆生成新的数据，采用狄利克雷分布为每辆车生成数据批次，该数据批次即为车辆在环境中实时采集到的数据批次。<br>
    获取环境状态。获取每辆车的置信度、测试损失和数据质量评分。置信度为每辆车当前新采集到的数据在全局模型上的平均置信度，即每个样本最大概率的平均值。测试损失为每辆车上传数据的测试损失，在第一个session中，由于车辆还未上传数据，因此其测试损失为默认值1.0。当session>1时，车辆已有上传数据，测试损失的计算为全局模型推理结果与黄金模型标注结果之间的交叉熵。此时的全局模型为当前阶段中的全局模型，也是上一个阶段得到的全局模型，因为全局模型还未进行更新。数据质量评分为每辆车缓存数据批次评分的均值，由于每辆车的最大缓存为5个批次，所以当车辆缓存为5个批次时，数据质量评分为这5个数据批次的质量评分的均值，当车辆缓存小于5个批次时，数据质量评分为车辆实际缓存批次的质量评分的均值。在第一个session中，由于车辆还未缓存数据批次，因此其质量评分为默认值0；当session>1时，由于每辆车已经缓存了数据批次，其质量评分为计算值。质量评分应在模型训练过程由多臂老虎机算法得到，预计在下一版本中实现。<br>
    智能体进行决策。输入为当前环境状态，包括置信度、测试损失和数据质量评分，输出为上传数据批次分配和带宽分配比例。在智能体决策过程中，使用了ε-greedy策略。在训练早期，广泛尝试不同的策略，即随机生成动作，而后逐步减少探索，在训练后期，更多利用智能体的policy网络输出动作。<br>
    执行通信和数据收集。根据智能体输出的动作，上传数据批次。由于智能体输出动作中上传数据批次可能大于车辆已有的数据批次，因此实际上传两者中的最小值。而后，计算车辆通信时延。该通信时延已经包含上传、标注、训练和广播时延。其中通信参数需要在整体流程跑通后进行修改。<br>
    缓存管理和数据选择。计算每辆车上传数据的质量评分，该评分是利用多臂老虎机算法中奖励计算函数实现的，是车辆上传数据在全局模型上的测试损失的负值，待修改。而后更新云端缓存管理，将车辆实际上传数据放到每辆车数据缓存的“新数据”里，并维护每辆车的缓存大小为固定的5个数据批次。<br>
    训练并更新全局模型。收集云端缓存器中所有车辆的缓存数据构建出一个全局训练集，计算这一阶段新上传数据在全局模型上的平均测试损失，而后训练模型，得到新的全局模型，计算这一阶段新上传数据在新训练出的全局模型上的测试损失。<br>
    评估模型性能。计算新训练出来的模型在当前域测试集上的准确率和损失，在所有已见域上的平均准确率、平均损失和在每个域上的结果。<br>

## data
### ./mnist/MNIST/raw
- train-images-idx3-ubyte: 训练集图像数据
- train-labels-idx1-ubyte: 训练集标签数据
- t10k-images-idx3-ubyte: 测试集图像数据（t10k代表10000个测试样本）
- t10k-labels-idx1-ubyte: 测试集标签数据
xx.gz是上述文件的压缩版本
### ./emnist/EMNIST/raw
- emnist-digits-train-images-idx3-ubyte: 训练图像
- emnist-digits-train-labels-idx1-ubyte: 训练标签
- emnist-digits-test-images-idx3-ubyte: 测试图像
- emnist-digits-test-labels-idx1-ubyte: 测试标签
### ./svhn
每个图像都是32x32像素的RGB图像
### ./usps
16×16像素的灰度图像<br>
总计 9,298 个样本 (训练集: 7,291，测试集: 2,007)

## environment
### 车辆环境（vehicle_env）
1. 初始化车辆和基站环境。
2. 建立基站，确定其位置、覆盖范围、最大连接车辆数等属性；
3. 建立车辆集群，确定其初始位置，将车辆与距离其最近的基站连接；
4. 更新车辆位置，车辆以速度8-20 m/s(29-72 km/h)行驶，可确定其在任意时间点的位置；
5. 重置环境至初始状态。
6. 遗留问题：车辆行驶过程中可能碰撞

### 通信环境（communication_env）
1. 计算上下行通信速率
2. 计算数据传输时延 (t_trans)
3. 计算数据标注时延 (t_label)
4. 计算模型重训练时延 (t_retrain)
5. 计算模型广播时延 (t_broadcast)
6. 遗留问题：传输时延计算单位需要确认，总带宽为MHz，样本大小为bit

### 数据模拟（dataSimu_env）
功能：<br>
分别处理office-31、digit10、DomainNet三个数据集，将其按照Non-IID（狄利克雷分布）的方式划分给多个智能车辆，并支持增量学习。<br>
步骤：<br>
1. 支持三个数据集的读取，并进行数据预处理，使得样本的大小与通道数统一。大小：224x224；通道数：3。
2. 按照域顺序依次加载，第一个域作为初始数据，后续域作为增量数据。
3. 使用狄利克雷分布将每个域的数据非独立同分布的划分给多个车辆。
4. 每个车辆维护自己的数据缓存，并能够按照批次提供数据。

数据集：
|数据集|域|类别数|类别|样本数|图像大小|通道数|说明|
|------|------|------|------|------|------|------|------|
|office-31|Amazon、Webcam、DSLR|31|back_pack、bottle、desktop_computer、laptop_computer、mouse、phone、ring_binder、stapler、bike、calculator、file_cabinet、letter_tray、mug、printer、ruler、tape_dispenser、bike_helmet、desk_chair、headphones、mobile_phone、paper_notebook、projector、scissors、trash_can、bookcase、desk_lamp、keyboard、monitor、pen、punchers、speaker|Amazon:2817；dslr:498；webcam:795|图像大小|通道数|说明|
|digit10|MNIST、EMNIST、USPS、SVHN|10|0、1、2、3、4、5、6、7、8、9|样本数|图像大小|通道数|数字识别|
|DomainNet|Clipart、Infograph、Painting、Quickdraw、Real、Sketch|类别数|类别|样本数|图像大小|通道数|说明|
<br>

## models
### 多臂老虎机（mab_selector）
功能：<br>
用于选择高质量的数据批次。<br>
1. 使用ucb算法选择所有臂（数据批次）中价值最高的臂。
2. 更新臂的统计，包括被选择的次数、累积奖励和平均奖励。
3. 计算一个数据批次在当前模型下的损失，并以该损失的负值作为该数据批次的奖励。注意：并未更新模型。
4. 计算所有批次的归一化质量评分。
5. MAB状态重置，即被选择的此时、累积奖励、平均奖励均为0。


### 强化学习智能体（drl_agent）
1. 选择动作。以一定概率生成随机动作，否则利用策略网络选择动作。
2. 将数据标注量动作通过线性变换映射到[0, MAX_UPLOAD_BATCHES]的整数，将带宽分配进行归一化，使得总和为1。
3. 模型优化。从回放缓冲区随机采样一个批次的经验，使用策略网络计算当前状态-动作对的Q值，使用目标网络计算下一个状态最大的Q值，根据贝尔曼方程计算目标Q值。使用均方误差损失函数计算当前Q值和目标Q值之间的车衣，并通过优化器更新策略网络。
4. 保存和加载模型，包括策略网络、目标网络和优化器的状态字典。


####  接下来要做的
1. 根据mab进行数据质量评分，估计需要修改逻辑架构 90%，待检查
2. 修改模型训练，划分出验证集 2days，模型训练的时候，loss为0，需要早停
3. 遗忘指标计算 1day
4. 修改main.py，进行基线对比实验 10days
5. 消融实验 5days
6. 完善digit和dominNet数据集训练 7days

最佳预期：预计12月底完成。