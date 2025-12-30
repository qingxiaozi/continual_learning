## 一、项目结构
```
continual_learning/
├── main.py
├── config/
│   ├── parameters.py
│   └── paths.py
├── environment/
│   ├── vehicle_env.py
│   ├── communication_env.py
│   └── dataSimu_env.py
├── models/
│   ├── global_model.py
│   ├── gold_model.py
│   ├── mab_selector.py
│   ├── drl_agent.py
│   └── bandwidth_allocator.py
├── learning/
│   ├── continual_learner.py
│   ├── cache_manager.py
│   └── evaluator.py
├── results/
├── utils/
│   └── metrics.py
└── experiments/
    ├── __init__.py
    ├── baseline_comparison.py
    └── ablation_study.py
```

## 二、联合优化模型实验步骤
当前版本代码运行流程：<br>
```python
Initialize all components

get training sessions
get current dataset
get vehicle numbers

for each session:

    _update_session_environment(session)
        更新当前的session，将新域加入到已见域中，预加载新域的数据集，保存到 train_data_cache、test_data_cache和val_data_cache中；
        当域发生变化，将新数据缓存提升为旧数据缓存；
        更新车辆位置，time_delata = 1；
        采用狄利克雷分布为每辆车生成数据批次，保存至vehicle.data_batches，该数据批次即为车辆在环境中实时采集到的数据批次。

    _get_environment_state()
        for each vehicle in vehicles:
            计算置信度，该置信度为车辆所采集的数据样本在全局模型上的最大概率平均值，默认值为 0；
            计算测试损失，该测试损失为上一阶段车辆上传数据在全局模型推理结果与黄金模型标注结果之间的交叉熵，默认值为 1.0。由于当前阶段全局模型还未更新，因此此处的全局模型是上一个阶段训练得到的全局模型；
            计算数据质量评分，该评分为车辆缓存数据的评分的均值，默认值为 0。在第一个session，数据还未到缓存中，因此质量评分为默认值。
        return 环境状态向量

    _get_available_batches()
        return 每辆车实际可用于DRL决策的数据批次数量，即 len(vehicle.data_batches)

    _integrated_decision(state, available_batches, session)
        输入为当前环境状态向量，每辆车实际数据批次数量和session，输出为上传数据批次分配值和带宽分配比例。
        在批次决策过程中，使用了ε-greedy策略。在训练早期，随机生成动作，而后逐步减少探索，在训练后期，更多利用智能体的policy网络输出动作。
        ε(t) = ε_end + (ε_start - ε_end) * exp(-t/τ)
        ε(t)是第 t 步时的探索率
        t是已执行的训练步数
        ε_start是初始探索率，0.9
        ε_end是最终探索率，0.05
        τ为衰减时间常数，800
        exp(-t/τ)为指数衰减因子
        使用当前策略网络（policy network）对输入状态进行前向推理，得到每辆车在各个可选动作（上传批次数）上的 Q 值估计。
        最大允许上传批次为车辆实际数据批次与最大上传数据批次之间的最小值，即 max_allowed = min(available_batches[v], Config.MAX_UPLOAD_BATCHES)
        if max_allowed == 0:
            batch = 0
        else:
            进入ε策略。探索时在randint(0, max_allowed)中随机，利用时切片 :max_allowed+1 进行掩码
        通过spicy计算带宽分配结果

    _upload_datas(batch_choices)
        每辆车根据drl数据批次上传结果随机选择进行上传，即从vehicle.data_batches中随机选择放到vehicle.uploaded_data中。如果结果是0，则清空vehicle.uploaded_data。

    _manage_cache_and_data_selection()
        for vehilce in vehicles:
            将vehicle.uploaded_data放置于缓存中车辆id对应的新数据中，并维护缓存的大小。
        统计缓存总批次和缓存中新旧数据的批次

    _train_and_update_global_model(session)
        收集所有缓存数据构建为全局训练集，并维护全局训练集批次索引到车辆缓存的映射batch_mapping。
        获取验证集。
        计算当前阶段上传的数据集在全局模型上的测试损失和，由于全局模型尚未更新，因此当前的全局模型仍旧是上一阶段训练后的全局模型。
        训练模型得到新模型。该阶段集成MAB选择对数据批次进行质量评分。
        计算当前阶段上传的数据集在全局模型上的测试损失和，由于全局模型已经更新，此时的全局模型为新全局模型。

    _calculate_communication_delay(action, session, corrected_upload_decisions)
        计算通信时延

    _train_and_update_global_model(session)
        for vehicle in vehicles:
            获取车辆的缓存
            for batch_idx, batch in cache["old_data"]:
                将batch添加到全局训练集中
                记录旧数据的映射
            for batch_idx, batch in cache["new_data"]:
                将batch添加到全局训练集中
                记录新数据的映射
        计算所有车辆上传数据在模型上的平均损失；
        利用全局训练集对模型进行训练。
            for each epoch:
                if epoch_count > init_epochs:
                    根据mab算法选择arm
                拿出一个batch
                计算该batch在当前模型上的损失
                训练模型
                计算该batch在训练后模型上的损失
                以损失变化作为计算奖励值


        计算所有车辆上传数据在模型上的平均损失，并同时运行MAB算法更新arm的counts、reward、avg_reward和ucb_counts
        根据训练过程中MAB统计信息更新数据缓存的质量评分

    _evaluate_model_performance(session)
        评估模型在当前域的准确率与损失
        评估模型在已有域上的准确率和损失

    _calculate_reward_and_optimize(state, action, eval_results, comm_results, training_results)
        计算奖励，该奖励为每阶段新上传数据在模型更新前后的损失变化
        获取DRL智能体下一个状态
        存储经验，当经验长度 > Config.DRL_BATCH_SIZE时，优化DRL智能体

    _record_session_results(session, evaluation_results, communication_results, training_results
            )
        记录当前域下的准确率、损失和通信延时

_final_evaluation_and_summary()
    最终准确率、平均准确率、平均通信时延、平均缓存利用率
    各个域上的准确率


```

### data
#### ./mnist/MNIST/raw
- train-images-idx3-ubyte: 训练集图像数据
- train-labels-idx1-ubyte: 训练集标签数据
- t10k-images-idx3-ubyte: 测试集图像数据（t10k代表10000个测试样本）
- t10k-labels-idx1-ubyte: 测试集标签数据
xx.gz是上述文件的压缩版本
#### ./emnist/EMNIST/raw
- emnist-digits-train-images-idx3-ubyte: 训练图像
- emnist-digits-train-labels-idx1-ubyte: 训练标签
- emnist-digits-test-images-idx3-ubyte: 测试图像
- emnist-digits-test-labels-idx1-ubyte: 测试标签
#### ./svhn
每个图像都是32x32像素的RGB图像
#### ./usps
16×16像素的灰度图像<br>
总计 9,298 个样本 (训练集: 7,291，测试集: 2,007)

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

数据集：
|数据集|域|类别数|类别|样本数|图像大小|通道数|说明|
|------|------|------|------|------|------|------|------|
|office-31|Amazon、Webcam、DSLR|31|back_pack、bottle、desktop_computer、laptop_computer、mouse、phone、ring_binder、stapler、bike、calculator、file_cabinet、letter_tray、mug、printer、ruler、tape_dispenser、bike_helmet、desk_chair、headphones、mobile_phone、paper_notebook、projector、scissors、trash_can、bookcase、desk_lamp、keyboard、monitor、pen、punchers、speaker|Amazon:2817；dslr:498；webcam:795|图像大小|通道数|说明|
|digit10|MNIST、EMNIST、USPS、SVHN|10|0、1、2、3、4、5、6、7、8、9|MNIST：70000；EMNIST：280000；USPS：7291；SVHN：99289|图像大小|通道数|数字识别|
|DomainNet|Clipart、Infograph、Painting、Quickdraw、Real、Sketch|类别数|类别|Real: 约 172,000 张；Clipart: 约 48,000 张；Painting: 约 72,000 张；Quickdraw: 约 172,000 张；Sketch: 约 72,000 张；Infograph: 约 48,000 张|图像大小|通道数|说明|
<br>
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
2. 深度学习buffer为32，说明至少得32个session，花时间
3. 修改main.py，进行基线对比实验 10days
4. 消融实验 5days
5. 完善digit和dominNet数据集训练 10days
