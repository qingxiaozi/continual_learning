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
train-images-idx3-ubyte: 训练集图像数据<br>
train-labels-idx1-ubyte: 训练集标签数据<br>
t10k-images-idx3-ubyte: 测试集图像数据（t10k代表10000个测试样本）<br>
t10k-labels-idx1-ubyte: 测试集标签数据<br>
xx.gz是上述文件的压缩版本<br>
### ./emnist/EMNIST/raw
emnist-digits-train-images-idx3-ubyte: 训练图像<br>
emnist-digits-train-labels-idx1-ubyte: 训练标签<br>
emnist-digits-test-images-idx3-ubyte: 测试图像<br>
emnist-digits-test-labels-idx1-ubyte: 测试标签<br>
### ./svhn
每个图像都是32x32像素的RGB图像
### ./usps
16×16像素的灰度图像<br>
总计 9,298 个样本 (训练集: 7,291，测试集: 2,007)

## Environment
### 车辆环境（VehicleEnvironment）：
初始化车辆和基站环境。<br>
建立基站，确定其位置、覆盖范围、最大连接车辆数等属性；<br>
建立车辆集群，确定其初始位置，将车辆与距离其最近的基站连接；<br>
更新车辆位置，车辆以速度8-20 m/s(29-72 km/h)行驶，可确定其在任意时间点的位置；<br>
重置环境至初始状态。<br>
遗留问题：车辆行驶过程中可能碰撞<br>

### 通信系统（CommunicationSystem）：
计算上下行通信速率<br>
计算数据传输时延 (t_trans)<br>
计算数据标注时延 (t_label)<br>
计算模型重训练时延 (t_retrain)<br>
计算模型广播时延 (t_broadcast)<br>

### 数据模拟（DataDistributionSimulator）：
功能：<br>
分别处理office-31、digit10、DomainNet三个数据集，将其按照Non-IID（狄利克雷分布）的方式划分给多个智能车辆，并支持增量学习。<br>
步骤：<br>
1、支持三个数据集的读取，并进行数据预处理，使得样本的大小与通道数统一。大小：224x224；通道数：3。<br>
2、按照域顺序依次加载，第一个域作为初始数据，后续域作为增量数据。<br>
3、使用狄利克雷分布将每个域的数据非独立同分布的划分给多个车辆。<br>
4、每个车辆维护自己的数据缓存，并能够按照批次提供数据。<br><br>
数据集：
|数据集|域|类别数|类别|样本数|图像大小|通道数|说明|
|------|------|------|------|------|------|------|------|
|office-31|Amazon、Webcam、DSLR|31|back_pack、bottle、desktop_computer、laptop_computer、mouse、phone、ring_binder、stapler、bike、calculator、file_cabinet、letter_tray、mug、printer、ruler、tape_dispenser、bike_helmet、desk_chair、headphones、mobile_phone、paper_notebook、projector、scissors、trash_can、bookcase、desk_lamp、keyboard、monitor、pen、punchers、speaker|Amazon:2871；dslr:498；webcam:795|图像大小|通道数|说明|
|digit10|MNIST、EMNIST、USPS、SVHN|10|0、1、2、3、4、5、6、7、8、9|样本数|图像大小|通道数|数字识别|
|DomainNet|Clipart、Infograph、Painting、Quickdraw、Real、Sketch|类别数|类别|样本数|图像大小|通道数|说明|