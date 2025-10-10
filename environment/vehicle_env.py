import numpy as np
import torch
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.parameters import config
import random

class Vehicle:
    '''
    智能车辆类，代表车路协同系统中的分布式感知节点

    1. 维护车辆自身的状态信息（位置、连接、数据等）
    2. 执行本地模型推理和置信度计算
    3. 管理本地数据缓存
    4. 与基站和边缘服务器进行通信协调
    '''
    def __init__(self, vehicle_id, position):
        self.id = vehicle_id  # 车辆唯一标识符
        self.position = position  # 车辆当前位置坐标，用于确定与哪个基站连接，计算通信质量，模拟车辆运动
        self.cache_data = []
        self.local_model = None
        self.confidence_history = []  # 历史置信度记录
        self.bs_connection = None  # 当前连接的基站ID
        self.data_batches = []  #车辆采集的实时数据批次，未标注
        self.cache_data = []  # 1

    def set_bs_connection(self, bs_id):
        self.bs_connection = bs_id

    def add_data_batch(self, data_batch):
        '''添加数据批次
        data_batch：数据批次，DataLoader对象
        '''
        self.data_batches.append(data_batch)
            # 缓存管理逻辑（可选）：
        if len(self.data_batches) > config.MAX_LOCAL_BATCHES:
            # 移除最旧的数据批次（FIFO策略）
            self.data_batches.pop(0)

    def get_inference_confidence(self, model, data_loader):
        '''计算模型在本地数据上的推理置信度
        1. 对于每个样本，模型输出一个向量logits，通过softmax函数转换为概率分布
        2. 取概率分布中最大的概率值作为该样本的置信度
        3. 对于一个批次的数据，计算所有样本置信度的平均值作为该批次的平均置信度
        '''
        if not data_loader:
            return 0.0

        model.eval()
        total_confidence = 0.0
        count = 0
        # 禁用梯度计算以提升效率
        with torch.no_grad():
            for batch in data_loader:
                # 数据预处理：提取输入
                if isinstance(batch, (list, tuple)):
                    inputs, _ = batch  # 忽略标签，只使用输入
                else:
                    inputs = batch

                # 前向传播获取模型输出
                outputs = model(inputs)

                # 处理不同类型的模型输出
                if hasattr(outputs, 'logits'):
                    # 某些模型（如 transformers）输出包含logits属性
                    outputs = outputs.logits
                # 计算置信度
                probabilities = torch.softmax(outputs, dim=1)  # 转换为概率分布
                batch_confidence = torch.max(probabilities, dim=1)[0]  # 获取每个样本的最大概率
                mean_confidence = batch_confidence.mean().item()  # 批次平均置信度

                total_confidence += mean_confidence
                count += 1
        # 计算整体平均置信度
        avg_confidence = total_confidence / count if count > 0 else 0.0

        # 记录历史置信度
        self.confidence_history.append(avg_confidence)

        # 历史记录管理（保持合理长度）
        if len(self.confidence_history) > config.MAX_CONFIDENCE_HISTORY:
            self.confidence_history.pop(0)

        return avg_confidence


class VehicleEnvironment:
    '''
    车辆环境类

    1. 初始化环境。创建车辆和基站，并将车辆连接到最近的基站
    2. 更新车辆位置。模拟车辆移动，更新车辆与基站的连接
    3. 获取环境状态。为DRL智能体提供状态信息，包括每辆车的置信度、测试损失和数据质量评分
    '''
    def __init__(self):
        # 实体集合
        self.vehicles = []  # 车辆对象列表
        self.base_stations = []  # 基站对象列表，为基站信息字典
        self.current_session = 0  # 当前训练会话编号
        self.road_length = 5000     # 道路长度（米）
        self.road_width = 20        # 道路宽度（米），-10到10
        self.num_lanes = 4          # 车道数量
        self._initialize_environment()  # 初始化环境

    def _initialize_environment(self):
        '''初始化车辆和基站环境
        '''
        # 步骤1: 初始化基站网络
        self._initialize_base_stations()

        # 步骤2: 初始化车辆集群
        self._initialize_vehicles()

        # # 步骤3: 建立初始连接
        # self._establish_initial_connections()

        # # 步骤4: 记录初始状态
        # self._log_initial_environment()

    def _initialize_base_stations(self):
        coverage_radius = config.BASE_STATION_COVERAGE
        optimal_bs_count = max(3, int(self.road_length / (coverage_radius * 0.8)) + 1)
        num_bs = min(optimal_bs_count, 5)  # 最多5个基站
        print(f"初始化 {num_bs} 个基站，覆盖半径: {coverage_radius}米")

        for i in range(num_bs):
            #基站位置：沿着道路均匀分布
            bs_x = i * (self.road_length / (num_bs - 1)) if num_bs > 1 else self.road_length /2
            bs_position = np.array([bs_x, 0])  # 假设基站位于道路中心线
            #基站属性配置
            base_station = {
                'id': i,
                'position': bs_position,
                'coverage': coverage_radius,
                'capacity': 50,  # 最大连接车辆数
                'connected_vehicles': [],  # 当前连接的车辆ID列表
                'utilization': 0.0,  # 利用率
                'signal_strength': 1.0  # 信号强度因子
            }
            self.base_stations.append(base_station)
            print(f"基站 {i} 创建于位置 {bs_position}")

    def _initialize_vehicles(self):
        print(f"初始化 {config.NUM_VEHICLES}辆智能车辆")
        for i in range(config.NUM_VEHICLES):
            position = self._generate_vehicle_position(i)
            # 创建车辆实例
            vehicle = Vehicle(
                vehicle_id = i,
                position = position,
            )
            self.vehicles.append(vehicle)

            # if i < 5:  # 只打印前5辆的信息避免过多输出
            print(f"车辆 {i} 创建于位置 {position}")

    def _generate_vehicle_position(self, vehicle_id):
        """
        为车辆生成初始位置的策略
        """
        # 策略1: 沿道路长度均匀分布
        road_segments = 5
        segment_length = self.road_length / road_segments
        segment_idx = vehicle_id % road_segments
        base_x = segment_idx * segment_length + np.random.uniform(0, segment_length * 0.8)

        # 策略2: 车道分配 - 模拟真实交通流
        lane_width = self.road_width / self.num_lanes
        lane_centers = [-(self.road_width/2) + lane_width/2 + i*lane_width
                    for i in range(self.num_lanes)]

        # 车辆ID决定初始车道，增加随机性
        preferred_lane = vehicle_id % self.num_lanes
        lane_variation = np.random.randint(-1, 2)  # -1, 0, 1
        actual_lane = max(0, min(self.num_lanes-1, preferred_lane + lane_variation))

        y_position = lane_centers[actual_lane] + np.random.uniform(-0.5, 0.5)

        return np.array([base_x, y_position])


if __name__ == "__main__":
    tmp = VehicleEnvironment()