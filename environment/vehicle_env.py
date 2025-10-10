import numpy as np
import torch
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.parameters import config

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
