from models.base_agent import BaseAgent
from config.parameters import Config
import random
import torch
import torch.nn as nn
import copy
from models.global_model import GlobalModel


class StaticAgent(BaseAgent):
    """5.1 静态学习"""
    def select_action(self, state, available_batches):
        # 返回空动作，环境看到 UPLOAD_POLICY="STATIC" 会直接跳过
        return [0] * len(available_batches)

class FixedRatioAgent(BaseAgent):
    """5.2 固定比例"""
    def __init__(self, ratio: float = 0.5):
        self.ratio = ratio

    def select_action(self, state, available_batches):
        action = []
        for avail in available_batches:
            upload = int(avail * self.ratio)
            # 确保不超过单次最大上传限制
            upload = min(upload, Config.MAX_UPLOAD_BATCHES)
            action.append(upload)
        return action

class RandomAgent(BaseAgent):
    """随机策略（可选，用于调试）"""
    def select_action(self, state, available_batches):
        action = []
        for avail in available_batches:
            upload = random.randint(0, min(avail, Config.MAX_UPLOAD_BATCHES))
            action.append(upload)
        return action

class LossGreedyAgent(BaseAgent):
    """损失贪心智能体：每步用当前全局模型评估各车数据，优先上传高损失数据所在车辆的批次。"""

    def __init__(self):
        super().__init__()
        self.local_model = GlobalModel(
            dataset_name=Config.DATASET_NAMES,
            init_mode="random"
        ).to(Config.DEVICE)

    def select_action(self, state, available_batches, global_model, vehicles):
        if global_model is None:
            raise ValueError("LossGreedyAgent requires 'global_model' to be passed explicitly.")
        if len(available_batches) != len(vehicles):
            raise ValueError("Length mismatch between available_batches and vehicles.")

        self.local_model.load_state_dict(global_model.state_dict())
        self.local_model.eval()

        # 1. 收集所有批次及其元信息
        candidates = []  # [(vehicle_id, batch_idx, loss), ...]
        with torch.no_grad():
            for vid, count in enumerate(available_batches):
                if count == 0:
                    continue
                vehicle = vehicles[vid]
                for b_idx in range(count):
                    X, y = vehicle.data_batches[b_idx]
                    X, y = X.to(Config.DEVICE), y.to(Config.DEVICE)
                    
                    logits = self.local_model(X)
                    loss_val = nn.functional.cross_entropy(logits, y, reduction='mean').item()
                    candidates.append((loss_val, vid, b_idx))
        # 2. 按损失降序排序
        candidates.sort(key=lambda x: x[2], reverse=True)

        # 3. 贪心选择：最多上传 sum(available_batches) 个批次，且不超过全局限制
        total_quota = sum(available_batches)  # 总上传名额限制
        selected_counts = [0] * len(vehicles) # 记录每辆车被选中的次数
        current_selected_total = 0

        for loss_val, vid, b_idx in candidates:
            if current_selected_total >= total_quota:
                break
            if selected_counts[vid] < Config.MAX_UPLOAD_BATCHES:
                selected_counts[vid] += 1
                current_selected_total += 1

        return tuple(selected_counts)
