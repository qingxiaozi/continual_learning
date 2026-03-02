from models.base_agent import BaseAgent
from config.parameters import Config
import random
import torch
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

    def __init__(self, env):
        super().__init__()
        if env is None:
            raise ValueError("LossGreedyAgent requires a valid environment instance.")
        self.env = env

        # 初始化本地副本（结构与全局模型一致）
        self.local_model = GlobalModel(
            dataset_name=Config.DATASET_NAMES,
            init_mode="random"
        ).to(Config.DEVICE)

    def select_action(self, state, available_batches):
        global_model = self.env.continual_learner.model
        self.local_model.load_state_dict(global_model.state_dict())
        self.local_model.eval()

        # 1. 收集所有批次及其元信息
        all_batches_info = []  # [(vehicle_id, batch_idx, loss), ...]
        with torch.no_grad():
            for vehicle_id, avail in enumerate(available_batches):
                if avail == 0:
                    continue
                vehicle = self.env.vehicle_env.vehicles[vehicle_id]
                for batch_idx, (X, y) in enumerate(vehicle.data_batches):
                    X, y = X.to(Config.DEVICE), y.to(Config.DEVICE)
                    logits = self.local_model(X)
                    loss = torch.nn.functional.cross_entropy(logits, y, reduction='mean')
                    all_batches_info.append((vehicle_id, batch_idx, loss.item()))

        # 2. 按损失降序排序
        all_batches_info.sort(key=lambda x: x[2], reverse=True)

        # 3. 贪心选择：最多上传 sum(available_batches) 个批次，且不超过全局限制
        total_quota = sum(available_batches)
        selected_vehicle_counts = [0] * len(available_batches)
        selected_count = 0

        for vid, _, _ in all_batches_info:
            if selected_count >= total_quota:
                break
            if selected_vehicle_counts[vid] < available_batches[vid]:
                selected_vehicle_counts[vid] += 1
                selected_count += 1

        # 4. 确保不超过单次最大上传限制（按车辆）
        final_action = [
            min(count, Config.MAX_UPLOAD_BATCHES)
            for count in selected_vehicle_counts
        ]
        return final_action
