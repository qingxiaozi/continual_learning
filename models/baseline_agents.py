from models.base_agent import BaseAgent
from config.parameters import Config
import random


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
    pass
