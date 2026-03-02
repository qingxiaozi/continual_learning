# models/base_agent.py
from abc import ABC, abstractmethod
from typing import List

class BaseAgent(ABC):
    """所有智能体的抽象基类"""
    
    @abstractmethod
    def select_action(self, state, available_batches: List[int]) -> List[int]:
        """
        根据当前状态和可用批次，选择上传动作。
        
        Args:
            state: 环境状态 (可选，某些基线不需要)
            available_batches: 每辆车当前可用的批次数 [a1, a2, ..., av]
            
        Returns:
            action: 上传决策 [m1, m2, ..., mv], 其中 0 <= mv <= av
        """
        pass