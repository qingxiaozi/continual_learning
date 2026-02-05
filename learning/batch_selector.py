from abc import ABC, abstractmethod


class BatchSelector(ABC):
    """批次选择策略抽象接口"""

    @abstractmethod
    def select(self) -> int:
        pass

    @abstractmethod
    def update(self, arm: int, reward: float):
        pass

    @abstractmethod
    def get_quality_scores(self):
        pass

    @abstractmethod
    def reset(self):
        pass