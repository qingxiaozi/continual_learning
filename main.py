import numpy as np
from config.parameters import Config
from environment.communication_env import CommunicationSystem
from environment.dataSimu_env import DomainIncrementalDataSimulator
from environment.vehicle_env import VehicleEnvironment
from learning.cache_manager import CacheManager
from learning.evaluator import ModelEvaluator
from learning.continual_learner import ContinualLearner
from models.drl_agent import DRLAgent
from models.global_model import GlobalModel
from models.gold_model import GoldModel
from models.mab_selector import MABDataSelector

class BaselineComparison:
    """基线算法比较实验"""
    def __init__(self):
        # 初始化所有组件
        self.config = Config
        self.data_simulator = DomainIncrementalDataSimulator()

        # 初始化模型
        self.gold_model = GoldModel(self.data_simulator.current_dataset)
        self.global_model = GlobalModel(self.data_simulator.current_dataset)

        # 初始化学习组件
        self.cache_manager = CacheManager()
        self.evaluator = ModelEvaluator(self.gold_model)
        self.continual_learner = ContinualLearner(self.global_model, self.gold_model)

        # 初始化环境
        self.vehicle_env = VehicleEnvironment(
            self.global_model,
            self.gold_model,
            self.cache_manager,
            self.data_simulator
        )

        # 初始化通信系统
        self.communication_system = CommunicationSystem(self.vehicle_env)

        # 初始化MAB选择器
        self.mab_selector = MABDataSelector(num_arms=Config.MAX_LOCAL_BATCHES)

        # 初始化DRL智能体
        state_dim = 3 * Config.NUM_VEHICLES  # 置信度、测试损失、质量评分
        action_dim = 2 * Config.NUM_VEHICLES  # 上传批次、带宽分配
        self.drl_agent = DRLAgent(state_dim, action_dim)

        # 记录实验数据
        self.results = {
            'session_accuracies': [],
            'session_losses': [],
            'communication_delays': [],
            'cache_utilization': [],
            'domain_performance': defaultdict(list)
        }

    def run_joint_optimization(self, num_sessions=None):
        """运行完整的联合优化过程"""
        if num_sessions is None:
            num_sessions = Config.NUM_TRAINING_SESSIONS
        print("=" * 60)
        print("开始车路协同持续学习联合优化")
        print(f"数据集: {self.data_simulator.current_dataset}")
        print(f"车辆数量: {Config.NUM_VEHICLES}")
        print(f"训练会话数: {num_sessions}")
        print("=" * 60)





if __name__ == "__main__":
    a = BaselineComparison()
    a.run_mab_drl()
