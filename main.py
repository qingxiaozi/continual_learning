# main.py (新增)

from config.parameters import Config
from experiment.rl_env import VehicleEdgeEnv
from models.baseline_agents import (
    StaticAgent, FixedRatioAgent, RandomAgent, LossGreedyAgent
)
from models.drl_agent import DRLAgent
from experiment.rl_test import RLTester
import copy

# ========== 定义所有对比实验 ==========
EXPERIMENT_CONFIGS = {
    # 基线
    "Static": {"BW": None, "UPLOAD": "STATIC", "TRAIN": None},
    "Naive_Baseline": {"BW": "EQUAL", "UPLOAD": "FIXED_RATIO", "TRAIN": "FIXED_RATIO"},
    
    # 带宽消融
    "Proportional_BW": {"BW": "PROPORTIONAL", "UPLOAD": "FIXED_RATIO", "TRAIN": "FIXED_RATIO"},
    "Greedy_Channel_BW": {"BW": "GREEDY_CHANNEL", "UPLOAD": "FIXED_RATIO", "TRAIN": "FIXED_RATIO"},
    
    # 训练策略消融
    "New_Only": {"BW": "MINMAX_DELAY", "UPLOAD": "DRL", "TRAIN": "NEW_ONLY"},
    "Fixed_Ratio_Train": {"BW": "MINMAX_DELAY", "UPLOAD": "DRL", "TRAIN": "FIXED_RATIO"},
    
    # 上传策略消融
    "w_o_DRL": {"BW": "EQUAL", "UPLOAD": "FIXED_RATIO", "TRAIN": "MAB"},
    
    # 完整方案
    "Full_Model": {"BW": "MINMAX_DELAY", "UPLOAD": "DRL", "TRAIN": "MAB"}
}

def create_agent(upload_strategy):
    if upload_strategy == "STATIC":
        return StaticAgent()
    elif upload_strategy == "FIXED_RATIO":
        return FixedRatioAgent(ratio=0.5)
    elif upload_strategy == "RANDOM":
        return RandomAgent()
    elif upload_strategy == "LOSS_GREEDY":
        return LossGreedyAgent()
    elif upload_strategy == "DRL":
        agent = DRLAgent(state_dim=Config.STATE_DIM, action_dim=Config.NUM_VEHICLES)
        agent.load_model("path/to/your/drl_model.pth") # 请替换为您的模型路径
        return agent
    else:
        raise ValueError(f"Unknown upload strategy: {upload_strategy}")

def run_experiment(exp_name, config):
    print(f"\n=== Running Experiment: {exp_name} ===")
    
    # 1. 设置配置
    if config["BW"]: Config.BANDWIDTH_STRATEGY = config["BW"]
    if config["UPLOAD"]: Config.UPLOAD_STRATEGY = config["UPLOAD"]
    if config["TRAIN"]: Config.TRAINING_STRATEGY = config["TRAIN"]
    
    # 2. 创建环境和智能体
    env = VehicleEdgeEnv(mode="test")
    agent = create_agent(Config.UPLOAD_STRATEGY)
    
    # 3. 运行测试
    tester = RLTester(env, agent)
    results = tester.test()
    
    return results

if __name__ == "__main__":
    all_results = {}
    
    # 运行所有实验
    for exp_name, config in EXPERIMENT_CONFIGS.items():
        results = run_experiment(exp_name, config)
        all_results[exp_name] = results
        
        # 保存结果到文件
        with open(f"results_{exp_name}.json", "w") as f:
            import json
            json.dump(results, f, indent=4)
    
    print("\n=== All experiments completed! ===")