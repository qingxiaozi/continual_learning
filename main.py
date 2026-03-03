# main.py (新增)

from config.parameters import Config
from experiment.rl_env import VehicleEdgeEnv

from models.drl_agent import DRLAgent
from experiment.rl_test import RLTester
import copy

# ========== 定义所有对比实验 ==========
EXPERIMENT_CONFIGS = {
        "Base_Static": {
            "BW": "EQUAL", 
            "UPLOAD": "STATIC", 
            "TRAIN": "NEW_ONLY",
        },
        "Base_Uniform": {
            "BW": "EQUAL", 
            "UPLOAD": "FIXED_RATIO", 
            "TRAIN": "FIXED_RATIO",
        },
        "Abl_BW_Opt": {
            "BW": "MINMAX_DELAY", 
            "UPLOAD": "FIXED_RATIO", 
            "TRAIN": "FIXED_RATIO",
        },
        "Abl_UP_Greedy": {
            "BW": "EQUAL", 
            "UPLOAD": "LOSS_GREEDY", 
            "TRAIN": "FIXED_RATIO",
        },
        "Abl_UP_DRL": {
            "BW": "EQUAL", 
            "UPLOAD": "DRL", 
            "TRAIN": "FIXED_RATIO",
        },
        "Abl_TR_MAB": {
            "BW": "EQUAL", 
            "UPLOAD": "FIXED_RATIO", 
            "TRAIN": "MAB",
        },
        "Abl_NoReplay": {
            "BW": "EQUAL", 
            "UPLOAD": "FIXED_RATIO", 
            "TRAIN": "NEW_ONLY",
        },
        "Combo_Comm": {
            "BW": "MINMAX_DELAY", 
            "UPLOAD": "DRL", 
            "TRAIN": "FIXED_RATIO",
        },
        "Combo_Learn": {
            "BW": "EQUAL", 
            "UPLOAD": "DRL", 
            "TRAIN": "MAB",
        },        
        "Ours_Full": {
            "BW": "MINMAX_DELAY", 
            "UPLOAD": "DRL", 
            "TRAIN": "MAB",
        }
    }

# def create_agent(upload_strategy):
#     if upload_strategy == "STATIC":
#         return StaticAgent()
#     elif upload_strategy == "FIXED_RATIO":
#         return FixedRatioAgent(ratio=0.5)
#     elif upload_strategy == "RANDOM":
#         return RandomAgent()
#     elif upload_strategy == "LOSS_GREEDY":
#         return LossGreedyAgent()
#     elif upload_strategy == "DRL":
#         agent = DRLAgent(state_dim=Config.STATE_DIM, action_dim=Config.NUM_VEHICLES)
#         agent.load_model("./results/pth/trained_drl_model.pth")
#         return agent
#     else:
#         raise ValueError(f"Unknown upload strategy: {upload_strategy}")

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