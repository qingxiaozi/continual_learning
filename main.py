import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

import json
import time
import os
import pprint
import traceback
from config.parameters import Config
from config.paths import Paths
from experiment.rl_env import VehicleEdgeEnv
from models.drl_agent import DRLAgent
from experiment.rl_test import RLTester, AgentFactory
from torch.utils.data import DataLoader


# ========== 定义所有对比实验 ==========
EXPERIMENT_CONFIGS = {
        "Base_Uniform": {
            "BW": "EQUAL",
            "UPLOAD": "FIXED_RATIO",
            "TRAIN": "FIXED_RATIO",
            "env_group": "group_uniform",
        },
        "Abl_BW_Opt": {
            "BW": "MINMAX_DELAY",
            "UPLOAD": "FIXED_RATIO",
            "TRAIN": "FIXED_RATIO",
            "env_group": "group_uniform",
        },
        "Abl_UP_Greedy": {
            "BW": "EQUAL",
            "UPLOAD": "LOSS_GREEDY",
            "TRAIN": "FIXED_RATIO",
            "env_group": "group_greedy",
        },
        "Abl_UP_DRL": {
            "BW": "EQUAL",
            "UPLOAD": "DRL",
            "TRAIN": "FIXED_RATIO",
            "env_group": "group_drl",
        },
        "Abl_TR_MAB": {
            "BW": "EQUAL",
            "UPLOAD": "FIXED_RATIO",
            "TRAIN": "MAB",
            "env_group": "group_mab",
        },
        "Abl_NoReplay": {
            "BW": "EQUAL",
            "UPLOAD": "FIXED_RATIO",
            "TRAIN": "NEW_ONLY",
            "env_group": "group_noreplay",
        },
        "Combo_Comm": {
            "BW": "MINMAX_DELAY",
            "UPLOAD": "DRL",
            "TRAIN": "FIXED_RATIO",
            "env_group": "group_drl",
        },
        "Combo_Learn": {
            "BW": "EQUAL",
            "UPLOAD": "DRL",
            "TRAIN": "MAB",
            "env_group": "group_combo",
        },
        "Ours_Full": {
            "BW": "MINMAX_DELAY",
            "UPLOAD": "DRL",
            "TRAIN": "MAB",
            "env_group": "group_combo",
        }
    }

# 环境分组 → 相同随机种子（同一组内物理环境完全相同）
ENV_GROUP_SEEDS = {
    "group_uniform": 42,
    "group_greedy": 43,
    "group_drl": 44,
    "group_mab": 45,
    "group_noreplay": 46,
    "group_combo": 47,
}

import multiprocessing as mp

# 必须在任何多进程代码或 CUDA 初始化之前调用
mp.set_start_method('spawn', force=True)

    
def run_and_save(exp_name, config, output_dir):
    env_group = config.get("env_group", exp_name)
    seed = ENV_GROUP_SEEDS.get(env_group, 42)

    Config.BANDWIDTH_STRATEGY = config["BW"]
    Config.UPLOAD_STRATEGY = config["UPLOAD"]
    Config.TRAINING_STRATEGY = config["TRAIN"]
    Config.set_seed(seed)

    tester = RLTester()
    results = tester.test()

    with open(f"{output_dir}/results_{exp_name}.txt", "w", encoding="utf-8") as f:
        formatted_text = pprint.pformat(results, width=120, compact=False)
        f.write(formatted_text + '\n')
    return exp_name, results


if __name__ == "__main__":
    import concurrent.futures
    import multiprocessing
    ctx = mp.get_context('spawn')
    
    output_dir = Paths.get_dataset_dir('com_exp')
    os.makedirs(output_dir, exist_ok=True)
    
    # 运行所有实验（最多6个进程并行）
    with concurrent.futures.ProcessPoolExecutor(max_workers=6, mp_context=ctx) as executor:
        futures = {
            executor.submit(run_and_save, exp_name, config, output_dir): exp_name
            for exp_name, config in EXPERIMENT_CONFIGS.items()
        }

        for future in concurrent.futures.as_completed(futures):
            exp_name = futures[future]
            try:
                future.result()
                logger.info(f"Experiment {exp_name} completed")
            except Exception as e:
                logger.error(f"Experiment {exp_name} failed: {e}")
    
    logger.info("\n=== All experiments completed! ===")