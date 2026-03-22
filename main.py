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


# ========== 定义所有对比实验 ==========
EXPERIMENT_CONFIGS = {
        # "Base_Static": {
        #     "BW": "EQUAL", 
        #     "UPLOAD": "STATIC", 
        #     "TRAIN": "NEW_ONLY",
        # },
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

def run_experiment(exp_name, config):
    print(f"\n=== Running Experiment: {exp_name} ===")
    
    try:
        Config.BANDWIDTH_STRATEGY = config["BW"]
        Config.UPLOAD_STRATEGY = config["UPLOAD"]
        Config.TRAINING_STRATEGY = config["TRAIN"]

        tester = RLTester()
        results = tester.test()
    
        return results
    except Exception as e:
        print(f"Experiment {exp_name} FAILED: {str(e)}")
        traceback.print_exc()
        return {"error": str(e), "status": "failed"} 


if __name__ == "__main__":
    import concurrent.futures
    import multiprocessing
    
    output_dir = Paths.RESULTS_COM_EXP_DIR
    os.makedirs(output_dir, exist_ok=True)
    
    def run_and_save(exp_name, config):
        Config.BANDWIDTH_STRATEGY = config["BW"]
        Config.UPLOAD_STRATEGY = config["UPLOAD"]
        Config.TRAINING_STRATEGY = config["TRAIN"]
        
        tester = RLTester()
        results = tester.test()

        with open(f"{output_dir}/results_{exp_name}.txt", "w", encoding="utf-8") as f:
            formatted_text = pprint.pformat(results, width=120, compact=False)
            f.write(formatted_text + '\n')
        return exp_name, results
    
    # 运行所有实验（8个进程并行）
    with concurrent.futures.ProcessPoolExecutor(max_workers=16) as executor:
        futures = {
            executor.submit(run_and_save, exp_name, config): exp_name 
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