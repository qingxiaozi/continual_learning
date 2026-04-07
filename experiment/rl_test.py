from collections import defaultdict
import logging

logging.basicConfig(
    level=logging.WARNING,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

import os
import wandb
import torch
import itertools
import numpy as np
from config.parameters import Config
from config.paths import Paths

logger = logging.getLogger(__name__)
from experiment.rl_env import VehicleEdgeEnv
from models.drl_agent import DRLAgent
from utils.metrics import IncrementalMetricsCalculator
from utils.visualizer import ResultVisualizer
from models.baseline_agents import (
    StaticAgent, FixedRatioAgent, RandomAgent, LossGreedyAgent
)


class AgentFactory:
    @staticmethod
    def create_agent(upload_strategy,state_dim):
        if upload_strategy == "STATIC":
            return StaticAgent()
        elif upload_strategy == "FIXED_RATIO":
            return FixedRatioAgent(ratio=0.5)
        elif upload_strategy == "RANDOM":
            return RandomAgent()
        elif upload_strategy == "LOSS_GREEDY":
            return LossGreedyAgent()
        elif upload_strategy == "DRL":
            agent = DRLAgent(state_dim)
            agent.load_model(Paths.get_drl_model_path())
            agent.set_eval_mode()
            return agent
        else:
            raise ValueError(f"Unknown upload strategy: {upload_strategy}")


class RLTester:
    def __init__(self):
        Config.set_seed()
        self.env = VehicleEdgeEnv(mode="test")
        dummy_state = self.env.reset() 
        state_dim = dummy_state.shape[0]
        self.agent = AgentFactory.create_agent(Config.UPLOAD_STRATEGY, state_dim)

        self.num_episodes = Config.NUM_TEST_EPISODES
        self.max_timesteps = Config.NUM_TESTING_SESSIONS
        self.domain_interval = Config.DOMAIN_CHANGE_INTERVAL       # e.g., 7 or 20

        # ===== System metrics =====
        self.episode_rewards = []
        self.episode_delays = []

        # ===== CL metrics buffers (across episodes) =====
        self.AA_all = []
        self.FM_all = []
        self.BWT_all = []
        self.AIA_all = []

        self.AA_steps = [] # list of list, per episode
        self.FM_steps = []
        self.BWT_steps = []

        # Accuracy matrix buffer (paper visualization)
        self.accuracy_matrices = []
        
        # 可视化器
        self.visualizer = ResultVisualizer()

    def test(self):
        """测试循环"""
        logger.info("testing...")

        run_name = f"mp_{Config.BANDWIDTH_STRATEGY}_{Config.UPLOAD_STRATEGY}_{Config.TRAINING_STRATEGY}_reward_1"
        self.wandb_run = wandb.init(
            project=f"Vehicle-Edge-CL-Testing_{Config.CURRENT_DATASET}",
            name=run_name,
            config={
                "upload_strategy": Config.UPLOAD_STRATEGY,
                "bandwidth_strategy": Config.BANDWIDTH_STRATEGY,
                "training_strategy": Config.TRAINING_STRATEGY,
                "num_episodes": self.num_episodes,
                "max_timesteps": self.max_timesteps
            },
            reinit=True  # 允许在同一个进程中多次 init
        )

        for episode in range(self.num_episodes):
            logger.info(f"\n===== Test Episode {episode+1}/{self.num_episodes} =====")
            state = self.env.reset()
            total_reward = 0
            step_delays = []

            seen_tasks = []
            accuracy_history = {}
            acc_matrix = []
            AA_curve, FM_curve, BWT_curve = [], [], []

            cached_loaders = {}

            for t in range(self.max_timesteps):
                available_batches = state[-Config.NUM_VEHICLES:].astype(int).tolist()

                if Config.UPLOAD_STRATEGY == "LOSS_GREEDY":
                    action = self.agent.select_action(
                        state,
                        available_batches=available_batches,
                        global_model=self.env.global_model,
                        vehicles=self.env.vehicle_env.vehicles
                    )
                else:
                    action = self.agent.select_action(state, available_batches=available_batches)

                next_state, reward, done, info = self.env.step(action)
                state = next_state

                total_reward += reward
                step_delays.append(info["comm"]["t_total_comm"])

                current_domain = self.env.current_domain
                current_sub_idx = self.env.data_simulator._get_current_sub_domain_idx()
                current_task = (current_domain, current_sub_idx)

                if current_task not in seen_tasks:
                    seen_tasks.append(current_task)

                cumulative_datasets = self.env.data_simulator.get_cumulative_test_datasets()

                row = []
                for task in seen_tasks:
                    if task not in cached_loaders and task in cumulative_datasets:
                        test_dataset = cumulative_datasets[task]
                        cached_loaders[task] = DataLoader(test_dataset, batch_size=Config.BATCH_SIZE)

                    if task in cached_loaders:
                        acc, _ = self.env.evaluator.evaluate_model(self.env.global_model.model, cached_loaders[task])
                    else:
                        acc = 0.0
                    row.append(acc)
                    if task not in accuracy_history:
                        accuracy_history[task] = []
                    accuracy_history[task].append(acc)

                acc_matrix.append(row)

                metrics = IncrementalMetricsCalculator.compute_metrics(
                    seen_tasks, accuracy_history
                )

                AA_curve.append(metrics["AA"])
                FM_curve.append(metrics["FM"])
                BWT_curve.append(metrics["BWT"])

                if done:
                    break

            final_aa = AA_curve[-1] if AA_curve else 0.0
            final_fm = FM_curve[-1] if FM_curve else 0.0
            final_bwt = BWT_curve[-1] if BWT_curve else 0.0
            AIA = IncrementalMetricsCalculator.compute_aia(AA_curve)

            self.AA_all.append(final_aa)
            self.FM_all.append(final_fm)
            self.BWT_all.append(final_bwt)
            self.AIA_all.append(AIA)

            self.AA_steps.append(AA_curve)
            self.FM_steps.append(FM_curve)
            self.BWT_steps.append(BWT_curve)
            self.accuracy_matrices.append(acc_matrix)

            self.episode_rewards.append(total_reward)
            self.episode_delays.append(np.mean(step_delays))

            self.wandb_run.log({
                "episode": episode,
                "episode_reward": total_reward,
                "episode_delay": np.mean(step_delays),
                "episode_AA": final_aa,
                "episode_FM": final_fm,
                "episode_BWT": final_bwt,
                "episode_AIA": AIA
            })

        results_dict =self.report_results()
        self.save_results()

        wandb.log({
            "CL_Curves/AA_Steps": self.AA_steps,
            "CL_Curves/FM_Steps": self.FM_steps,
            "CL_Curves/BWT_Steps": self.BWT_steps
        })
        self.wandb_run.finish()
        
        return results_dict

    def report_results(self):
        logger.info("\n================ FINAL PAPER RESULTS ================")

        results = {
            "config": {
                "BW": Config.BANDWIDTH_STRATEGY,
                "UPLOAD": Config.UPLOAD_STRATEGY,
                "TRAIN": Config.TRAINING_STRATEGY
            },
            "metrics": {}
        }
        
        logger.info("\n[System configure]")
        logger.info(f"BW: {Config.BANDWIDTH_STRATEGY}")
        logger.info(f"UPLOAD: {Config.UPLOAD_STRATEGY}")
        logger.info(f"TRAIN: {Config.TRAINING_STRATEGY}")

        def report(name, values, category="metrics"):
            mean_val = float(np.mean(values))
            std_val = float(np.std(values))
            results[category][name] = {"mean": mean_val, "std": std_val}
            print(f"{name}: {np.mean(values):.4f} ± {np.std(values):.4f}")

        logger.info("\n[Continual Learning Metrics]")
        report("AA", self.AA_all)
        report("FM", self.FM_all)
        report("BWT", self.BWT_all)
        report("AIA", self.AIA_all)

        logger.info("\n[System Metrics]")
        report("Mean Reward", self.episode_rewards)
        report("Mean Communication Delay", self.episode_delays)

        return results


    def save_results(self):
        # os.makedirs(Paths.RESULTS_NPY_DIR, exist_ok=True)
        prefix = f"{Config.UPLOAD_STRATEGY}_{Config.BANDWIDTH_STRATEGY}_{Config.TRAINING_STRATEGY}"
        np.save(f"{Paths.get_dataset_dir('npy')}/{prefix}_AA_steps.npy", np.array(self.AA_steps))
        np.save(f"{Paths.get_dataset_dir('npy')}/{prefix}_FM_steps.npy", np.array(self.FM_steps))
        np.save(f"{Paths.get_dataset_dir('npy')}/{prefix}_BWT_steps.npy", np.array(self.BWT_steps))
        np.save(f"{Paths.get_dataset_dir('npy')}/{prefix}_accuracy_matrices.npy", np.array(self.accuracy_matrices, dtype=object))
        np.save(f"{Paths.get_dataset_dir('npy')}/{prefix}_AA_all.npy", np.array(self.AA_all))
        np.save(f"{Paths.get_dataset_dir('npy')}/{prefix}_FM_all.npy", np.array(self.FM_all))
        np.save(f"{Paths.get_dataset_dir('npy')}/{prefix}_BWT_all.npy", np.array(self.BWT_all))
        np.save(f"{Paths.get_dataset_dir('npy')}/{prefix}_AIA_all.npy", np.array(self.AIA_all))
        np.save(f"{Paths.get_dataset_dir('npy')}/{prefix}_episode_rewards.npy", np.array(self.episode_rewards))
        np.save(f"{Paths.get_dataset_dir('npy')}/{prefix}_episode_delays.npy", np.array(self.episode_delays))
    
if __name__ == "__main__":
    tester = RLTester()
    tester.test()