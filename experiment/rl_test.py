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

        run_name = f"mp_{Config.BANDWIDTH_STRATEGY}_{Config.UPLOAD_STRATEGY}_{Config.TRAINING_STRATEGY}"
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

            seen_tasks  = []
            accuracy_history = defaultdict(list)
            acc_matrix = []
            AA_curve, FM_curve, BWT_curve = [], [], []

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

                if (t + 1) % self.domain_interval == 0:
                    current_task = self.env.current_domain
                    if current_task not in seen_tasks:
                        seen_tasks.append(current_task)
                        
                        # # ===== 域切换时调用可视化函数 =====
                        # print(f"\n Visualizing domain shift and data heterogeneity...")
                        # self.visualizer.plot_vehicle_data_heterogeneity(
                        #     self.env.data_simulator, 
                        #     session=self.env.session,
                        #     save_plot=True
                        # )
                        # self.visualizer.plot_tsne_domain_shift(
                        #     self.env.global_model,
                        #     self.env.data_simulator,
                        #     session=self.env.session,
                        #     num_samples=500,
                        #     save_plot=True
                        # )

                    row = []
                    
                    for d in seen_tasks:
                        acc = self.env.evaluate_model(d)
                        row.append(acc)
                        accuracy_history[d].append(acc)

                    acc_matrix.append(row)

                    metrics = IncrementalMetricsCalculator.compute_metrics(
                        seen_tasks, accuracy_history
                    )

                    AA_curve.append(metrics["AA"])
                    FM_curve.append(metrics["FM"])
                    BWT_curve.append(metrics["BWT"])

                if done:
                    break

            final_metrics = IncrementalMetricsCalculator.compute_metrics(
                seen_tasks, accuracy_history
            )
            AIA = IncrementalMetricsCalculator.compute_aia(AA_curve)

            self.AA_all.append(final_metrics["AA"])
            self.FM_all.append(final_metrics["FM"])
            self.BWT_all.append(final_metrics["BWT"])
            self.AIA_all.append(AIA)

            self.AA_steps.append(AA_curve)
            self.FM_steps.append(FM_curve)
            self.BWT_steps.append(BWT_curve)
            self.accuracy_matrices.append(acc_matrix)

            # ===== System metrics =====
            self.episode_rewards.append(total_reward)
            self.episode_delays.append(np.mean(step_delays))

            # Log to wandb for each episode
            self.wandb_run.log({
                "episode": episode,
                "episode_reward": total_reward,
                "episode_delay": np.mean(step_delays),
                "episode_AA": final_metrics["AA"],
                "episode_FM": final_metrics["FM"],
                "episode_BWT": final_metrics["BWT"],
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
        os.makedirs(Paths.RESULTS_NPY_DIR, exist_ok=True)
        prefix = f"{Config.UPLOAD_STRATEGY}_{Config.BANDWIDTH_STRATEGY}_{Config.TRAINING_STRATEGY}"
        np.save(f"{Paths.RESULTS_NPY_DIR}/{prefix}_AA_steps.npy", np.array(self.AA_steps))
        np.save(f"{Paths.RESULTS_NPY_DIR}/{prefix}_FM_steps.npy", np.array(self.FM_steps))
        np.save(f"{Paths.RESULTS_NPY_DIR}/{prefix}_BWT_steps.npy", np.array(self.BWT_steps))
        np.save(f"{Paths.RESULTS_NPY_DIR}/{prefix}_accuracy_matrices.npy", np.array(self.accuracy_matrices, dtype=object))
        np.save(f"{Paths.RESULTS_NPY_DIR}/{prefix}_AA_all.npy", np.array(self.AA_all))
        np.save(f"{Paths.RESULTS_NPY_DIR}/{prefix}_FM_all.npy", np.array(self.FM_all))
        np.save(f"{Paths.RESULTS_NPY_DIR}/{prefix}_BWT_all.npy", np.array(self.BWT_all))
        np.save(f"{Paths.RESULTS_NPY_DIR}/{prefix}_AIA_all.npy", np.array(self.AIA_all))
        np.save(f"{Paths.RESULTS_NPY_DIR}/{prefix}_episode_rewards.npy", np.array(self.episode_rewards))
        np.save(f"{Paths.RESULTS_NPY_DIR}/{prefix}_episode_delays.npy", np.array(self.episode_delays))
    
if __name__ == "__main__":
    tester = RLTester()
    tester.test()