from collections import defaultdict
import torch
import numpy as np
from config.parameters import Config
from experiment.rl_env import VehicleEdgeEnv
from models.drl_agent import DRLAgent
from utils.metrics import IncrementalMetricsCalculator, ResultVisualizer


class RLTester:
    def __init__(self, model_path="./results/trained_drl_model.pth"):
        self.env = VehicleEdgeEnv()
        self.agent = DRLAgent(state_dim=self.env.config.STATE_DIM)
        self.agent.load_model(model_path)
        self.agent.set_eval_mode()
        self.visualizer = ResultVisualizer()
        self.num_episodes = Config.NUM_TEST_EPISODES
        self.max_timesteps = Config.NUM_TESTING_SESSIONS
        
        # ===== System metrics =====
        self.episode_rewards = []
        self.episode_delays = []

        # ===== Continual learning metrics buffers =====
        self.seen_domains = []
        self.accuracy_history = defaultdict(list)
        self.aa_history = []

    def test(self):
        """测试循环"""
        print("testing...")
        self.agent.set_eval_mode()  # 设置为评估模式

        for episode in range(self.num_episodes):
            print(f"\n===== Test Episode {episode+1}/{self.num_episodes} =====")
            state = self.env.reset()
            total_reward = 0
            total_delay = 0

            for t in range(self.max_timesteps):
                current_domain = self.env.current_domain
                if current_domain not in self.seen_domains:
                    self.seen_domains.append(current_domain)

                # 选择动作（评估模式，不使用epsilon-greedy）
                action = self.agent.select_action(state)
                next_state, reward, done, info = self.env.step(action)

                total_reward += reward
                total_delay += info["comm"]["t_total_comm"]

                # ===== Evaluate model on all seen domains =====
                for domain in self.seen_domains:
                    acc = self.env.evaluate_model(domain)
                    self.accuracy_history[domain].append(acc)

                # ===== Compute incremental CL metrics =====
                metrics = IncrementalMetricsCalculator.compute_metrics(
                    self.seen_domains, self.accuracy_history
                )
                if "AA" in metrics:
                    self.aa_history.append(metrics["AA"])

                state = next_state

                if done:
                    break

            self.episode_rewards.append(total_reward)
            self.episode_delays.append(total_delay)

        # ===== Final CL metrics =====
        final_metrics = IncrementalMetricsCalculator.compute_metrics(
            self.seen_domains, self.accuracy_history
        )
        AIA = IncrementalMetricsCalculator.compute_aia(self.aa_history)

        # ===== System metrics =====
        mean_reward = np.mean(self.episode_rewards)
        mean_delay = np.mean(self.episode_delays)

        # ===== Output =====
        print("\n================ FINAL TEST RESULTS ================")

        print("\n[Layer-1 Continual Learning Metrics]")
        print(f"AA  = {final_metrics.get('AA', 0):.4f}")
        print(f"FM  = {final_metrics.get('FM', 0):.4f}")
        print(f"BWT = {final_metrics.get('BWT', 0):.4f}")
        print(f"AIA = {AIA:.4f}")

        print("\n[Layer-2 System Metrics]")
        print(f"Mean Reward = {mean_reward:.4f}")
        print(f"Mean Communication Delay = {mean_delay:.4f}")

        return self.episode_rewards


if __name__ == "__main__":
    tester = RLTester()
    tester.test()