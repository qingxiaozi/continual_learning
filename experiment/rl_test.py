from collections import defaultdict
import torch
import itertools
import numpy as np
from config.parameters import Config
from experiment.rl_env import VehicleEdgeEnv
from models.drl_agent import DRLAgent
from utils.metrics import IncrementalMetricsCalculator, ResultVisualizer


class RLTester:
    def __init__(self, model_path="./results/trained_drl_model.pth"):
        self.env = VehicleEdgeEnv()
        self.agent = DRLAgent(state_dim=4 * Config.NUM_VEHICLES)
        self.agent.load_model(model_path)
        self.agent.set_eval_mode()
        self.visualizer = ResultVisualizer()
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

        # Accuracy matrix buffer (paper visualization)
        self.accuracy_matrices = []

    def test(self):
        """测试循环"""
        print("testing...")

        for episode in range(self.num_episodes):
            print(f"\n===== Test Episode {episode+1}/{self.num_episodes} =====")
            state = self.env.reset()
            total_reward = 0
            step_delays = []
            seen_domains = []
            accuracy_history = defaultdict(list)
            aa_history = []
            acc_matrix = []

            for t in range(self.max_timesteps):
                action = self.agent.select_action(state)
                next_state, reward, done, info = self.env.step(action)
                total_reward += reward
                step_delays.append(info["comm"]["t_total_comm"])
                state = next_state

                if (t + 1) % self.domain_interval == 0:
                    current_domain = self.env.current_domain
                    if current_domain not in seen_domains:
                        seen_domains.append(current_domain)
                    self.evaluate_all_domains(seen_domains, accuracy_history)

                    row = [accuracy_history[d][-1] for d in seen_domains]
                    acc_matrix.append(row)

                    metrics = IncrementalMetricsCalculator.compute_metrics(
                        seen_domains, accuracy_history
                    )
                    aa_history.append(metrics["AA"])

                if done:
                    break

            final_metrics = IncrementalMetricsCalculator.compute_metrics(
                seen_domains, accuracy_history
            )
            AIA = IncrementalMetricsCalculator.compute_aia(aa_history)

            self.AA_all.append(final_metrics["AA"])
            self.FM_all.append(final_metrics["FM"])
            self.BWT_all.append(final_metrics["BWT"])
            self.AIA_all.append(AIA)
            self.accuracy_matrices.append(acc_matrix)

            # ===== System metrics =====
            self.episode_rewards.append(total_reward)
            self.episode_delays.append(np.mean(step_delays))


        print("\n================ FINAL PAPER RESULTS ================")

        def report(name, values):
            print(f"{name}: {np.mean(values):.4f} ± {np.std(values):.4f}")

        print("\n[Continual Learning Metrics]")
        report("AA", self.AA_all)
        report("FM", self.FM_all)
        report("BWT", self.BWT_all)
        report("AIA", self.AIA_all)

        print("\n[System Metrics]")
        report("Mean Reward", self.episode_rewards)
        report("Mean Communication Delay", self.episode_delays)

    def evaluate_all_domains(self, seen_domains, accuracy_history):
        """
        Evaluate model on all seen domains
        """
        for d in seen_domains:
            acc = self.env.evaluate_model(d)
            accuracy_history[d].append(acc)

if __name__ == "__main__":
    tester = RLTester()
    tester.test()