from collections import defaultdict
import torch
import itertools
import numpy as np
from config.parameters import Config
from experiment.rl_env import VehicleEdgeEnv
from models.drl_agent import DRLAgent
from utils.metrics import IncrementalMetricsCalculator
from utils.visualizer import ResultVisualizer


class RLTester:
    def __init__(self, model_path="./results/trained_drl_model.pth"):
        self.env = VehicleEdgeEnv()
        self.agent = DRLAgent(state_dim=4 * Config.NUM_VEHICLES)
        self.agent.load_model(model_path)
        self.agent.set_eval_mode()

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

            AA_curve, FM_curve, BWT_curve = [], [], []

            for t in range(self.max_timesteps):
                available_batches = state[-Config.NUM_VEHICLES:].astype(int).tolist()
                action = self.agent.select_action(state, available_batches=available_batches)
                next_state, reward, done, info = self.env.step(action)
                state = next_state

                total_reward += reward
                step_delays.append(info["comm"]["t_total_comm"])

                if (t + 1) % self.domain_interval == 0:
                    current_domain = self.env.current_domain
                    if current_domain not in seen_domains:
                        seen_domains.append(current_domain)
                    
                    for d in seen_domains:
                        acc = self.env.evaluate_model(d)
                        accuracy_history[d].append(acc)

                    row = [accuracy_history[d][-1] for d in seen_domains]
                    acc_matrix.append(row)

                    metrics = IncrementalMetricsCalculator.compute_metrics(
                        seen_domains, accuracy_history
                    )
                    aa_history.append(metrics["AA"])

                    AA_curve.append(metrics["AA"])
                    FM_curve.append(metrics["FM"])
                    BWT_curve.append(metrics["BWT"])

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

            self.AA_steps.append(AA_curve)
            self.FM_steps.append(FM_curve)
            self.BWT_steps.append(BWT_curve)
            self.accuracy_matrices.append(acc_matrix)

            # ===== System metrics =====
            self.episode_rewards.append(total_reward)
            self.episode_delays.append(np.mean(step_delays))

        self.report_results()
        self.save_results()

    def report_results(self):
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


    def save_results(self):
        np.save("results/AA_steps.npy", np.array(self.AA_steps))
        np.save("results/FM_steps.npy", np.array(self.FM_steps))
        np.save("results/BWT_steps.npy", np.array(self.BWT_steps))
        np.save("results/accuracy_matrices.npy", np.array(self.accuracy_matrices, dtype=object))
        np.save("results/AA_final.npy", np.array(self.AA_all))
        np.save("results/FM_final.npy", np.array(self.FM_all))
        np.save("results/BWT_final.npy", np.array(self.BWT_all))
        np.save("results/AIA_final.npy", np.array(self.AIA_all))
        np.save("results/episode_rewards.npy", np.array(self.episode_rewards))
        np.save("results/episode_delays.npy", np.array(self.episode_delays))
    

if __name__ == "__main__":
    tester = RLTester()
    tester.test()