import torch
import numpy as np
from config.parameters import Config
from environment.rl_env import VehicleEdgeEnv
from models.drl_agent import DRLAgent
from utils.metrics import ResultVisualizer


class RLTester:
    def __init__(self, model_path="./results/trained_drl_model.pth"):
        self.env = VehicleEdgeEnv()
        self.agent = DRLAgent(state_dim=self.env.config.STATE_DIM)
        self.agent.load_model(model_path)
        self.visualizer = ResultVisualizer()
        self.num_episodes = Config.NUM_TEST_EPISODES
        self.max_timesteps = Config.NUM_TRAINING_SESSIONS
        self.episode_rewards = []

    def test(self):
        """测试循环"""
        self.agent.set_eval_mode()  # 设置为评估模式

        for episode in range(self.num_episodes):
            state = self.env.reset()
            total_reward = 0

            for t in range(self.max_timesteps):
                # 选择动作（评估模式，不使用epsilon-greedy）
                action = self.agent.select_action(state)

                # 执行环境中的动作
                next_state, reward, done, _ = self.env.step(action)

                total_reward += reward
                state = next_state

                if done:
                    break

            self.episode_rewards.append(total_reward)
            self.visualizer.plot_test_results(self.episode_rewards)

            print(f"Test Episode {episode + 1}/{self.num_episodes}, Total Reward: {total_reward:.4f}")

        # 输出最终测试的平均奖励
        avg_reward = np.mean(self.episode_rewards)
        print(f"测试完成，平均奖励: {avg_reward:.4f}")

        return self.episode_rewards


if __name__ == "__main__":
    tester = RLTester()
    tester.test()