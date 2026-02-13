import torch
import numpy as np
from config.parameters import Config
from experiment.rl_env import VehicleEdgeEnv
from models.drl_agent import DRLAgent
from utils.visualizer import ResultVisualizer


class RLTrainer:
    def __init__(self):
        self.env = VehicleEdgeEnv(mode="train")
        self.agent = DRLAgent(state_dim=4 * Config.NUM_VEHICLES)
        self.visualizer = ResultVisualizer()
        self.num_episodes = Config.NUM_EPISODES
        self.max_timesteps = Config.NUM_TRAINING_SESSIONS
        self.episode_rewards = []

    def train(self):
        """训练循环"""
        for episode in range(self.num_episodes):
            state = self.env.reset()
            total_reward = 0

            for t in range(self.max_timesteps):
                available_batches = state[-Config.NUM_VEHICLES:].astype(int).tolist()
                # 选择动作
                action = self.agent.select_action(state, available_batches)
                
                # 执行环境中的动作
                next_state, reward, done, _ = self.env.step(action)
                
                # 存储经验
                self.agent.store_experience(state, action, reward, next_state, done)

                # 优化模型
                loss = self.agent.optimize_model()

                total_reward += reward
                state = next_state

                if done:
                    break

            self.episode_rewards.append(total_reward)
            self.visualizer.plot_training_loss(self.episode_rewards)

            # 定期更新目标网络
            if episode % Config.TARGET_UPDATE_INTERVAL == 0:
                self.agent.hard_update_target_network()
                print(f"已更新目标网络 (Episode {episode+1})")

            print(f"Episode {episode + 1}/{self.num_episodes}, Total Reward: {total_reward:.4f}")

        # 训练完成后保存模型
        self.agent.save_model("./results/trained_drl_model.pth")

        return self.episode_rewards


if __name__ == "__main__":
    trainer = RLTrainer()
    trainer.train()