import torch
import wandb
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

        wandb.init(
            project="Vehicle-Edge-DRL", 
            name=f"Run_Ep{Config.NUM_EPISODES}",
            config={
                "num_episodes": self.num_episodes,
                "max_timesteps": self.max_timesteps,
                "state_dim": 4 * Config.NUM_VEHICLES,
                "learning_rate": Config.DRL_LEARNING_RATE,
                "gamma": Config.DRL_GAMMA,
                "batch_size": Config.DRL_BATCH_SIZE,
                "epsilon_start": Config.DRL_EPSILON_START,
                "epsilon_end": Config.DRL_EPSILON_END,
                "epsilon_decay": Config.DRL_EPSILON_DECAY,
                "target_update_interval": Config.TARGET_UPDATE_INTERVAL,
                "buffer_size": Config.DRL_BUFFER_SIZE,
            }
        )
        print(f"WandB 监控已启动: {wandb.run.url}")

    def train(self):
        """训练循环"""
        global_step = 0  # 全局步数，用于 X 轴对齐
        
        for episode in range(self.num_episodes):
            state = self.env.reset()
            total_reward = 0.0
            total_delay = 0.0

            for t in range(self.max_timesteps):
                available_batches = state[-Config.NUM_VEHICLES:].astype(int).tolist()
                # 选择动作
                action = self.agent.select_action(state, available_batches)
                
                # 执行环境中的动作
                next_state, reward, done, info = self.env.step(action)
                
                # 存储经验
                self.agent.store_experience(state, action, reward, next_state, done)

                # 优化模型
                loss = self.agent.optimize_model()

                if loss is not None:
                    wandb.log({
                        "loss/raw": loss,
                        "step/global": global_step
                    })

                total_reward += reward
                if 'comm' in info:
                    total_delay += info['comm'].get('t_total_comm', 0.0)

                global_step += 1
                state = next_state
                if done:
                    break

            # 记录 Episode 级汇总 (仅 Reward, Delay, Epsilon)
            wandb.log({
                "episode": episode + 1,
                "reward/total": total_reward,
                "metrics/avg_delay": total_delay / max(t + 1, 1),
                "params/epsilon": self.agent._get_epsilon(),
                "step/global": global_step
            })

            # 下面两行可以不要，先放着吧
            self.episode_rewards.append(total_reward)
            self.visualizer.plot_training_loss(self.episode_rewards)

            # 定期更新目标网络
            if episode % Config.TARGET_UPDATE_INTERVAL == 0:
                self.agent.hard_update_target_network()
                print(f"已更新目标网络 (Episode {episode+1})")

            print(f"Episode {episode + 1}/{self.num_episodes}, Total Reward: {total_reward:.4f}")

        # 训练完成后保存模型
        self.agent.save_model("./results/pth/trained_drl_model.pth")
        artifact = wandb.Artifact("final-drl-model", type="model")
        artifact.add_file("./results/pth/trained_drl_model.pth")
        wandb.log_artifact(artifact)
        wandb.finish()

        return self.episode_rewards


if __name__ == "__main__":
    trainer = RLTrainer()
    trainer.train()