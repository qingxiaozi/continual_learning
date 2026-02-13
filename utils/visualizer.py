import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime


class ResultVisualizer:
    """结果可视化类"""

    def __init__(self, save_dir="./results"):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        self.style_config = {
            'colors': {'AA': '#1f77b4', 'AIA': '#2ca02c', 'FM': '#d62728', 'BWT': '#ff7f0e'},
            'markers': {'AA': 'o', 'AIA': 's', 'FM': '^', 'BWT': 'D'},
            'linewidth': 2,
            'markersize': 8,
            'figsize': (10, 6),
            'dpi': 150,
        }

    def plot_training_loss(
        self,
        epoch_losses,
        val_losses=None,
        epoch_elastic_losses=None,
        save_plot=True,
        plot_name="training_loss.png"
    ):
        """绘制训练损失、验证损失和弹性损失曲线"""
        plt.figure(figsize=(10, 6))

        epochs = range(1, len(epoch_losses) + 1)
        plt.plot(epochs, epoch_losses, "b-", linewidth=2, label="Training Loss", marker='o', markersize=4)

        if val_losses and len(val_losses) > 0:
            val_epochs = range(1, len(val_losses) + 1)
            plt.plot(val_epochs, val_losses, "r-", linewidth=2, label="Validation Loss", marker='s', markersize=4)

        if epoch_elastic_losses and len(epoch_elastic_losses) > 0:
            elastic_epochs = range(1, len(epoch_elastic_losses) + 1)
            plt.plot(elastic_epochs, epoch_elastic_losses, "g-", linewidth=2, label="Elastic Loss", marker='^', markersize=4)

        # 动态设置标题
        labels = ["Training"]
        if val_losses and len(val_losses) > 0:
            labels.append("Validation")
        if epoch_elastic_losses and len(epoch_elastic_losses) > 0:
            labels.append("Elastic")
        title = " vs ".join(labels) + " Loss"
        plt.title(title, fontsize=14, fontweight="bold")

        plt.xlabel("Epoch", fontsize=12)
        plt.ylabel("Loss", fontsize=12)
        plt.legend(fontsize=10)
        plt.xticks(epochs)

        # 自动调整 y 轴范围
        all_losses = list(epoch_losses)
        if val_losses and len(val_losses) > 0:
            all_losses.extend(val_losses[:len(epoch_losses)])
        if epoch_elastic_losses and len(epoch_elastic_losses) > 0:
            all_losses.extend(epoch_elastic_losses[:len(epoch_losses)])

        if len(all_losses) > 1:
            loss_min, loss_max = min(all_losses), max(all_losses)
            margin = (loss_max - loss_min) * 0.1 or 0.1  # 避免除零
            plt.ylim(loss_min - margin, loss_max + margin)

        if save_plot:
            plot_path = os.path.join(self.save_dir, plot_name)
            plt.savefig(plot_path, dpi=300, bbox_inches="tight")
            print(f"训练损失图已保存至: {plot_path}")

        plt.close()

    def plot_data_heterogeneity(self, data_simulator, session, save_plot=True):
        """
        绘制数据异质性示意图
        """
        # 获取数据
        domain = data_simulator.get_current_domain()
        domain_key = f"{data_simulator.current_dataset}_{domain}"

        if domain_key not in data_simulator.vehicle_data_assignments:
            return

        num_classes = data_simulator.dataset_info[data_simulator.current_dataset]["num_classes"]
        vehicle_assignments = data_simulator.vehicle_data_assignments[domain_key]
        train_dataset = data_simulator.train_data_cache[domain_key]

        # 统计样本数量
        vehicle_class_counts = {}
        for vehicle_id, indices in vehicle_assignments.items():
            class_counts = [0] * num_classes
            for idx in indices:
                _, label = train_dataset[idx]
                class_counts[label] += 1
            vehicle_class_counts[vehicle_id] = class_counts

        # 准备绘图数据
        vehicle_ids, class_ids, sample_counts = [], [], []
        for vehicle_id in range(data_simulator.num_vehicles):
            if vehicle_id in vehicle_class_counts:
                for class_id in range(num_classes):
                    count = vehicle_class_counts[vehicle_id][class_id]
                    if count > 0:
                        vehicle_ids.append(vehicle_id)
                        class_ids.append(class_id)
                        sample_counts.append(count)

        if not sample_counts:
            return

        fig, ax = plt.subplots(figsize=(9, 6))
        fig.patch.set_facecolor('white')
        ax.set_facecolor('white')

        scatter = ax.scatter(
            vehicle_ids,
            class_ids,
            s=[count * 3 + 30 for count in sample_counts],
            c=sample_counts,
            cmap="Reds",
            alpha=0.7,
            edgecolors="darkred",
            linewidth=0.5,
        )

        # 图表设置
        ax.set_title(f"Data Distribution - Session {session}", fontsize=11, pad=8)
        ax.set_xlabel("Vehicle ID", fontsize=9)
        ax.set_ylabel("Class", fontsize=9)
        ax.set_xticks(range(data_simulator.num_vehicles))
        ax.set_yticks(range(num_classes))
        ax.set_yticklabels([f"C{i}" for i in range(num_classes)])
        ax.tick_params(axis='y', labelsize=8)

        cbar = fig.colorbar(scatter, ax=ax)
        cbar.set_label("Samples", fontsize=8)
        plt.tight_layout()

        if save_plot:
            plot_name = f"data_heterogeneity_session_{session}.png"
            plot_path = os.path.join(self.save_dir, plot_name)
            plt.savefig(plot_path, dpi=150, bbox_inches="tight", facecolor='white')
            print(f"数据异质性图已保存至{plot_path}")

        plt.close()

    def _print_heterogeneity_statistics(self, vehicle_class_counts, domain, session):
        """打印数据异质性统计信息"""
        print(f"\n=== Session {session} - {domain} 数据异质性统计 ===")

        total_samples = 0
        class_coverage = {}  # 每个类别被多少车辆覆盖

        for vehicle_id, class_counts in vehicle_class_counts.items():
            vehicle_total = sum(class_counts.values())
            total_samples += vehicle_total

            # 统计每个类别的覆盖情况
            for class_id, count in class_counts.items():
                if count > 0:
                    class_coverage[class_id] = class_coverage.get(class_id, 0) + 1

            print(
                f"车辆 {vehicle_id}: {vehicle_total} 个样本, 覆盖 {sum(1 for c in class_counts.values() if c > 0)} 个类别"
            )

        # 计算异质性指标
        vehicle_totals = [
            sum(counts.values()) for counts in vehicle_class_counts.values()
        ]
        heterogeneity_std = np.std(vehicle_totals) if vehicle_totals else 0

        print(f"\n总体统计:")
        print(f"总样本数: {total_samples}")
        print(f"平均每车样本数: {np.mean(vehicle_totals):.1f}")
        print(f"样本数标准差: {heterogeneity_std:.1f} (异质性指标)")
        print(
            f"类别覆盖情况: 平均每个类别被 {np.mean(list(class_coverage.values())):.1f} 辆车覆盖"
        )
        print("====================================\n")

    def plot_sample_count_evolution(self, session_history: list[dict[str, any]]) -> plt.figure:
        """
        绘制样本数随session的变化

        Args:
            session_history: 会话历史记录列表

        Returns:
            matplotlib Figure对象
        """
        if not session_history:
            print("没有数据可绘制")
            return None

        sessions = [r.get("session", i) for i, r in enumerate(session_history)]
        total_samples = [r.get("total_samples", 0) for r in session_history]

        fig, ax = plt.subplots(figsize=self.style_config['figsize'])

        # 绘制柱状图
        bars = ax.bar(sessions, total_samples,
                     color=self.style_config['colors']['samples'],
                     alpha=0.7,
                     edgecolor='black',
                     linewidth=1)

        ax.set_title('Sample Count Per Session', fontsize=14, fontweight='bold')
        ax.set_xlabel('Session', fontsize=12)
        ax.set_ylabel('Number of Samples', fontsize=12)

        # 在柱子上添加数值标签
        for bar in bars:
            height = bar.get_height()
            if height > 0:  # 只在有样本的时候显示
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{int(height):,}',
                       ha='center', va='bottom',
                       fontsize=9, fontweight='bold')

        # 添加累积样本数的折线图（次坐标轴）
        if len(total_samples) > 1:
            ax2 = ax.twinx()
            cumulative_samples = np.cumsum(total_samples)
            ax2.plot(sessions, cumulative_samples,
                    color='red',
                    marker='o',
                    linewidth=self.style_config['linewidth'],
                    markersize=self.style_config['markersize'],
                    label='Cumulative Samples')
            ax2.set_ylabel('Cumulative Samples', fontsize=12, color='red')
            ax2.tick_params(axis='y', labelcolor='red')

            # 合并图例
            lines, labels = ax.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax.legend(lines + lines2, labels + labels2, loc='upper left')
        else:
            ax.legend(['Samples per Session'], loc='upper left')

        plt.tight_layout()
        return fig

    def _add_trend_line(self, ax, x, y, color):
        """在图上添加趋势线"""
        if len(x) >= 3:
            z = np.polyfit(x, y, 1)
            p = np.poly1d(z)
            ax.plot(x, p(x), "--", color=color, alpha=0.5, linewidth=1.5)

            # 显示趋势斜率
            slope = z[0]
            trend_text = "↑ Improving" if slope > 0.001 else "↓ Declining" if slope < -0.001 else "→ Stable"
            ax.text(0.02, 0.95, f"Trend: {trend_text}",
                   transform=ax.transAxes,
                   fontsize=8,
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    def _load_if_exists(self, filename):
        path = os.path.join(self.save_dir, filename)
        return np.load(path, allow_pickle=True) if os.path.exists(path) else None

    def plot_cl_metric_curves(self, AA_steps, FM_steps, BWT_steps, filename="cl_metrics_curve.png"):
        if AA_steps is None:
            return
            
        x = np.arange(1, len(AA_steps[0]) + 1)
        plt.figure(figsize=self.style_config['figsize'])
        
        for name, data, color_key in [('AA', AA_steps, 'AA'), ('FM', FM_steps, 'FM'), ('BWT', BWT_steps, 'BWT')]:
            if data is None:
                continue
            data = np.array(data)
            mean = np.mean(data, axis=0)
            std = np.std(data, axis=0)
            plt.plot(x, mean, marker=self.style_config['markers'][color_key],
                     color=self.style_config['colors'][color_key], linewidth=2, label=name)
            plt.fill_between(x, mean - std, mean + std, alpha=0.2, color=self.style_config['colors'][color_key])
        
        plt.xlabel("Domain Index")
        plt.ylabel("Metric Value")
        plt.title("Continual Learning Metrics (Mean ± Std)")
        plt.legend()
        plt.grid(alpha=0.3)
        self._save_fig(filename)

    def plot_accuracy_matrix(self, acc_matrices, filename="accuracy_matrix_heatmap.png"):
        if acc_matrices is None or len(acc_matrices) == 0:
            print("No accuracy matrix data to plot.")
            return

        max_domains = max(len(mat) for mat in acc_matrices)
        full_mats = []

        for mat in acc_matrices:
            mat = np.array(mat)
            full = np.full((max_domains, max_domains), np.nan)
            for i, row in enumerate(mat):
                full[i, :len(row)] = row
            full_mats.append(full)

        mean_acc = np.nanmean(full_mats, axis=0)

        plt.figure(figsize=(6, 5))
        sns.heatmap(mean_acc, annot=True, fmt=".3f", cmap="viridis", cbar_kws={'label': 'Accuracy'})
        plt.xlabel("Test Domain")
        plt.ylabel("After Learning Domain")
        plt.title("Average Accuracy Matrix")
        self._save_fig(filename)

    def plot_episode_boxplot(self, AA, FM, BWT, AIA, filename="cl_metrics_boxplot.png"):
        data = [AA, FM, BWT, AIA]
        labels = ["AA", "FM", "BWT", "AIA"]
        plt.figure(figsize=(8, 6))
        plt.boxplot(data, labels=labels)
        plt.ylabel("Metric Value"); plt.title("Episode-level CL Metrics")
        plt.grid(alpha=0.3)
        self._save_fig(filename)

    def plot_episode_reward(self, rewards, filename="episode_reward.png"):
        plt.figure()
        plt.plot(rewards)
        plt.xlabel("Episode"); plt.ylabel("Total Reward")
        plt.title("Episode-level Total Reward")
        self._save_fig(filename)

    def plot_episode_delay(self, delays, filename="episode_delay.png"):
        plt.figure()
        plt.plot(delays)
        plt.xlabel("Episode"); plt.ylabel("Communication Delay (s)")
        plt.title("Episode-level Communication Delay")
        self._save_fig(filename)

    def _save_fig(self, filename):
        path = os.path.join(self.save_dir, filename)
        plt.savefig(path, dpi=300, bbox_inches='tight')
        print(f"Saved: {path}")
        plt.close()

    def visualize_all(self):
        """加载 results/*.npy 并生成所有可视化图表"""
        # 加载 CL 指标
        AA_steps = self._load_if_exists("AA_steps.npy")
        FM_steps = self._load_if_exists("FM_steps.npy")
        BWT_steps = self._load_if_exists("BWT_steps.npy")
        acc_matrices = self._load_if_exists("accuracy_matrices.npy")
        AA_all = self._load_if_exists("AA_all.npy")
        FM_all = self._load_if_exists("FM_all.npy")
        BWT_all = self._load_if_exists("BWT_all.npy")
        AIA_all = self._load_if_exists("AIA_all.npy")

        # 加载系统指标
        rewards = self._load_if_exists("episode_rewards.npy")
        delays = self._load_if_exists("episode_delays.npy")

        # 绘制 CL 曲线
        if AA_steps is not None:
            self.plot_cl_metric_curves(AA_steps, FM_steps, BWT_steps)
        
        # 绘制 Accuracy Matrix（取最后一个 episode 的矩阵）
        if acc_matrices is not None and len(acc_matrices) > 0:
            self.plot_accuracy_matrix(acc_matrices)

        # 绘制 Boxplot
        if AA_all is not None:
            self.plot_episode_boxplot(AA_all, FM_all, BWT_all, AIA_all)

        # 绘制系统指标
        if rewards is not None:
            self.plot_episode_reward(rewards)
        if delays is not None:
            self.plot_episode_delay(delays)

        print("All visualizations completed.")

if __name__ == "__main__":
    vis = ResultVisualizer()
    vis.visualize_all()