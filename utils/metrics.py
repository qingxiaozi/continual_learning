import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import os
import time
from datetime import datetime


class ResultVisualizer:
    """结果可视化类"""

    def __init__(self, save_dir="./results"):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

        # 设置绘图风格
        self.style_config = {
            'colors': {
                'AA': '#1f77b4',      # 蓝色
                'AIA': '#2ca02c',     # 绿色
                'FM': '#d62728',      # 红色
                'BWT': '#ff7f0e',     # 橙色
                'accuracy': '#9467bd', # 紫色
                'delay': '#8c564b',    # 棕色
                'samples': '#17becf',  # 青色
                'loss': '#e377c2',     # 粉色
            },
            'markers': {
                'AA': 'o',
                'AIA': 's',
                'FM': '^',
                'BWT': 'D',
                'accuracy': 'v',
                'current_accuracy': '*',
            },
            'linewidth': 2,
            'markersize': 8,
            'figsize': (10, 6),
            'dpi': 100,
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

    def plot_core_metrics(self, session_history: list[dict[str, any]]) -> plt.figure:
        """
        绘制四个核心连续学习指标的变化

        Args:
            session_history: 会话历史记录列表

        Returns:
            matplotlib Figure对象
        """
        if len(session_history) < 2:
            print("需要至少2个session的数据来绘制趋势")
            return None

        sessions = [r.get("session", i) for i, r in enumerate(session_history)]

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle('Core Continuous Learning Metrics Evolution', fontsize=14, fontweight='bold')

        # AA - 平均准确率
        aa_values = [r.get("AA", 0) for r in session_history]
        ax1.plot(sessions, aa_values,
                marker=self.style_config['markers']['AA'],
                color=self.style_config['colors']['AA'],
                linewidth=self.style_config['linewidth'],
                markersize=self.style_config['markersize'],
                label='Average Accuracy (AA)')
        ax1.set_title('Average Accuracy (AA)', fontsize=12, fontweight='bold')
        ax1.set_xlabel('Session')
        ax1.set_ylabel('Accuracy')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        self._add_trend_line(ax1, sessions, aa_values, self.style_config['colors']['AA'])

        # AIA - 平均增量准确率
        aia_values = [r.get("AIA", 0) for r in session_history]
        ax2.plot(sessions, aia_values,
                marker=self.style_config['markers']['AIA'],
                color=self.style_config['colors']['AIA'],
                linewidth=self.style_config['linewidth'],
                markersize=self.style_config['markersize'],
                label='Average Incremental Accuracy (AIA)')
        ax2.set_title('Average Incremental Accuracy (AIA)', fontsize=12, fontweight='bold')
        ax2.set_xlabel('Session')
        ax2.set_ylabel('Accuracy')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        self._add_trend_line(ax2, sessions, aia_values, self.style_config['colors']['AIA'])

        # FM - 遗忘度量
        fm_values = [r.get("FM", 0) for r in session_history]
        ax3.plot(sessions, fm_values,
                marker=self.style_config['markers']['FM'],
                color=self.style_config['colors']['FM'],
                linewidth=self.style_config['linewidth'],
                markersize=self.style_config['markersize'],
                label='Forgetting Measure (FM)')
        ax3.set_title('Forgetting Measure (FM)', fontsize=12, fontweight='bold')
        ax3.set_xlabel('Session')
        ax3.set_ylabel('Forgetting (lower is better)')
        ax3.grid(True, alpha=0.3)
        ax3.legend()
        self._add_trend_line(ax3, sessions, fm_values, self.style_config['colors']['FM'])

        # BWT - 反向迁移
        bwt_values = [r.get("BWT", 0) for r in session_history]
        ax4.plot(sessions, bwt_values,
                marker=self.style_config['markers']['BWT'],
                color=self.style_config['colors']['BWT'],
                linewidth=self.style_config['linewidth'],
                markersize=self.style_config['markersize'],
                label='Backward Transfer (BWT)')
        ax4.set_title('Backward Transfer (BWT)', fontsize=12, fontweight='bold')
        ax4.set_xlabel('Session')
        ax4.set_ylabel('Transfer (higher is better)')
        ax4.grid(True, alpha=0.3)
        ax4.legend()
        self._add_trend_line(ax4, sessions, bwt_values, self.style_config['colors']['BWT'])

        plt.tight_layout()
        return fig

    def plot_accuracy_evolution(self, session_history: list[dict[str, any]]) -> plt.figure:
        """
        绘制准确率随session的变化

        Args:
            session_history: 会话历史记录列表

        Returns:
            matplotlib Figure对象
        """
        if len(session_history) < 2:
            print("需要至少2个session的数据来绘制趋势")
            return None

        sessions = [r.get("session", i) for i, r in enumerate(session_history)]

        fig, ax = plt.subplots(figsize=self.style_config['figsize'])

        # 当前域的准确率
        current_acc = [r.get("current_domain_accuracy", 0) for r in session_history]
        ax.plot(sessions, current_acc,
               marker=self.style_config['markers']['current_accuracy'],
               color=self.style_config['colors']['accuracy'],
               linewidth=self.style_config['linewidth'],
               markersize=self.style_config['markersize'],
               label='Current Domain Accuracy',
               alpha=0.8)

        # 平均准确率 (AA) 作为参考
        aa_values = [r.get("AA", 0) for r in session_history]
        ax.plot(sessions, aa_values,
               marker=self.style_config['markers']['AA'],
               color=self.style_config['colors']['AA'],
               linewidth=self.style_config['linewidth'],
               markersize=self.style_config['markersize'] - 2,
               label='Average Accuracy (AA)',
               linestyle='--',
               alpha=0.7)

        ax.set_title('Accuracy Evolution Across Sessions', fontsize=14, fontweight='bold')
        ax.set_xlabel('Session', fontsize=12)
        ax.set_ylabel('Accuracy', fontsize=12)
        ax.set_ylim([-0.05, 1.05])
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best')

        # 添加最后的准确率值标注
        if current_acc:
            ax.annotate(f'{current_acc[-1]:.3f}',
                       xy=(sessions[-1], current_acc[-1]),
                       xytext=(10, -10),
                       textcoords='offset points',
                       fontsize=10,
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))

        plt.tight_layout()
        return fig

    def plot_communication_delay(self, session_history: list[dict[str, any]]) -> plt.figure:
        """
        绘制通信时延随session的变化

        Args:
            session_history: 会话历史记录列表

        Returns:
            matplotlib Figure对象
        """
        if len(session_history) < 2:
            print("需要至少2个session的数据来绘制趋势")
            return None

        sessions = [r.get("session", i) for i, r in enumerate(session_history)]
        delays = [r.get("total_delay", 0) for r in session_history]

        fig, ax = plt.subplots(figsize=self.style_config['figsize'])

        # 使用颜色编码表示延迟变化
        colors = []
        for i, delay in enumerate(delays):
            if i == 0:
                colors.append(self.style_config['colors']['delay'])
            else:
                if delay <= delays[i-1]:
                    colors.append('#2ca02c')  # 绿色，延迟降低
                else:
                    colors.append('#d62728')  # 红色，延迟增加

        bars = ax.bar(sessions, delays,
                     color=colors,
                     alpha=0.7,
                     edgecolor='black',
                     linewidth=1)

        ax.set_title('Communication Delay Per Session', fontsize=14, fontweight='bold')
        ax.set_xlabel('Session', fontsize=12)
        ax.set_ylabel('Total Delay (ms)', fontsize=12)

        # 在柱子上添加数值标签
        for bar, delay in zip(bars, delays):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{delay:.1f}',
                   ha='center', va='bottom',
                   fontsize=9, fontweight='bold')

        # 添加平均线和趋势线
        avg_delay = np.mean(delays) if delays else 0
        ax.axhline(y=avg_delay, color='blue', linestyle='--', alpha=0.5,
                  label=f'Average: {avg_delay:.1f} ms')

        if len(delays) >= 3:
            z = np.polyfit(sessions, delays, 1)
            p = np.poly1d(z)
            ax.plot(sessions, p(sessions), color='black', linestyle=':', linewidth=2,
                   label=f'Trend (slope: {z[0]:.2f})')

        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        return fig

    def plot_training_loss_evolution(self, session_history: list[dict[str, any]]) -> plt.figure:
        """
        绘制训练损失随session的变化

        Args:
            session_history: 会话历史记录列表

        Returns:
            matplotlib Figure对象
        """
        if len(session_history) < 2:
            print("需要至少2个session的数据来绘制趋势")
            return None

        sessions = [r.get("session", i) for i, r in enumerate(session_history)]

        fig, ax = plt.subplots(figsize=self.style_config['figsize'])

        # 获取各种损失
        loss_before = [r.get("loss_before", 0) for r in session_history]
        loss_after = [r.get("loss_after", 0) for r in session_history]
        training_loss = [r.get("training_loss", 0) for r in session_history]

        # 绘制三条损失曲线
        ax.plot(sessions, loss_before,
               marker='o',
               color='gray',
               linewidth=self.style_config['linewidth'] - 0.5,
               markersize=self.style_config['markersize'] - 2,
               label='Loss Before Training',
               alpha=0.7,
               linestyle=':')

        ax.plot(sessions, loss_after,
               marker='s',
               color='blue',
               linewidth=self.style_config['linewidth'],
               markersize=self.style_config['markersize'],
               label='Loss After Training')

        ax.plot(sessions, training_loss,
               marker='^',
               color='red',
               linewidth=self.style_config['linewidth'],
               markersize=self.style_config['markersize'],
               label='Training Loss',
               alpha=0.8)

        ax.set_title('Training Loss Evolution', fontsize=14, fontweight='bold')
        ax.set_xlabel('Session', fontsize=12)
        ax.set_ylabel('Loss', fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best')

        # 计算并显示损失降低的比例
        if loss_before and loss_after:
            improvement = [(before - after) / before * 100 if before > 0 else 0
                          for before, after in zip(loss_before, loss_after)]

            # 在图上标注最后的改进
            if improvement[-1] > 0:
                ax.annotate(f'Improvement: {improvement[-1]:.1f}%',
                           xy=(sessions[-1], loss_after[-1]),
                           xytext=(20, 20),
                           textcoords='offset points',
                           fontsize=9,
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgreen', alpha=0.7),
                           arrowprops=dict(arrowstyle='->', color='green'))

        plt.tight_layout()
        return fig

    def save_all_plots(self, session_history: list[dict[str, any]], prefix: str = ""):
            """
            保存所有可视化图表

            Args:
                session_history: 会话历史记录列表
                prefix: 文件名前缀
            """
            import os

            # 创建保存目录
            os.makedirs(self.save_dir, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            # 生成所有图表并保存
            plots = [
                (self.plot_core_metrics, "core_metrics"),
                (self.plot_accuracy_evolution, "accuracy_evolution"),
                (self.plot_sample_count_evolution, "sample_count"),
                (self.plot_communication_delay, "communication_delay"),
                (self.plot_training_loss_evolution, "training_loss"),
            ]

            saved_files = []
            for plot_func, name in plots:
                fig = plot_func(session_history)
                if fig is not None:
                    filename = f"{prefix}{name}_session{session_history[-1]['session']}_{timestamp}.png"
                    filepath = os.path.join(self.save_dir, filename)
                    fig.savefig(filepath, dpi=150, bbox_inches='tight')
                    plt.close(fig)
                    saved_files.append(filepath)
                    print(f"Saved: {filename}")

            return saved_files