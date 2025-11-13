import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import os


class ResultVisualizer:
    """结果可视化类"""

    def __init__(self, save_dir="./results"):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

        # 设置绘图风格
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")

    def plot_training_loss(self, epoch_losses, save_plot=True, plot_name="training_loss.png"):
        """绘制训练损失曲线"""
        plt.figure(figsize=(10, 6))

        # 绘制损失曲线
        epochs = range(1, len(epoch_losses) + 1)
        plt.plot(epochs, epoch_losses, 'b-', linewidth=2, label='Training Loss')
        plt.scatter(epochs, epoch_losses, color='red', s=30, zorder=5)

        # 设置图表属性
        plt.title('Training Loss vs Epochs', fontsize=14, fontweight='bold')
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Loss', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=10)

        # 设置x轴为整数
        plt.xticks(epochs)

        # 自动调整y轴范围，确保能看清下降趋势
        if len(epoch_losses) > 1:
            loss_range = max(epoch_losses) - min(epoch_losses)
            plt.ylim(min(epoch_losses) - 0.1 * loss_range,
                    max(epoch_losses) + 0.1 * loss_range)

        # 保存或显示图表
        if save_plot:
            plot_path = os.path.join(self.save_dir, plot_name)
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            print(f"训练损失图已保存至: {plot_path}")

        plt.close()

        # 打印训练总结
        if len(epoch_losses) > 1:
            self._print_training_summary(epoch_losses)

    def _print_training_summary(self, epoch_losses):
        """打印训练总结"""
        print(f"\n训练总结:")
        print(f"初始损失: {epoch_losses[0]:.4f}")
        print(f"最终损失: {epoch_losses[-1]:.4f}")
        print(f"损失下降: {epoch_losses[0] - epoch_losses[-1]:.4f}")
        print(f"下降百分比: {(epoch_losses[0] - epoch_losses[-1]) / epoch_losses[0] * 100:.2f}%")

    def calculate_metrics(self, performance_history, time_history=None, communication_costs=None):
        """计算各种评估指标"""
        metrics = {}

        if time_history is None:
            time_history = []
        if communication_costs is None:
            communication_costs = []

        # 模型性能指标
        accuracies = [perf['accuracy'] for perf in performance_history if 'accuracy' in perf]
        losses = [perf['loss'] for perf in performance_history if 'loss' in perf]

        if accuracies:
            metrics['final_accuracy'] = accuracies[-1]
            metrics['average_accuracy'] = np.mean(accuracies)
            metrics['min_accuracy'] = np.min(accuracies)
            metrics['max_accuracy'] = np.max(accuracies)

            # 计算准确率稳定性
            metrics['accuracy_std'] = np.std(accuracies)
        else:
            metrics.update({
                'final_accuracy': 0,
                'average_accuracy': 0,
                'min_accuracy': 0,
                'max_accuracy': 0,
                'accuracy_std': 0
            })

        # 遗忘度量
        if len(accuracies) > 1:
            forgetting = 0.0
            for i in range(1, len(accuracies)):
                forgetting += max(0, accuracies[i-1] - accuracies[i])
            metrics['forgetting'] = forgetting / (len(accuracies) - 1)
        else:
            metrics['forgetting'] = 0.0

        # 损失指标
        if losses:
            metrics['final_loss'] = losses[-1]
            metrics['average_loss'] = np.mean(losses)
            metrics['min_loss'] = np.min(losses)
            metrics['max_loss'] = np.max(losses)
        else:
            metrics.update({
                'final_loss': 0,
                'average_loss': 0,
                'min_loss': 0,
                'max_loss': 0
            })

        # 系统效率指标
        if communication_costs:
            metrics['total_communication_cost'] = np.sum(communication_costs)
            metrics['average_communication_cost'] = np.mean(communication_costs)
        else:
            metrics.update({
                'total_communication_cost': 0,
                'average_communication_cost': 0
            })

        if time_history:
            metrics['total_training_time'] = np.sum(time_history)
            metrics['average_time_per_session'] = np.mean(time_history)
            metrics['max_time_per_session'] = np.max(time_history)
        else:
            metrics.update({
                'total_training_time': 0,
                'average_time_per_session': 0,
                'max_time_per_session': 0
            })

        return metrics

    def plot_results(self, performance_history, algorithm_name, save_plot=True):
        """绘制结果图表"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

        # 准确率曲线
        sessions = range(len(performance_history))
        accuracies = [perf.get('accuracy', 0) for perf in performance_history]
        losses = [perf.get('loss', 0) for perf in performance_history]

        ax1.plot(sessions, accuracies, 'b-', linewidth=2, marker='o')
        ax1.set_xlabel('Training Session')
        ax1.set_ylabel('Accuracy')
        ax1.set_title(f'{algorithm_name} - Model Accuracy Over Time')
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0, 1)  # 准确率范围0-1

        # 损失曲线
        ax2.plot(sessions, losses, 'r-', linewidth=2, marker='s')
        ax2.set_xlabel('Training Session')
        ax2.set_ylabel('Loss')
        ax2.set_title(f'{algorithm_name} - Model Loss Over Time')
        ax2.grid(True, alpha=0.3)

        # 置信度分布
        confidences = []
        for perf in performance_history:
            if 'confidence' in perf and perf['confidence']:
                confidences.extend(perf['confidence'])

        if confidences:
            ax3.hist(confidences, bins=20, alpha=0.7, edgecolor='black', color='green')
            ax3.set_xlabel('Confidence')
            ax3.set_ylabel('Frequency')
            ax3.set_title(f'{algorithm_name} - Confidence Distribution')
            ax3.grid(True, alpha=0.3)
        else:
            ax3.text(0.5, 0.5, 'No confidence data',
                    horizontalalignment='center', verticalalignment='center',
                    transform=ax3.transAxes, fontsize=12)
            ax3.set_title(f'{algorithm_name} - Confidence Distribution')

        # 缓存使用情况
        cache_sizes = []
        for perf in performance_history:
            if 'cache_stats' in perf and perf['cache_stats']:
                total_size = sum(stats.get('total_size', 0) for stats in perf['cache_stats'].values())
                cache_sizes.append(total_size)

        if cache_sizes:
            ax4.plot(range(len(cache_sizes)), cache_sizes, 'g-', linewidth=2, marker='^')
            ax4.set_xlabel('Training Session')
            ax4.set_ylabel('Total Cache Size')
            ax4.set_title(f'{algorithm_name} - Cache Usage Over Time')
            ax4.grid(True, alpha=0.3)
        else:
            ax4.text(0.5, 0.5, 'No cache data',
                    horizontalalignment='center', verticalalignment='center',
                    transform=ax4.transAxes, fontsize=12)
            ax4.set_title(f'{algorithm_name} - Cache Usage Over Time')

        plt.tight_layout()

        if save_plot:
            plot_path = os.path.join(self.save_dir, f"{algorithm_name}_results.png")
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            print(f"结果图表已保存至: {plot_path}")

        plt.show()
        return fig

    def plot_comparison(self, algorithms_results, save_plot=True):
        """比较不同算法的性能"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.flatten()

        # 准确率比较
        for algo_name, results in algorithms_results.items():
            accuracies = [perf.get('accuracy', 0) for perf in results['performance_history']]
            sessions = range(len(accuracies))
            axes[0].plot(sessions, accuracies, label=algo_name, linewidth=2)

        axes[0].set_xlabel('Training Session')
        axes[0].set_ylabel('Accuracy')
        axes[0].set_title('Accuracy Comparison')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # 损失比较
        for algo_name, results in algorithms_results.items():
            losses = [perf.get('loss', 0) for perf in results['performance_history']]
            sessions = range(len(losses))
            axes[1].plot(sessions, losses, label=algo_name, linewidth=2)

        axes[1].set_xlabel('Training Session')
        axes[1].set_ylabel('Loss')
        axes[1].set_title('Loss Comparison')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        # 通信成本比较
        algo_names = []
        comm_costs = []
        for algo_name, results in algorithms_results.items():
            if 'communication_costs' in results:
                algo_names.append(algo_name)
                comm_costs.append(np.sum(results['communication_costs']))

        if comm_costs:
            bars = axes[2].bar(algo_names, comm_costs, alpha=0.7)
            axes[2].set_xlabel('Algorithm')
            axes[2].set_ylabel('Total Communication Cost')
            axes[2].set_title('Communication Cost Comparison')
            # 在柱状图上显示数值
            for bar, cost in zip(bars, comm_costs):
                axes[2].text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                           f'{cost:.0f}', ha='center', va='bottom')

        # 训练时间比较
        algo_names = []
        train_times = []
        for algo_name, results in algorithms_results.items():
            if 'time_history' in results and results['time_history']:
                algo_names.append(algo_name)
                train_times.append(np.sum(results['time_history']))

        if train_times:
            bars = axes[3].bar(algo_names, train_times, alpha=0.7, color='orange')
            axes[3].set_xlabel('Algorithm')
            axes[3].set_ylabel('Total Training Time (s)')
            axes[3].set_title('Training Time Comparison')
            # 在柱状图上显示数值
            for bar, time_val in zip(bars, train_times):
                axes[3].text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                           f'{time_val:.1f}s', ha='center', va='bottom')

        plt.tight_layout()

        if save_plot:
            plot_path = os.path.join(self.save_dir, "algorithm_comparison.png")
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            print(f"算法比较图已保存至: {plot_path}")

        plt.show()
        return fig

    def print_detailed_metrics(self, metrics, algorithm_name):
        """打印详细的指标报告"""
        print(f"\n{'='*50}")
        print(f"{algorithm_name} 详细指标报告")
        print(f"{'='*50}")

        print(f"\n📊 模型性能指标:")
        print(f"  最终准确率: {metrics.get('final_accuracy', 0):.4f}")
        print(f"  平均准确率: {metrics.get('average_accuracy', 0):.4f}")
        print(f"  准确率范围: {metrics.get('min_accuracy', 0):.4f} - {metrics.get('max_accuracy', 0):.4f}")
        print(f"  准确率标准差: {metrics.get('accuracy_std', 0):.4f}")
        print(f"  遗忘度量: {metrics.get('forgetting', 0):.4f}")

        print(f"\n⚡ 系统效率指标:")
        print(f"  总通信成本: {metrics.get('total_communication_cost', 0):.0f}")
        print(f"  平均通信成本: {metrics.get('average_communication_cost', 0):.2f}")
        print(f"  总训练时间: {metrics.get('total_training_time', 0):.2f}s")
        print(f"  平均每轮时间: {metrics.get('average_time_per_session', 0):.2f}s")

        print(f"\n📈 损失指标:")
        print(f"  最终损失: {metrics.get('final_loss', 0):.4f}")
        print(f"  平均损失: {metrics.get('average_loss', 0):.4f}")
        print(f"  损失范围: {metrics.get('min_loss', 0):.4f} - {metrics.get('max_loss', 0):.4f}")