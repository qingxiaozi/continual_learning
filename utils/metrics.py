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
        plt.style.use("seaborn-v0_8")
        sns.set_palette("husl")

    def plot_training_loss(
        self, epoch_losses, val_losses=None, save_plot=True, plot_name="training_loss.png"
    ):
        """绘制训练损失和验证损失曲线"""
        plt.figure(figsize=(10, 6))

        # 绘制训练损失曲线
        epochs = range(1, len(epoch_losses) + 1)
        plt.plot(epochs, epoch_losses, "b-", linewidth=2, label="Training Loss", marker='o', markersize=4)

        # 如果提供了验证损失，绘制验证损失曲线
        if val_losses and len(val_losses) > 0:
            val_epochs = range(1, len(val_losses) + 1)
            plt.plot(val_epochs, val_losses, "r-", linewidth=2, label="Validation Loss", marker='s', markersize=4)

        # 设置图表属性
        title = "Training Loss vs Epochs" if not val_losses else "Training and Validation Loss"
        plt.title(title, fontsize=14, fontweight="bold")
        plt.xlabel("Epoch", fontsize=12)
        plt.ylabel("Loss", fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=10)

        # 设置x轴为整数
        plt.xticks(epochs)

        # 自动调整y轴范围
        if len(epoch_losses) > 1:
            all_losses = epoch_losses
            if val_losses:
                all_losses = epoch_losses + val_losses[:len(epoch_losses)]  # 只取与训练损失对应长度的验证损失
            loss_range = max(all_losses) - min(all_losses)
            plt.ylim(
                min(all_losses) - 0.1 * loss_range,
                max(all_losses) + 0.1 * loss_range,
            )

        # 保存或显示图表
        if save_plot:
            plot_path = os.path.join(self.save_dir, plot_name)
            plt.savefig(plot_path, dpi=300, bbox_inches="tight")
            print(f"训练损失图已保存至: {plot_path}")

        plt.close()

        # # 打印训练总结
        # if len(epoch_losses) > 1:
        #     self._print_training_summary(epoch_losses, val_losses)

    def _print_training_summary(self, train_losses, val_losses=None):
        """打印训练总结"""
        print("\n训练总结:")
        print(f"总训练轮次: {len(train_losses)}")
        print(f"最终训练损失: {train_losses[-1]:.6f}")

        if train_losses[0] > 0:
            improvement = (train_losses[0] - train_losses[-1]) / train_losses[0] * 100
            print(f"训练损失改善: {improvement:.2f}%")

        if val_losses and len(val_losses) > 0:
            print(f"最终验证损失: {val_losses[-1]:.6f}")

            # 找出最佳验证损失及其对应的epoch
            best_val_epoch = val_losses.index(min(val_losses)) + 1
            best_val_loss = min(val_losses)
            print(f"最佳验证损失: {best_val_loss:.6f} (第 {best_val_epoch} 轮)")

            if val_losses[0] > 0:
                val_improvement = (val_losses[0] - val_losses[-1]) / val_losses[0] * 100
                print(f"验证损失改善: {val_improvement:.2f}%")

    def calculate_metrics(
        self, performance_history, time_history=None, communication_costs=None
    ):
        """计算各种评估指标"""
        metrics = {}

        if time_history is None:
            time_history = []
        if communication_costs is None:
            communication_costs = []

        # 模型性能指标
        accuracies = [
            perf["accuracy"] for perf in performance_history if "accuracy" in perf
        ]
        losses = [perf["loss"] for perf in performance_history if "loss" in perf]

        if accuracies:
            metrics["final_accuracy"] = accuracies[-1]
            metrics["average_accuracy"] = np.mean(accuracies)
            metrics["min_accuracy"] = np.min(accuracies)
            metrics["max_accuracy"] = np.max(accuracies)

            # 计算准确率稳定性
            metrics["accuracy_std"] = np.std(accuracies)
        else:
            metrics.update(
                {
                    "final_accuracy": 0,
                    "average_accuracy": 0,
                    "min_accuracy": 0,
                    "max_accuracy": 0,
                    "accuracy_std": 0,
                }
            )

        # 遗忘度量
        if len(accuracies) > 1:
            forgetting = 0.0
            for i in range(1, len(accuracies)):
                forgetting += max(0, accuracies[i - 1] - accuracies[i])
            metrics["forgetting"] = forgetting / (len(accuracies) - 1)
        else:
            metrics["forgetting"] = 0.0

        # 损失指标
        if losses:
            metrics["final_loss"] = losses[-1]
            metrics["average_loss"] = np.mean(losses)
            metrics["min_loss"] = np.min(losses)
            metrics["max_loss"] = np.max(losses)
        else:
            metrics.update(
                {"final_loss": 0, "average_loss": 0, "min_loss": 0, "max_loss": 0}
            )

        # 系统效率指标
        if communication_costs:
            metrics["total_communication_cost"] = np.sum(communication_costs)
            metrics["average_communication_cost"] = np.mean(communication_costs)
        else:
            metrics.update(
                {"total_communication_cost": 0, "average_communication_cost": 0}
            )

        if time_history:
            metrics["total_training_time"] = np.sum(time_history)
            metrics["average_time_per_session"] = np.mean(time_history)
            metrics["max_time_per_session"] = np.max(time_history)
        else:
            metrics.update(
                {
                    "total_training_time": 0,
                    "average_time_per_session": 0,
                    "max_time_per_session": 0,
                }
            )

        return metrics

    def plot_results(self, performance_history, algorithm_name, save_plot=True):
        """绘制结果图表"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

        # 准确率曲线
        sessions = range(len(performance_history))
        accuracies = [perf.get("accuracy", 0) for perf in performance_history]
        losses = [perf.get("loss", 0) for perf in performance_history]

        ax1.plot(sessions, accuracies, "b-", linewidth=2, marker="o")
        ax1.set_xlabel("Training Session")
        ax1.set_ylabel("Accuracy")
        ax1.set_title(f"{algorithm_name} - Model Accuracy Over Time")
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0, 1)  # 准确率范围0-1

        # 损失曲线
        ax2.plot(sessions, losses, "r-", linewidth=2, marker="s")
        ax2.set_xlabel("Training Session")
        ax2.set_ylabel("Loss")
        ax2.set_title(f"{algorithm_name} - Model Loss Over Time")
        ax2.grid(True, alpha=0.3)

        # 置信度分布
        confidences = []
        for perf in performance_history:
            if "confidence" in perf and perf["confidence"]:
                confidences.extend(perf["confidence"])

        if confidences:
            ax3.hist(confidences, bins=20, alpha=0.7, edgecolor="black", color="green")
            ax3.set_xlabel("Confidence")
            ax3.set_ylabel("Frequency")
            ax3.set_title(f"{algorithm_name} - Confidence Distribution")
            ax3.grid(True, alpha=0.3)
        else:
            ax3.text(
                0.5,
                0.5,
                "No confidence data",
                horizontalalignment="center",
                verticalalignment="center",
                transform=ax3.transAxes,
                fontsize=12,
            )
            ax3.set_title(f"{algorithm_name} - Confidence Distribution")

        # 缓存使用情况
        cache_sizes = []
        for perf in performance_history:
            if "cache_stats" in perf and perf["cache_stats"]:
                total_size = sum(
                    stats.get("total_size", 0) for stats in perf["cache_stats"].values()
                )
                cache_sizes.append(total_size)

        if cache_sizes:
            ax4.plot(
                range(len(cache_sizes)), cache_sizes, "g-", linewidth=2, marker="^"
            )
            ax4.set_xlabel("Training Session")
            ax4.set_ylabel("Total Cache Size")
            ax4.set_title(f"{algorithm_name} - Cache Usage Over Time")
            ax4.grid(True, alpha=0.3)
        else:
            ax4.text(
                0.5,
                0.5,
                "No cache data",
                horizontalalignment="center",
                verticalalignment="center",
                transform=ax4.transAxes,
                fontsize=12,
            )
            ax4.set_title(f"{algorithm_name} - Cache Usage Over Time")

        plt.tight_layout()

        if save_plot:
            plot_path = os.path.join(self.save_dir, f"{algorithm_name}_results.png")
            plt.savefig(plot_path, dpi=300, bbox_inches="tight")
            print(f"结果图表已保存至: {plot_path}")

        plt.show()
        return fig

    def plot_comparison(self, algorithms_results, save_plot=True):
        """比较不同算法的性能"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.flatten()

        # 准确率比较
        for algo_name, results in algorithms_results.items():
            accuracies = [
                perf.get("accuracy", 0) for perf in results["performance_history"]
            ]
            sessions = range(len(accuracies))
            axes[0].plot(sessions, accuracies, label=algo_name, linewidth=2)

        axes[0].set_xlabel("Training Session")
        axes[0].set_ylabel("Accuracy")
        axes[0].set_title("Accuracy Comparison")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # 损失比较
        for algo_name, results in algorithms_results.items():
            losses = [perf.get("loss", 0) for perf in results["performance_history"]]
            sessions = range(len(losses))
            axes[1].plot(sessions, losses, label=algo_name, linewidth=2)

        axes[1].set_xlabel("Training Session")
        axes[1].set_ylabel("Loss")
        axes[1].set_title("Loss Comparison")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        # 通信成本比较
        algo_names = []
        comm_costs = []
        for algo_name, results in algorithms_results.items():
            if "communication_costs" in results:
                algo_names.append(algo_name)
                comm_costs.append(np.sum(results["communication_costs"]))

        if comm_costs:
            bars = axes[2].bar(algo_names, comm_costs, alpha=0.7)
            axes[2].set_xlabel("Algorithm")
            axes[2].set_ylabel("Total Communication Cost")
            axes[2].set_title("Communication Cost Comparison")
            # 在柱状图上显示数值
            for bar, cost in zip(bars, comm_costs):
                axes[2].text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height(),
                    f"{cost:.0f}",
                    ha="center",
                    va="bottom",
                )

        # 训练时间比较
        algo_names = []
        train_times = []
        for algo_name, results in algorithms_results.items():
            if "time_history" in results and results["time_history"]:
                algo_names.append(algo_name)
                train_times.append(np.sum(results["time_history"]))

        if train_times:
            bars = axes[3].bar(algo_names, train_times, alpha=0.7, color="orange")
            axes[3].set_xlabel("Algorithm")
            axes[3].set_ylabel("Total Training Time (s)")
            axes[3].set_title("Training Time Comparison")
            # 在柱状图上显示数值
            for bar, time_val in zip(bars, train_times):
                axes[3].text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height(),
                    f"{time_val:.1f}s",
                    ha="center",
                    va="bottom",
                )

        plt.tight_layout()

        if save_plot:
            plot_path = os.path.join(self.save_dir, "algorithm_comparison.png")
            plt.savefig(plot_path, dpi=300, bbox_inches="tight")
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
        print(
            f"  准确率范围: {metrics.get('min_accuracy', 0):.4f} - {metrics.get('max_accuracy', 0):.4f}"
        )
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
        print(
            f"  损失范围: {metrics.get('min_loss', 0):.4f} - {metrics.get('max_loss', 0):.4f}"
        )

    def plot_data_heterogeneity(
        self, data_simulator, session, save_plot=True, plot_name=None
    ):
        """
        绘制数据异质性示意图

        参数:
            data_simulator: DomainIncrementalDataSimulator实例
            session: 当前会话ID
            save_plot: 是否保存图片
            plot_name: 图片名称，如果为None则自动生成
        """
        if plot_name is None:
            plot_name = f"data_heterogeneity_session_{session}.png"

        # 获取当前域的信息
        current_domain = data_simulator.get_current_domain()
        domain_key = f"{data_simulator.current_dataset}_{current_domain}"

        # 检查是否有数据分配
        if domain_key not in data_simulator.vehicle_data_assignments:
            print(f"警告: 域 {domain_key} 没有数据分配信息")
            return

        # 获取类别信息
        num_classes = data_simulator.dataset_info[data_simulator.current_dataset][
            "num_classes"
        ]
        class_labels = [f"Class {i}" for i in range(num_classes)]

        # 获取车辆分配数据
        vehicle_assignments = data_simulator.vehicle_data_assignments[domain_key]
        train_dataset = data_simulator.train_data_cache[domain_key]

        # 统计每个车辆每个类别的样本数量
        vehicle_class_counts = {}

        for vehicle_id, indices in vehicle_assignments.items():
            class_counts = {i: 0 for i in range(num_classes)}

            for idx in indices:
                # 获取样本的标签
                _, label = train_dataset[idx]
                class_counts[label] += 1

            vehicle_class_counts[vehicle_id] = class_counts

        # 准备绘图数据
        vehicle_ids = []
        class_ids = []
        sample_counts = []

        for vehicle_id in range(data_simulator.num_vehicles):
            if vehicle_id in vehicle_class_counts:
                for class_id in range(num_classes):
                    count = vehicle_class_counts[vehicle_id][class_id]
                    if count > 0:  # 只绘制有样本的类别
                        vehicle_ids.append(vehicle_id)
                        class_ids.append(class_id)
                        sample_counts.append(count)

        if not sample_counts:
            print("警告: 没有找到可绘制的数据")
            return

        # 创建图形
        plt.figure(figsize=(12, 8))

        # 创建散点图，点的大小表示样本数量
        scatter = plt.scatter(
            vehicle_ids,
            class_ids,
            s=[
                min(100 + count * 2, 500) for count in sample_counts
            ],  # 动态调整点的大小
            c=sample_counts,
            cmap="viridis",
            alpha=0.7,
            edgecolors="black",
            linewidth=0.5,
        )

        # 设置图表属性
        plt.title(
            f"Data Heterogeneity - Session {session}\n(Domain: {current_domain}, Dataset: {data_simulator.current_dataset})",
            fontsize=14,
            fontweight="bold",
            pad=20,
        )
        plt.xlabel("Vehicle ID", fontsize=12)
        plt.ylabel("Class Label", fontsize=12)

        # 设置坐标轴
        plt.xticks(range(data_simulator.num_vehicles))
        plt.yticks(range(num_classes), class_labels)
        plt.grid(True, alpha=0.3, linestyle="--")

        # 添加颜色条
        cbar = plt.colorbar(scatter, shrink=0.8)
        cbar.set_label("Number of Samples", fontsize=10)

        # 添加样本数量标注（只标注较大的点）
        for i, (vehicle_id, class_id, count) in enumerate(
            zip(vehicle_ids, class_ids, sample_counts)
        ):
            if count > max(sample_counts) * 0.3:  # 只标注较大的样本点
                plt.annotate(
                    str(count),
                    (vehicle_id, class_id),
                    xytext=(5, 5),
                    textcoords="offset points",
                    fontsize=8,
                    ha="left",
                    va="bottom",
                )

        # 调整布局
        plt.tight_layout()

        # 保存或显示图表
        if save_plot:
            plot_path = os.path.join(self.save_dir, plot_name)
            plt.savefig(plot_path, dpi=300, bbox_inches="tight")
            print(f"数据异质性图已保存至: {plot_path}")

        plt.close()

        # 打印统计信息
        # self._print_heterogeneity_statistics(vehicle_class_counts, current_domain, session)

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

    def plot_continual_learning_metrics(self, results, save_path=None, show_plot=True):
        """
        绘制连续学习指标随域切换的变化趋势

        参数:
            results: 包含记录结果的字典
            save_path: 保存图片的路径（可选）
            show_plot: 是否显示图片
        """
        if "continual_learning_metrics" not in results or not results["continual_learning_metrics"]:
            print("没有找到连续学习指标数据")
            return

        metrics_data = results["continual_learning_metrics"]

        # 提取数据
        sessions = [m["session"] for m in metrics_data]
        tasks = [m["task"] for m in metrics_data]
        domains = [m["domain"] for m in metrics_data]

        # 四个核心指标
        aa_values = [m["AA"] for m in metrics_data]
        aia_values = [m["AIA"] for m in metrics_data]
        fm_values = [m["FM"] for m in metrics_data]
        bwt_values = [m["BWT"] for m in metrics_data]

        # 创建图形
        fig, axes = plt.subplots(3, 2, figsize=(14, 12))
        fig.suptitle('Continual Learning Metrics Evolution', fontsize=16, fontweight='bold')

        # 设置白色背景
        fig.patch.set_facecolor('white')
        for ax in axes.flat:
            ax.set_facecolor('white')

        # 1. 平均准确率 (AA) 趋势
        ax1 = axes[0, 0]
        ax1.plot(sessions, aa_values, 'b-', linewidth=2, marker='o', markersize=4)
        ax1.set_xlabel('Session', fontsize=10)
        ax1.set_ylabel('AA (Average Accuracy)', fontsize=10)
        ax1.set_title('Average Accuracy Trend', fontsize=12, fontweight='bold')
        ax1.grid(True, alpha=0.3, linestyle='--')
        ax1.set_ylim(0, 1.05)

        # 标记域切换点
        self._mark_domain_changes(ax1, sessions, tasks, domains)

        # 2. 平均增量准确率 (AIA) 趋势
        ax2 = axes[0, 1]
        ax2.plot(sessions, aia_values, 'g-', linewidth=2, marker='s', markersize=4)
        ax2.set_xlabel('Session', fontsize=10)
        ax2.set_ylabel('AIA (Average Incremental Accuracy)', fontsize=10)
        ax2.set_title('Average Incremental Accuracy Trend', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3, linestyle='--')
        ax2.set_ylim(0, 1.05)

        # 3. 遗忘度量 (FM) 趋势
        ax3 = axes[1, 0]
        ax3.plot(sessions, fm_values, 'r-', linewidth=2, marker='^', markersize=4)
        ax3.set_xlabel('Session', fontsize=10)
        ax3.set_ylabel('FM (Forgetting Measure)', fontsize=10)
        ax3.set_title('Forgetting Measure Trend (lower is better)', fontsize=12, fontweight='bold')
        ax3.grid(True, alpha=0.3, linestyle='--')
        ax3.axhline(y=0, color='gray', linestyle='--', alpha=0.5)

        # 4. 反向迁移 (BWT) 趋势
        ax4 = axes[1, 1]
        ax4.plot(sessions, bwt_values, 'purple', linewidth=2, marker='d', markersize=4)
        ax4.set_xlabel('Session', fontsize=10)
        ax4.set_ylabel('BWT (Backward Transfer)', fontsize=10)
        ax4.set_title('Backward Transfer Trend (positive is good)', fontsize=12, fontweight='bold')
        ax4.grid(True, alpha=0.3, linestyle='--')
        ax4.axhline(y=0, color='gray', linestyle='--', alpha=0.5)

        # 5. 任务/域切换信息
        ax5 = axes[2, 0]
        self._plot_task_domain_info(ax5, sessions, tasks, domains)

        # 6. 指标对比图
        ax6 = axes[2, 1]
        self._plot_metrics_comparison(ax6, sessions, aa_values, aia_values, fm_values, bwt_values)

        plt.tight_layout()

        # 保存图片
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
            print(f"图片已保存到: {save_path}")

        # 显示图片
        if show_plot:
            plt.show()
        else:
            plt.close()

    def _mark_domain_changes(self, ax, sessions, tasks, domains):
        """在图上标记域切换点"""
        current_domain = None
        change_points = []

        for i, domain in enumerate(domains):
            if domain != current_domain:
                change_points.append((sessions[i], domain))
                current_domain = domain

        for session, domain in change_points:
            ax.axvline(x=session, color='orange', linestyle=':', alpha=0.5, linewidth=1)
            ax.text(session, ax.get_ylim()[1]*0.95, domain,
                   rotation=90, fontsize=8, alpha=0.7,
                   verticalalignment='top')

    def _plot_task_domain_info(self, ax, sessions, tasks, domains):
        """绘制任务和域信息"""
        ax.set_title('Task/Domain Progression', fontsize=12, fontweight='bold')
        ax.set_xlabel('Session', fontsize=10)

        # 创建颜色映射
        unique_domains = list(dict.fromkeys(domains))  # 保持顺序去重
        colors = plt.cm.Set3(np.linspace(0, 1, len(unique_domains)))
        domain_to_color = {domain: colors[i] for i, domain in enumerate(unique_domains)}

        # 绘制任务条形图
        for i, (session, task, domain) in enumerate(zip(sessions, tasks, domains)):
            color = domain_to_color.get(domain, 'gray')
            ax.barh(task, 1, left=session-0.5, height=0.8,
                   color=color, alpha=0.7, edgecolor='black', linewidth=0.5)
            if i == 0 or domains[i] != domains[i-1]:
                ax.text(session, task, f"{domain}\n(T{task})",
                       fontsize=7, ha='center', va='center')

        ax.set_yticks(sorted(set(tasks)))
        ax.set_ylabel('Task Number', fontsize=10)
        ax.grid(True, alpha=0.3, axis='x')

    def _plot_metrics_comparison(self, ax, sessions, aa, aia, fm, bwt):
        """绘制指标对比图（归一化后）"""
        ax.set_title('Normalized Metrics Comparison', fontsize=12, fontweight='bold')
        ax.set_xlabel('Session', fontsize=10)
        ax.set_ylabel('Normalized Value', fontsize=10)

        # 归一化处理（FM和BWT需要特殊处理）
        aa_norm = aa  # AA已经在0-1范围内
        aia_norm = aia  # AIA也在0-1范围内

        # FM归一化到0-1（越小越好）
        if max(fm) > 0:
            fm_norm = [1 - f/max(fm) for f in fm]  # 反转，越高越好
        else:
            fm_norm = [0] * len(fm)

        # BWT归一化到0-1（处理负值）
        bwt_min = min(bwt)
        bwt_max = max(bwt)
        if bwt_max > bwt_min:
            bwt_norm = [(b - bwt_min) / (bwt_max - bwt_min) for b in bwt]
        else:
            bwt_norm = [0.5] * len(bwt)

        # 绘制归一化后的指标
        ax.plot(sessions, aa_norm, 'b-', label='AA', linewidth=2)
        ax.plot(sessions, aia_norm, 'g--', label='AIA', linewidth=2)
        ax.plot(sessions, fm_norm, 'r-.', label='FM (inverted)', linewidth=2)
        ax.plot(sessions, bwt_norm, 'purple:', label='BWT (normalized)', linewidth=2)

        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_ylim(-0.1, 1.1)