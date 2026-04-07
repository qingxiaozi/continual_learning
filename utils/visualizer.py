import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from config.parameters import Config
from config.paths import Paths

plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['axes.grid'] = False
plt.rcParams['legend.frameon'] = False


class ResultVisualizer:
    """结果可视化类"""

    def __init__(self, save_dir=None):
        if save_dir is None:
            save_dir = Paths.get_dataset_dir("png")
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)
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

        all_losses = list(epoch_losses)
        if val_losses and len(val_losses) > 0:
            all_losses.extend(val_losses[:len(epoch_losses)])
        if epoch_elastic_losses and len(epoch_elastic_losses) > 0:
            all_losses.extend(epoch_elastic_losses[:len(epoch_losses)])

        if len(all_losses) > 1:
            loss_min, loss_max = min(all_losses), max(all_losses)
            margin = (loss_max - loss_min) * 0.1 or 0.1
            plt.ylim(loss_min - margin, loss_max + margin)

        if save_plot:
            plot_path = os.path.join(self.save_dir, plot_name)
            plt.savefig(plot_path, dpi=300, bbox_inches="tight")
            print(f"训练损失图已保存至: {plot_path}")

        plt.close()

    def plot_accuracy_curve(
        self,
        val_accuracies,
        save_plot=True,
        plot_name="accuracy_curve.png"
    ):
        """绘制验证准确率曲线"""
        plt.figure(figsize=(10, 6))

        epochs = range(1, len(val_accuracies) + 1)
        plt.plot(epochs, val_accuracies, "b-", linewidth=2, label="Validation Accuracy", marker='o', markersize=4)

        plt.title("Validation Accuracy", fontsize=14, fontweight="bold")
        plt.xlabel("Epoch", fontsize=12)
        plt.ylabel("Accuracy (%)", fontsize=12)
        plt.legend(fontsize=10)
        plt.xticks(epochs)

        if len(val_accuracies) > 1:
            acc_min, acc_max = min(val_accuracies), max(val_accuracies)
            margin = (acc_max - acc_min) * 0.1 or 1.0
            plt.ylim(acc_min - margin, acc_max + margin)

        if save_plot:
            plot_path = os.path.join(self.save_dir, plot_name)
            plt.savefig(plot_path, dpi=300, bbox_inches="tight")
            print(f"验证准确率曲线已保存至: {plot_path}")

        plt.close()

    def plot_data_heterogeneity(self, data_simulator, session, save_plot=True):
        """绘制数据异质性示意图"""
        domain = data_simulator.get_current_domain()
        domain_key = f"{data_simulator.current_dataset}_{domain}"

        if domain_key not in data_simulator.vehicle_data_assignments:
            return

        num_classes = data_simulator.dataset_info[data_simulator.current_dataset]["num_classes"]
        vehicle_assignments = data_simulator.vehicle_data_assignments[domain_key]
        train_dataset = data_simulator.train_data_cache[domain_key]

        vehicle_class_counts = {}
        for vehicle_id, indices in vehicle_assignments.items():
            class_counts = [0] * num_classes
            for idx in indices:
                _, label = train_dataset[idx]
                class_counts[label] += 1
            vehicle_class_counts[vehicle_id] = class_counts

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

    def plot_sample_count_evolution(self, session_history):
        """绘制样本数随session的变化"""
        if not session_history:
            print("没有数据可绘制")
            return None

        sessions = [r.get("session", i) for i, r in enumerate(session_history)]
        total_samples = [r.get("total_samples", 0) for r in session_history]

        fig, ax = plt.subplots(figsize=self.style_config['figsize'])

        bars = ax.bar(sessions, total_samples,
                     color=self.style_config.get('colors', {}).get('samples', '#1f77b4'),
                     alpha=0.7,
                     edgecolor='black',
                     linewidth=1)

        ax.set_title('Sample Count Per Session', fontsize=14, fontweight='bold')
        ax.set_xlabel('Session', fontsize=12)
        ax.set_ylabel('Number of Samples', fontsize=12)

        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{int(height):,}',
                       ha='center', va='bottom',
                       fontsize=9, fontweight='bold')

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

            lines, labels = ax.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax.legend(lines + lines2, labels + labels2, loc='upper left')
        else:
            ax.legend(['Samples per Session'], loc='upper left')

        plt.tight_layout()
        return fig

    def _load_if_exists(self, filename, base_dir=None):
        if base_dir is None:
            base_dir = os.path.dirname(self.save_dir)
        path = os.path.join(base_dir, "npy", filename)
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
        plt.boxplot(data, tick_labels=labels, patch_artist=False)
        plt.ylabel("Metric Value"); plt.title("Episode-level CL Metrics")
        plt.grid(alpha=0.3)
        self._save_fig(filename)

    def plot_episode_reward(self, rewards, filename="episode_reward.png"):
        plt.figure()
        plt.plot(rewards)
        plt.xlabel("Episode"); plt.ylabel("Total Reward")
        plt.title("Episode-level Total Reward")
        print(f"{filename}")
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

    def plot_vehicle_data_heterogeneity(self, data_simulator, session, save_plot=True):
        """显示车辆-类别样本分布，点大小表示数量"""
        domain = data_simulator.get_current_domain()
        key = f"{data_simulator.current_dataset}_{domain}"
        if key not in data_simulator.vehicle_data_assignments:
            print(f"No data assignments for {key}")
            return

        num_classes = data_simulator.dataset_info[data_simulator.current_dataset]["num_classes"]
        va = data_simulator.vehicle_data_assignments[key]
        td = data_simulator.train_data_cache[key]

        vehicle_ids, class_ids, sample_counts = [], [], []
        for vid, idxs in va.items():
            counts = [0] * num_classes
            for i in idxs:
                _, lbl = td[i]
                counts[lbl] += 1
            for cls, cnt in enumerate(counts):
                if cnt:
                    vehicle_ids.append(vid); class_ids.append(cls); sample_counts.append(cnt)

        if not sample_counts:
            print(f"No sample data to visualize for session {session}")
            return

        fig, ax = plt.subplots(figsize=(12, 8))
        scatter = ax.scatter(
            vehicle_ids, class_ids,
            s=[c * 5 + 50 for c in sample_counts],
            c=sample_counts, cmap="YlOrRd", alpha=0.7,
            edgecolors="darkred", linewidth=0.8,
        )
        ax.set(xlabel="Vehicle ID", ylabel="Class",
               title=f"Vehicle Data Heterogeneity (Dirichlet Non-IID) - Domain: {domain}, Session {session}")
        ax.set_xticks(range(data_simulator.num_vehicles))
        ax.set_yticks(range(num_classes))
        ax.set_yticklabels([f"C{i}" for i in range(num_classes)], fontsize=9)
        ax.grid(True, alpha=0.3, linestyle="--")
        fig.colorbar(scatter, ax=ax, label='Sample Count')
        plt.tight_layout()

        if save_plot:
            plot_path = os.path.join(
                self.save_dir,
                f"vehicle_data_heterogeneity_domain_{domain}_session_{session}.png",
            )
            plt.savefig(plot_path, dpi=150, bbox_inches='tight', facecolor='white')
            print(f"Vehicle data heterogeneity plot saved to {plot_path}")
        plt.close()

    def plot_tsne_domain_shift(self, global_model, data_simulator, session, num_samples=500, save_plot=True):
        """t-SNE 展示已见域间的特征分布差异"""
        from torch.utils.data import DataLoader
        import torch

        device = next(global_model.model.parameters()).device
        global_model.model.eval()

        seen = data_simulator.seen_domains
        if not seen:
            print("No domains seen yet")
            return

        features, labels = [], []
        for domain in seen:
            key = f"{data_simulator.current_dataset}_{domain}"
            dataset = data_simulator.test_data_cache.get(key)
            if dataset is None:
                continue
            loader = DataLoader(dataset, batch_size=64, shuffle=True)
            collected = 0
            for imgs, _ in loader:
                if collected >= num_samples:
                    break
                imgs = imgs.to(device)
                with torch.no_grad():
                    out = global_model.model(imgs)
                    if hasattr(out, 'pooler_output'):
                        feats = out.pooler_output
                    elif hasattr(out, 'last_hidden_state'):
                        feats = out.last_hidden_state.mean(dim=1)
                    else:
                        feats = out
                arr = feats.cpu().numpy()
                take = min(len(arr), num_samples - collected)
                features.append(arr[:take])
                labels.extend([domain] * take)
                collected += take

        if not features:
            print("No features extracted")
            return

        all_feats = np.vstack(features)
        scaled = StandardScaler().fit_transform(all_feats)

        print("Running t-SNE (this may take a while)...")
        tsne = TSNE(n_components=2, random_state=42,
                    perplexity=min(30, len(all_feats) - 1), max_iter=1000)
        emb = tsne.fit_transform(scaled)

        fig, ax = plt.subplots(figsize=(12, 9))
        colors = plt.cm.tab10(np.linspace(0, 1, len(seen)))
        for idx, domain in enumerate(seen):
            mask = np.array(labels) == domain
            ax.scatter(
                emb[mask, 0], emb[mask, 1],
                c=[colors[idx]], label=domain,
                s=50, alpha=0.6,
                edgecolors='black', linewidth=0.5,
            )

        ax.set(xlabel="t-SNE Dimension 1", ylabel="t-SNE Dimension 2",
               title=f"Domain Shift Visualization (t-SNE) - Session {session}\n"
                     f"({len(seen)} domains, {len(all_feats)} samples total)")
        ax.legend(title="Domain", fontsize=10, title_fontsize=11, loc='best', framealpha=0.9)
        ax.grid(True, alpha=0.3, linestyle='--')

        plt.tight_layout()
        if save_plot:
            path = os.path.join(self.save_dir, f"tsne_domain_shift_session_{session}.png")
            plt.savefig(path, dpi=150, bbox_inches='tight', facecolor='white')
            print(f"t-SNE domain shift plot saved to {path}")
        plt.close()

    def visualize_all(self):
        """加载 results/npy/{dataset}/npy_1/*.npy 并生成所有可视化图表"""
        dataset_name = Config.CURRENT_DATASET
        base_npy_dir = Paths.get_dataset_dir("npy")
        base_npy_dir = os.path.join(base_npy_dir, "npy_1")
        
        legend_mapping = {
            'DRL_EQUAL_FIXED_RATIO': 'Abl_UP_DRL',
            'DRL_MINMAX_DELAY_FIXED_RATIO': 'OURS FULL',
            'DRL_EQUAL_LOSS_GREEDY_FIXED_RATIO': 'Abl_UP_G',
            'FIXED_RATIO_EQUAL_FIXED_RATIO': 'BASE',
            'FIXED_RATIO_EQUAL_NEW_ONLY': 'Abl_TR_NEW',
            'FIXED_RATIO_MINMAX_DELAY_FIXED_RATIO': 'Abl_BW',
            'LOSS_GREEDY_EQUAL_FIXED_RATIO': 'AbI_UP_G',
        }
        
        if not os.path.exists(base_npy_dir):
            print(f"目录不存在: {base_npy_dir}")
            return
        
        all_files = []
        for root, dirs, files in os.walk(base_npy_dir):
            for f in files:
                if f.endswith('.npy'):
                    all_files.append((root, f))
        
        if not all_files:
            print(f"没有找到npy文件: {base_npy_dir}")
            return
        
        prefixes = set()
        file_map = {}
        
        for root, f in all_files:
            fname = f.replace('.npy', '')
            parts = fname.split('_')
            
            if fname.endswith('_all'):
                metric_name = parts[-2] if len(parts) >= 2 else None
                if metric_name in ['AA', 'AIA', 'FM', 'BWT']:
                    prefix = '_'.join(parts[:-2])
                    key = (prefix, metric_name)
                    file_map[key] = os.path.join(root, f)
                    prefixes.add(prefix)
            elif fname.endswith('_episode_rewards') or fname.endswith('_episode_delays'):
                suffix = 'rewards' if 'rewards' in fname else 'delays'
                prefix = '_'.join(parts[:-2])
                key = (prefix, suffix)
                file_map[key] = os.path.join(root, f)
                prefixes.add(prefix)
        
        prefixes = sorted(list(prefixes))
        print(f"找到 {len(prefixes)} 个实验: {prefixes}")
        
        metrics = ['AA', 'AIA', 'FM', 'BWT']
        system_metrics = [('rewards', 'Reward'), ('delays', 'Delay')]
        
        colors = plt.cm.tab10(np.linspace(0, 1, len(prefixes)))
        markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', 'h']
        
        output_dir = Paths.get_dataset_dir("png")
        os.makedirs(output_dir, exist_ok=True)
        
        for metric in metrics:
            fig, ax = plt.subplots(figsize=(8, 6))
            
            plotted = False
            for idx, prefix in enumerate(prefixes):
                if prefix not in legend_mapping:
                    continue
                    
                key = (prefix, metric)
                if key in file_map:
                    filepath = file_map[key]
                    try:
                        data = np.load(filepath)
                        if len(data) > 0:
                            x = np.arange(len(data))
                            ax.plot(x, data, label=legend_mapping[prefix], color=colors[idx], 
                                   marker=markers[idx % len(markers)], markersize=2, linewidth=1.5, alpha=0.8)
                            plotted = True
                    except Exception as e:
                        print(f"加载失败 {filepath}: {e}")
            
            if plotted:
                ax.set_xlabel('Episode', fontsize=14)
                ax.set_ylabel(metric, fontsize=14)
                ax.set_title(f'{metric} on {dataset_name.upper()}', fontsize=16, fontweight='bold')
                ax.legend(loc='center left', bbox_to_anchor=(1.0, 0.5), fontsize=9, frameon=False)
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                
                output_path = os.path.join(output_dir, f"{metric} on {dataset_name}.png")
                plt.tight_layout(rect=[0, 0, 0.85, 1])
                plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
                plt.close()
                print(f"保存: {output_path}")
        
        for metric, title in system_metrics:
            fig, ax = plt.subplots(figsize=(8, 6))
            
            plotted = False
            for idx, prefix in enumerate(prefixes):
                if prefix not in legend_mapping:
                    continue
                    
                key = (prefix, metric)
                if key in file_map:
                    filepath = file_map[key]
                    try:
                        data = np.load(filepath)
                        if len(data) > 0:
                            x = np.arange(len(data))
                            ax.plot(x, data, label=legend_mapping[prefix], color=colors[idx],
                                   marker=markers[idx % len(markers)], markersize=2, linewidth=1.5, alpha=0.8)
                            plotted = True
                    except Exception as e:
                        print(f"加载失败 {filepath}: {e}")
            
            if plotted:
                ax.set_xlabel('Episode', fontsize=14)
                ax.set_ylabel(title, fontsize=14)
                ax.set_title(f'{title} on {dataset_name.upper()}', fontsize=16, fontweight='bold')
                ax.legend(loc='center left', bbox_to_anchor=(1.0, 0.5), fontsize=9, frameon=False)
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                
                output_path = os.path.join(output_dir, f"{title} on {dataset_name}.png")
                plt.tight_layout(rect=[0, 0, 0.85, 1])
                plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
                plt.close()
                print(f"保存: {output_path}")
        
        print(f"\n{dataset_name.upper()} 可视化完成！共生成 6 张图")
        print(f"图片保存在: {output_dir}")


if __name__ == "__main__":
    vis = ResultVisualizer()
    vis.visualize_all()
