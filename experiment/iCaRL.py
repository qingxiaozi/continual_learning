"""
iCaRL复现代码 - 基于Avalanche框架 (修正版)
在CIFAR-100数据集上实现类增量学习
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Avalanche库导入
import avalanche
from avalanche.benchmarks.classic import SplitCIFAR100
from avalanche.training import ICaRL
from avalanche.training.plugins import EvaluationPlugin, LRSchedulerPlugin
from avalanche.evaluation.metrics import (
    accuracy_metrics,
    loss_metrics,
    forgetting_metrics
)
# 注意：backward_transfer_metrics 和 forward_transfer_metrics 已不存在
# 使用替代的评估方式
from avalanche.logging import InteractiveLogger, TensorboardLogger, CSVLogger
from avalanche.models import MobilenetV1
from torch.optim.lr_scheduler import MultiStepLR

class iCaRLReproducer:
    """iCaRL算法复现器"""

    def __init__(self, config=None):
        """
        初始化iCaRL复现器

        Args:
            config: 配置字典，如果为None则使用默认配置
        """
        self.config = config or self.get_default_config()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"使用设备: {self.device}")

        # 初始化组件
        self.scenario = None
        self.model = None
        self.strategy = None
        self.results = []

        # 设置随机种子确保可复现性
        torch.manual_seed(self.config['seed'])
        np.random.seed(self.config['seed'])
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.config['seed'])

    @staticmethod
    def get_default_config():
        """获取默认配置（匹配iCaRL论文设置）"""
        return {
            # 数据集配置
            'dataset': 'cifar100',
            'n_experiences': 5,           # 增量任务数量（从10改为5以加快测试）
            'seed': 1234,                  # 随机种子
            'total_classes': 100,          # 总类别数

            # 数据增强配置
            'train_transform': transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5071, 0.4867, 0.4408),
                                   (0.2675, 0.2565, 0.2761))
            ]),
            'test_transform': transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5071, 0.4867, 0.4408),
                                   (0.2675, 0.2565, 0.2761))
            ]),
            'buffer_transform': transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.Normalize((0.5071, 0.4867, 0.4408),
                                   (0.2675, 0.2565, 0.2761))
            ]),

            # 模型配置
            'model_name': 'simplecnn',
            # 训练配置
            'batch_size': 64,              # 减小批大小以节省内存
            'epochs': 20,                  # 每个任务的训练轮数（从70改为20以加快测试）
            'learning_rate': 2.0,
            'momentum': 0.9,
            'weight_decay': 0.00001,

            # iCaRL特定配置
            'memory_size': 1000,           # 记忆库大小（从2000改为1000）
            'fixed_memory': True,          # 固定记忆库大小

            # 学习率调度
            'lr_milestones': [10, 15],     # 学习率下降的epoch（适配20个epoch）
            'lr_gamma': 0.2,               # 学习率下降因子

            # 评估配置
            'eval_every': 1,               # 每个epoch都评估
            'log_tensorboard': False,      # 禁用Tensorboard以简化（可改回True）
            'log_csv': True,               # 启用CSV日志
            'log_dir': './logs',           # 日志目录
        }

    def setup_data(self):
        """设置数据集和增量学习场景"""
        print("\n" + "="*60)
        print("设置数据集和增量学习场景")
        print("="*60)

        # 创建SplitCIFAR100场景（类增量学习）
        self.scenario = SplitCIFAR100(
            n_experiences=self.config['n_experiences'],
            seed=self.config['seed'],
            fixed_class_order=list(range(100)),  # 使用固定类别顺序
            train_transform=self.config['train_transform'],
            eval_transform=self.config['test_transform'],
            shuffle=True,
        )

        # 打印场景信息
        print(f"数据集: {self.config['dataset'].upper()}")
        print(f"总类别数: {self.scenario.n_classes}")
        print(f"增量任务数: {self.scenario.n_experiences}")
        print(f"每个任务的类别数: {self.scenario.n_classes // self.scenario.n_experiences}")

        # 显示第一个任务的类别信息
        first_exp = self.scenario.train_stream[0]
        print(f"\n第一个任务包含类别: {sorted(first_exp.classes_in_this_experience)}")

        return self.scenario

    def setup_model(self):
        print("\n" + "="*60)
        print("设置特征提取器和分类器")
        print("="*60)

        if self.config['model_name'] == 'resnet18':
            print(f"resnet18待添加")

        elif self.config['model_name'] == 'simplecnn':
            from avalanche.models import SimpleCNN
            import torch.nn as nn

            # 创建 SimpleCNN（num_classes 任意，我们只取特征部分）
            full_model = SimpleCNN(num_classes=10)
            print(full_model)

            self.feature_extractor = nn.Sequential(
                full_model.features,
                nn.Flatten()
            )

            # 推断特征维度
            self.feature_extractor = self.feature_extractor.to(self.device)
            self.feature_extractor.eval()
            with torch.no_grad():
                dummy = torch.randn(1, 3, 32, 32).to(self.device)
                feat = self.feature_extractor(dummy)
                self.feature_dim = feat.shape[1]  # 应为 64

            # 创建分类器（输出总类别数）
            total_classes = self.config['total_classes']
            self.classifier = nn.Linear(self.feature_dim, total_classes).to(self.device)

            print(f"特征维度: {self.feature_dim}, 分类器输出: {total_classes}")

        else:
            raise ValueError(f"不支持模型: {self.config['model_name']}")

        total_params = sum(p.numel() for p in self.feature_extractor.parameters())
        print(f"模型: {self.config['model_name']}")
        print(f"特征维度: {self.feature_dim}")
        print(f"参数量: {total_params:,}")
        self.model = nn.Sequential(self.feature_extractor, self.classifier).to(self.device)
        return self.feature_extractor


    def setup_loggers(self):
        """设置日志记录器"""
        loggers = []

        # 交互式日志（控制台输出）
        interactive_logger = InteractiveLogger()
        loggers.append(interactive_logger)

        # CSV日志
        if self.config['log_csv']:
            import os
            os.makedirs(self.config['log_dir'], exist_ok=True)
            csv_logger = CSVLogger(f"{self.config['log_dir']}/icarl_results.csv")
            loggers.append(csv_logger)

        # Tensorboard日志
        if self.config['log_tensorboard']:
            tb_logger = TensorboardLogger(f"{self.config['log_dir']}/tensorboard")
            loggers.append(tb_logger)

        return loggers

    def setup_evaluation(self, loggers):
        """设置评估插件"""
        # 定义要跟踪的指标
        eval_plugin = EvaluationPlugin(
            # 准确率指标
            accuracy_metrics(
                epoch=True,           # 每个epoch的准确率
                experience=True,      # 每个经验的准确率
                stream=True,          # 整个流的准确率
            ),

            # 损失指标
            loss_metrics(
                epoch=True,
                experience=True,
                stream=True,
            ),

            # 持续学习特定指标 - 使用可用的指标
            forgetting_metrics(experience=True, stream=True),

            # 日志记录器
            loggers=loggers,
        )

        return eval_plugin

    def setup_strategy(self):
        """设置iCaRL训练策略"""
        print("\n" + "="*60)
        print("设置iCaRL训练策略")
        print("="*60)

        model_params = list(self.feature_extractor.parameters()) + list(self.classifier.parameters())

        optimizer = optim.SGD(
            model_params,
            lr=self.config['learning_rate'],
            momentum=self.config['momentum'],
            weight_decay=self.config['weight_decay']
        )

        scheduler = MultiStepLR(
            optimizer,
            milestones=self.config['lr_milestones'],
            gamma=self.config['lr_gamma']
        )

        loggers = self.setup_loggers()
        eval_plugin = self.setup_evaluation(loggers)

        # ✅ 关键修正：传入 feature_extractor，而不是完整模型
        self.strategy = ICaRL(
            feature_extractor=self.feature_extractor,
            classifier=self.classifier,
            optimizer=optimizer,
            memory_size=self.config['memory_size'],
            buffer_transform=self.config['buffer_transform'],
            fixed_memory=self.config['fixed_memory'],
            train_mb_size=self.config['batch_size'],
            train_epochs=self.config['epochs'],
            eval_mb_size=self.config['batch_size'],
            evaluator=eval_plugin,
            eval_every=self.config['eval_every'],
            device=self.device,
            plugins=[LRSchedulerPlugin(scheduler)]
        )

        print("iCaRL策略配置完成")
        return self.strategy

    def train_and_evaluate(self):
        """训练和评估主循环"""
        print("\n" + "="*60)
        print("开始iCaRL增量训练")
        print("="*60)

        self.results = []
        self.accuracy_matrix = []  # R[i][j]: 训练完任务 i 后，在任务 j 上的准确率
        n_exps = self.config['n_experiences']

        # 遍历所有增量任务
        for exp_id, experience in enumerate(self.scenario.train_stream):
            print(f"\n{'='*50}")
            print(f"任务 {exp_id+1}/{n_exps}")
            print(f"{'='*50}")

            # 显示当前任务的类别
            current_classes = sorted(experience.classes_in_this_experience)
            print(f"学习类别: {current_classes}")

            # 训练当前任务
            print(f"\n训练任务 {exp_id+1}...")
            self.strategy.train(experience)

            # 评估：对每个已见任务（0 到 exp_id）单独测试
            task_accuracies = []
            for j in range(exp_id + 1):
                test_exp = self.scenario.test_stream[j]
                eval_res = self.strategy.eval([test_exp])

                # 尝试匹配 Avalanche 的准确率键名
                acc = 0.0
                # 标准键名（Avalanche >= 0.5）
                key = f'Top1_Acc_Exp/eval_phase/test_stream/Task00{j}'
                if key in eval_res:
                    acc = eval_res[key]
                else:
                    # 兼容旧版或变体
                    for k, v in eval_res.items():
                        if f'Task00{j}' in k and 'Acc_Exp' in k and isinstance(v, (int, float)):
                            acc = v
                            break
                task_accuracies.append(acc)

            # 补零至总任务数（便于后续处理）
            while len(task_accuracies) < n_exps:
                task_accuracies.append(0.0)

            self.accuracy_matrix.append(task_accuracies)

            # 全局评估（用于日志）
            global_eval = self.strategy.eval(self.scenario.test_stream[:exp_id+1])
            self.results.append(global_eval)

            # 打印当前累计准确率（stream accuracy）
            seen_accs = task_accuracies[:exp_id+1]
            stream_acc = np.mean(seen_accs) if seen_accs else 0.0
            print(f"当前累计准确率 (Stream Acc): {stream_acc:.2%}")

        return self.results

    def extract_stream_accuracies_from_matrix(self):
        """从 accuracy_matrix 计算每一步的 stream accuracy"""
        if not hasattr(self, 'accuracy_matrix') or len(self.accuracy_matrix) == 0:
            return []

        stream_accs = []
        for i, row in enumerate(self.accuracy_matrix):
            seen = [acc for acc in row[:i+1] if acc > 0]
            if seen:
                stream_accs.append(np.mean(seen))
            else:
                stream_accs.append(0.0)
        return stream_accs

    def compute_cl_metrics(self, stream_accuracies, R=None):
        """
        计算 AA, AIA, FM, BWT
        """
        n = len(stream_accuracies)
        if n == 0:
            return {'AA': 0, 'AIA': 0, 'FM': 0, 'BWT': 0}

        AA = stream_accuracies[-1]
        AIA = np.mean(stream_accuracies)

        if R is not None:
            R = np.array(R)
            # 每个任务 j 的历史最高准确率
            R_max = np.max(R, axis=0)

            # Forgetting Measure (FM)
            forgetting = []
            for j in range(R.shape[1]):
                if R[-1, j] > 0:  # 只考虑已见任务
                    f = max(0, R_max[j] - R[-1, j])
                    forgetting.append(f)
            FM = np.mean(forgetting) if forgetting else 0.0

            # Backward Transfer (BWT)
            bwt_list = []
            for j in range(R.shape[1] - 1):  # 最后一个任务无 BWT
                if R[j, j] > 0 and R[-1, j] > 0:
                    bwt = R[-1, j] - R[j, j]
                    bwt_list.append(bwt)
            BWT = np.mean(bwt_list) if bwt_list else 0.0
        else:
            # 简化估计（不推荐，仅备用）
            FM = max(0, max(stream_accuracies) - stream_accuracies[-1])
            BWT = -FM

        return {
            'AA': AA,
            'AIA': AIA,
            'FM': FM,
            'BWT': BWT
        }

    def analyze_results(self):
        """分析并输出四个核心持续学习指标"""
        if not hasattr(self, 'accuracy_matrix') or len(self.accuracy_matrix) == 0:
            print("警告：未找到准确率矩阵，尝试从 results 提取（可能不准确）")
            # stream_accs = self.extract_stream_accuracies_from_results()
            # metrics = self.compute_cl_metrics(stream_accs)
        else:
            stream_accs = self.extract_stream_accuracies_from_matrix()
            metrics = self.compute_cl_metrics(stream_accs, self.accuracy_matrix)

        print("\n" + "="*60)
        print("持续学习核心指标分析")
        print("="*60)
        print(f"  最终平均准确率 (AA):      {metrics['AA']:.2%}")
        print(f"  增量平均准确率 (AIA):     {metrics['AIA']:.2%}")
        print(f"  平均遗忘度量 (FM):        {metrics['FM']:.4f}")
        print(f"  反向迁移 (BWT):           {metrics['BWT']:+.4f}")

        # 与论文对比（可选）
        n_exp = self.config['n_experiences']
        if n_exp == 10:
            print(f"\n参考 (iCaRL 原文, CIFAR-100, 10 tasks): AA ≈ 64.1%")
        elif n_exp == 5:
            print(f"\n参考 (典型值, CIFAR-100, 5 tasks): AA ≈ 50–55%")

        return metrics

    def visualize_results(self, save_path='icarl_results.png'):
        """可视化结果"""
        stream_accs = self.extract_stream_accuracies_from_matrix()
        if not stream_accs:
            print("无有效准确率数据")
            return

        tasks = list(range(1, len(stream_accs) + 1))
        accuracies = stream_accs  # 直接使用，避免键名问题

        # 创建图形
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # 1. 准确率曲线
        ax1 = axes[0]
        ax1.plot(tasks, accuracies, 'b-o', linewidth=2, markersize=8)
        ax1.fill_between(tasks,
                        np.array(accuracies) - 0.02,
                        np.array(accuracies) + 0.02,
                        alpha=0.2)
        ax1.set_xlabel('任务编号', fontsize=12)
        ax1.set_ylabel('累计准确率', fontsize=12)
        ax1.set_title('iCaRL在CIFAR-100上的增量学习性能', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim([max(0, min(accuracies)-0.1), min(1.0, max(accuracies)+0.1)])

        # 2. 准确率增长曲线
        ax2 = axes[1]
        if len(accuracies) > 1:
            accuracy_gains = [accuracies[0]] + [accuracies[i] - accuracies[i-1] for i in range(1, len(accuracies))]
            colors = ['green' if gain >= 0 else 'red' for gain in accuracy_gains]
            bars = ax2.bar(tasks, accuracy_gains, color=colors, edgecolor='black', alpha=0.7)
            ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
            ax2.set_xlabel('任务编号', fontsize=12)
            ax2.set_ylabel('准确率增长', fontsize=12)
            ax2.set_title('每个任务带来的准确率增长', fontsize=14, fontweight='bold')
            ax2.grid(True, alpha=0.3, axis='y')

            # 添加数值标签
            for i, (bar, gain) in enumerate(zip(bars, accuracy_gains)):
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height,
                        f'{gain:+.3f}', ha='center', va='bottom' if height >= 0 else 'top')

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\n可视化结果已保存到: {save_path}")
        plt.show()

    def save_model(self, path='icarl_model.pth'):
        """保存模型"""
        if hasattr(self, 'feature_extractor') and hasattr(self, 'classifier'):
            torch.save({
                'feature_extractor_state_dict': self.feature_extractor.state_dict(),
                'classifier_state_dict': self.classifier.state_dict(),
                'config': self.config,
                'accuracy_matrix': getattr(self, 'accuracy_matrix', None),
            }, path)
            print(f"模型已保存到: {path}")
        else:
            print("警告：模型组件未初始化，无法保存")

    def run_iCaRL(self):
        """运行完整的iCaRL复现实验"""
        try:
            # 1. 设置数据
            self.setup_data()

            # 2. 设置模型
            self.setup_model()

            # 3. 设置训练策略
            self.setup_strategy()

            # 4. 训练和评估
            self.train_and_evaluate()

            # 5. 分析结果
            analysis = self.analyze_results()

            # 6. 可视化结果
            self.visualize_results()

            # 7. 保存模型
            self.save_model()

            print("\n" + "="*60)
            print("iCaRL复现实验完成!")
            print("="*60)

            return analysis

        except Exception as e:
            print(f"\n实验出错: {e}")
            import traceback
            traceback.print_exc()
            return None


# 简化的测试脚本
def test_quick():
    """快速测试函数"""
    print("快速测试iCaRL实现...")

    # 极简配置用于快速测试
    config = {
        'n_experiences': 2,      # 只测试2个任务
        'batch_size': 32,
        'epochs': 5,            # 每个任务只训练5轮
        'memory_size': 200,
        'log_tensorboard': False,
        'log_csv': False,
    }

    reproducer = iCaRLReproducer(config)

    try:
        # 只运行部分步骤进行测试
        reproducer.setup_data()
        reproducer.setup_model()
        reproducer.setup_strategy()

        # 只运行第一个任务测试
        print("\n测试第一个任务...")
        first_exp = reproducer.scenario.train_stream[0]
        reproducer.strategy.train(first_exp)

        # 简单评估
        print("测试评估...")
        eval_result = reproducer.strategy.eval(reproducer.scenario.test_stream[:1])

        print("\n测试成功完成!")
        print(f"评估结果键: {list(eval_result.keys())[:5]}...")  # 只显示前5个键

        return True
    except Exception as e:
        print(f"测试失败: {e}")
        return False


def main():
    """主函数"""
    print("iCaRL算法复现 - CIFAR-100数据集")
    print("基于Avalanche框架实现")
    print("="*60)

    # 检查Avalanche版本
    try:
        import avalanche
        print(f"Avalanche版本: {avalanche.__version__}")
    except:
        print("无法获取Avalanche版本")

    # 询问用户选择模式
    print("\n请选择运行模式:")
    print("1. 快速测试 (2个任务，每个5轮)")
    print("2. 完整实验 (5个任务，每个20轮)")
    print("3. 论文配置 (10个任务，每个70轮)")

    choice = input("请输入选择 (1/2/3, 默认为2): ").strip()

    if choice == '1':
        # 快速测试模式
        success = test_quick()
        if success:
            print("\n快速测试通过，可以运行完整实验了!")
        else:
            print("\n快速测试失败，请检查错误信息")
    elif choice == '3':
        # 论文配置模式
        config = {
            'n_experiences': 10,
            'batch_size': 128,
            'epochs': 70,
            'memory_size': 2000,
            'lr_milestones': [49, 63],
            'log_tensorboard': True,
        }
        reproducer = iCaRLReproducer(config)
        results = reproducer.run_iCaRL()
    else:
        # 默认完整实验模式
        reproducer = iCaRLReproducer()
        results = reproducer.run_iCaRL()


    # 打印最终总结
    if results:
        print("\n" + "="*60)
        print("实验总结")
        print("="*60)
        print(f"数据集: CIFAR-100")
        print(f"增量任务: {reproducer.config['n_experiences']}")
        print(f"最终准确率 (AA): {results['AA']:.2%}")
        print(f"平均遗忘 (FM): {results['FM']:.4f}")
        print("="*60)


if __name__ == "__main__":
    main()