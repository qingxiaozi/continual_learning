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
                # 注意：没有 ToTensor！没有 Normalize！
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

            print(f"✅ 特征维度: {self.feature_dim}, 分类器输出: {total_classes}")

        else:
            raise ValueError(f"不支持模型: {self.config['model_name']}")

        total_params = sum(p.numel() for p in self.feature_extractor.parameters())
        print(f"模型: {self.config['model_name']}")
        print(f"特征维度: {self.feature_dim}")
        print(f"参数量: {total_params:,}")
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

        optimizer = optim.SGD(
            self.feature_extractor.parameters(),
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

        # 遍历所有增量任务
        for exp_id, experience in enumerate(self.scenario.train_stream):
            print(f"\n{'='*50}")
            print(f"任务 {exp_id+1}/{self.config['n_experiences']}")
            print(f"{'='*50}")

            # 显示当前任务的类别
            current_classes = sorted(experience.classes_in_this_experience)
            print(f"学习类别: {current_classes}")
            print(f"类别数量: {len(current_classes)}")

            # 训练当前任务
            print(f"\n训练任务 {exp_id+1}...")
            self.strategy.train(experience)

            # 评估当前模型在所有已见任务上的性能
            print(f"评估任务 {exp_id+1}...")
            eval_result = self.strategy.eval(self.scenario.test_stream)
            self.results.append(eval_result)

            # 显示当前性能
            # 尝试不同的键名
            acc_keys = [
                'Top1_Acc_Stream/eval_phase/test_stream',
                'Top1_Acc_Stream_Phase_eval_Stream_test',
                'Top1_Acc_Stream'
            ]

            for key in acc_keys:
                if key in eval_result:
                    print(f"当前累计准确率: {eval_result[key]:.2%}")
                    break

        return self.results

    def compute_custom_metrics(self):
        """计算自定义的持续学习指标"""
        if not self.results or len(self.results) < 2:
            return {}

        # 计算任务准确率矩阵（用于计算BWT等）
        task_acc_matrix = []

        # 首先收集每个任务在每个评估点的准确率
        # 注意：这需要从详细日志中提取，这里简化处理
        for result in self.results:
            # 这里假设我们可以从结果中提取任务级别的准确率
            # 实际实现可能需要修改日志配置
            pass

        # 简化计算：使用累计准确率的变化估计BWT
        if len(self.results) >= 2:
            final_acc = self.results[-1].get('Top1_Acc_Stream/eval_phase/test_stream', 0)
            penultimate_acc = self.results[-2].get('Top1_Acc_Stream/eval_phase/test_stream', 0)

            # 简单的反向迁移估计
            bwt_estimate = final_acc - penultimate_acc if final_acc > 0 else 0

            return {
                'estimated_backward_transfer': bwt_estimate,
                'final_accuracy': final_acc,
            }

        return {}

    def analyze_results(self):
        """分析训练结果"""
        if not self.results:
            print("没有可分析的结果")
            return None

        print("\n" + "="*60)
        print("结果分析")
        print("="*60)

        # 提取关键指标
        accuracies = []
        forgetting_scores = []

        for i, result in enumerate(self.results):
            # 累计准确率
            acc_keys = ['Top1_Acc_Stream/eval_phase/test_stream', 'Top1_Acc_Stream']
            acc_found = False
            for key in acc_keys:
                if key in result:
                    accuracies.append(result[key])
                    acc_found = True
                    break

            if not acc_found:
                # 如果没有找到标准键，尝试其他可能性
                for k, v in result.items():
                    if 'Acc' in k and 'Stream' in k:
                        accuracies.append(v)
                        break
                else:
                    accuracies.append(0)

            # 遗忘度量
            forget_keys = ['Forgotten_Means_Stream/eval_phase/test_stream', 'Forgetting']
            forget_found = False
            for key in forget_keys:
                if key in result:
                    forgetting_scores.append(result[key])
                    forget_found = True
                    break

        # 计算核心指标
        analysis = {
            # 准确率相关
            'final_accuracy': accuracies[-1] if accuracies else 0,
            'average_accuracy': np.mean(accuracies) if accuracies else 0,
            'accuracy_std': np.std(accuracies) if accuracies else 0,

            # 持续学习指标
            'average_forgetting': np.mean(forgetting_scores) if forgetting_scores else 0,

            # 任务性能
            'task_accuracies': accuracies,
            'task_forgetting': forgetting_scores,
        }

        # 计算自定义指标
        custom_metrics = self.compute_custom_metrics()
        analysis.update(custom_metrics)

        # 打印分析结果
        print(f"\n性能指标:")
        print(f"  最终准确率 (AA): {analysis['final_accuracy']:.2%}")
        print(f"  平均准确率: {analysis['average_accuracy']:.2%} (±{analysis['accuracy_std']:.4f})")
        print(f"  平均遗忘度量: {analysis['average_forgetting']:.4f}")

        if 'estimated_backward_transfer' in analysis:
            print(f"  估计反向迁移: {analysis['estimated_backward_transfer']:.4f}")

        # 与iCaRL论文结果对比（参考值）
        print(f"\n与iCaRL论文对比 (CIFAR-100):")
        print(f"  论文报告准确率 (10任务): ~64.10%")
        print(f"  本实验准确率 ({self.config['n_experiences']}任务): {analysis['final_accuracy']:.2%}")

        return analysis

    def visualize_results(self, save_path='icarl_results.png'):
        """可视化结果"""
        if not self.results:
            print("没有可可视化的结果")
            return

        # 提取数据
        tasks = list(range(1, len(self.results) + 1))
        accuracies = []

        for result in self.results:
            # 尝试多个可能的键
            acc_keys = ['Top1_Acc_Stream/eval_phase/test_stream', 'Top1_Acc_Stream']
            for key in acc_keys:
                if key in result:
                    accuracies.append(result[key])
                    break
            else:
                # 如果没有找到，尝试搜索
                for k, v in result.items():
                    if 'Acc' in k and isinstance(v, (int, float)):
                        accuracies.append(v)
                        break
                else:
                    accuracies.append(0)

        if not accuracies or all(a == 0 for a in accuracies):
            print("没有有效的准确率数据可可视化")
            return

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
        if self.model:
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'config': self.config,
                'results': self.results,
            }, path)
            print(f"模型已保存到: {path}")

    def run(self):
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
    print("基于Avalanche框架实现 (修正版)")
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
        results = reproducer.run()
    else:
        # 默认完整实验模式
        reproducer = iCaRLReproducer()
        results = reproducer.run()

    # 打印最终总结
    if results:
        print("\n" + "="*60)
        print("实验总结")
        print("="*60)
        print(f"数据集: CIFAR-100")
        print(f"增量任务: {reproducer.config['n_experiences']}")
        print(f"最终准确率: {results['final_accuracy']:.2%}")
        print(f"平均遗忘: {results['average_forgetting']:.4f}")
        print("="*60)


if __name__ == "__main__":
    main()