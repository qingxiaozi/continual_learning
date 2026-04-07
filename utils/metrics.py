import os
import time
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
from datetime import datetime

class IncrementalMetricsCalculator:
    """
    增量学习指标计算器
    不保存状态，只提供计算函数
    """
    @staticmethod
    def compute_metrics(seen_tasks, accuracy_history):
        """
        计算增量学习的核心指标：AA, FM, BWT

        根据论文公式：
        - a_{k,j} = 在第 k 个任务学完后，在第 j 个任务测试集上的准确率
        - AA_k = (1/k) * sum(a_{k,j}) for j in [1,k] = 当前所有已见任务的平均准确率
        - FM_k = (1/(k-1)) * sum(max_acc_j - a_{k,j}) for j in [1,k-1] = 遗忘率
        - BWT_k = (1/(k-1)) * sum(a_{k,j} - a_{j,j}) for j in [1,k-1] = 后向迁移

        Args:
            seen_tasks: 已见任务列表 [(domain, sub_idx), ...]
            accuracy_history: 准确率历史字典，格式为 {(domain, sub_idx): [acc1, acc2, ...]}

        Returns:
            dict: 包含 k, AA, FM, BWT 的字典
        """
        k = len(seen_tasks)
        if k == 0:
            return {"k": 0, "AA": 0.0, "FM": 0.0, "BWT": 0.0}

        current_accuracies = {}
        for task in seen_tasks:
            if task in accuracy_history and accuracy_history[task]:
                current_accuracies[task] = accuracy_history[task][-1]

        aa_k = np.mean(list(current_accuracies.values())) if current_accuracies else 0.0

        fm_k = 0.0
        if k > 1:
            forgetting_vals = []
            for task in seen_tasks[:-1]:
                if task in accuracy_history and len(accuracy_history[task]) >= 1:
                    max_acc = max(accuracy_history[task])
                    current_acc = accuracy_history[task][-1]
                    forgetting_vals.append(max_acc - current_acc)
            fm_k = np.mean(forgetting_vals) if forgetting_vals else 0.0

        bwt_k = 0.0
        if k > 1:
            bwt_vals = []
            for i, task in enumerate(seen_tasks[:-1]):
                if task in accuracy_history and len(accuracy_history[task]) >= 1:
                    first_acc = accuracy_history[task][0]
                    current_acc = accuracy_history[task][-1]
                    bwt_vals.append(current_acc - first_acc)
            bwt_k = np.mean(bwt_vals) if bwt_vals else 0.0

        return {
            "k": k,
            "AA": aa_k,
            "FM": fm_k,
            "BWT": bwt_k,
        }

    @staticmethod
    def compute_aia(aa_history):
        """计算 Average Incremental Accuracy"""
        return np.mean(aa_history) if aa_history else 0.0

