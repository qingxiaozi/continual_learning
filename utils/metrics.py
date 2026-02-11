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
    def compute_metrics(seen_domains, accuracy_history):
        """
        计算增量学习的核心指标：AA, FM, BWT

        Args:
            seen_domains: 已见领域列表
            accuracy_history: 准确率历史字典，格式为 {domain: [acc_session1, acc_session2, ...]}

        Returns:
            dict: 包含 k, AA, FM, BWT 的字典
        """
        k = len(seen_domains)
        if k == 0:
            return {"k": 0}

        # 当前各域的最新准确率
        current_accuracies = {
            domain: accuracy_history[domain][-1]
            for domain in seen_domains
            if domain in accuracy_history and accuracy_history[domain]
        }

        # 1. Average Accuracy (AA_k)
        aa_k = np.mean(list(current_accuracies.values())) if current_accuracies else 0.0

        # 2. Forgetting Measure (FM_k)
        fm_k = 0.0
        if k > 1:
            forgetting_vals = []
            for domain in seen_domains[:-1]:
                if domain in accuracy_history and len(accuracy_history[domain]) >= 2:
                    prev_accuracies = accuracy_history[domain][:-1]
                    max_acc = max(prev_accuracies)
                    current_acc = accuracy_history[domain][-1]
                    forgetting_vals.append(max_acc - current_acc)
            fm_k = np.mean(forgetting_vals) if forgetting_vals else 0.0

        # 3. Backward Transfer (BWT_k)
        bwt_k = 0.0
        if k > 1:
            bwt_vals = []
            for domain in seen_domains[:-1]:
                if domain in accuracy_history and len(accuracy_history[domain]) >= 2:
                    first_acc = accuracy_history[domain][0]
                    current_acc = accuracy_history[domain][-1]
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

