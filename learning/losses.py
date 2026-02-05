import torch
import torch.nn as nn


class ElasticNetLoss(nn.Module):
    """
    L = CE + α * λ1 * ||θ||_1
    """

    def __init__(self, alpha=0.5, l1_lambda=1e-3):
        super().__init__()
        self.alpha = alpha
        self.l1_lambda = l1_lambda
        self.ce = nn.CrossEntropyLoss()

    def forward(self, outputs, targets, model):
        ce_loss = self.ce(outputs, targets)
        l1_reg = sum(
            torch.norm(p, 1)
            for p in model.parameters()
            if p.requires_grad
        )
        return ce_loss + self.alpha * self.l1_lambda * l1_reg