# loss_functions.py

import torch
import torch.nn.functional as F
import numpy as np

# 角度損失函數
def angular_loss(predictions, targets):
    pred_vector = torch.stack([
        torch.cos(predictions[:, 0]) * torch.cos(predictions[:, 1]),
        torch.sin(predictions[:, 0]),
        torch.cos(predictions[:, 0]) * torch.sin(predictions[:, 1])
    ], dim=1)
    target_vector = torch.stack([
        torch.cos(targets[:, 0]) * torch.cos(targets[:, 1]),
        torch.sin(targets[:, 0]),
        torch.cos(targets[:, 0]) * torch.sin(targets[:, 1])
    ], dim=1)
    cos_similarity = torch.sum(pred_vector * target_vector, dim=1)
    return torch.mean(1 - cos_similarity)

# 正則化項 (L1 + L2)
def regularization_loss(model):
    l1_reg = torch.sum(torch.abs(model.subject_biases))
    l2_reg = torch.sum(model.subject_biases ** 2)
    return 0.01 * (l1_reg + l2_reg)

# 多任務損失
class MultiTaskLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weights = torch.nn.Parameter(torch.ones(2))

    def forward(self, loss1, loss2):
        weighted_loss1 = self.weights[0] * loss1
        weighted_loss2 = self.weights[1] * loss2
        return weighted_loss1 + weighted_loss2