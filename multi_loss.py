# loss_functions.py

import torch
import torch.nn.functional as F
import numpy as np

# 角度損失函數
def pitchyaw_to_3d_vector(pitchyaw: torch.Tensor) -> torch.Tensor:
    """
    2D pitch/yaw 轉換成 3D 視線向量
    """
    return torch.stack([
        -torch.cos(pitchyaw[:, 0]) * torch.sin(pitchyaw[:, 1]),  # x 分量
        -torch.sin(pitchyaw[:, 0]),  # y 分量
        -torch.cos(pitchyaw[:, 0]) * torch.cos(pitchyaw[:, 1])  # z 分量
    ], dim=1)

def angular_loss(y_pred, y_true):
    y_pred_3d = pitchyaw_to_3d_vector(y_pred)  # 轉換成 3D 向量
    y_true_3d = pitchyaw_to_3d_vector(y_true)

    cos_sim = F.cosine_similarity(y_pred_3d, y_true_3d, dim=1)
    return torch.mean(1 - cos_sim)

#MSE加上angular_loss(加上權重)
def angular_mix_mse(y_pred, y_true, alpha=0.5):
    """
    結合 MSE Loss 與 Angular Loss
    :param y_pred: 預測值 (batch, 2)
    :param y_true: 目標值 (batch, 2)
    :param alpha: 平衡係數 (0~1)，決定 MSE Loss 與 Angular Loss 的權重
    :return: 混合損失值
    """
    mse = F.mse_loss(y_pred, y_true)  # 傳統 MSE Loss
    ang = angular_loss(y_pred, y_true)  # 角度損失

    return alpha * mse + (1 - alpha) * ang

def l1_regularization(model, l1_lambda=1e-5):
    """
    計算 L1 正則化 (Lasso Regularization)
    :param model: 需要正則化的神經網絡模型
    :param l1_lambda: L1 正則化係數，控制 L1 對損失的影響程度
    :return: L1 正則化損失
    """
    l1_reg = torch.tensor(0.0, device=next(model.parameters()).device)
    for param in model.parameters():
        l1_reg += torch.norm(param, p=1)  # 計算所有權重的 L1 損失
    return l1_lambda * l1_reg  # 乘上權重係數

# 多任務損失
class MultiTaskLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weights = torch.nn.Parameter(torch.ones(2))

    def forward(self, loss1, loss2):
        weighted_loss1 = self.weights[0] * loss1
        weighted_loss2 = self.weights[1] * loss2
        return weighted_loss1 + weighted_loss2