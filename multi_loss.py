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