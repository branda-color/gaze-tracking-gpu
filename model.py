import torch
from pytorch_lightning import LightningModule
from torch import nn
from torchinfo import summary
from torchvision import models

# 實現 Squeeze-and-Excitation 層，這是一種通過學習通道之間的關係來增強模型性能的技術。通過 Squeeze 操作獲取通道統計，再通過 Excitation 層學習每個通道的重要性，最終調整輸入特徵的響應。
class SELayer(nn.Module):
    #class 子類名(父類名):
    """
    Squeeze-and-Excitation layer

    https://github.com/moskomule/senet.pytorch/blob/master/senet/se_module.py

    增強網路的表示能力
    1.Squeeze:這個操作使用全局平均池化來生成通道方向的統計資訊。它計算一個描述符來總結每個通道的全局分佈。
    2.Excitation:這一步涉及一個全連接(FC)層,它根據擠壓後的統計資訊學習每個通道的重要性。它應用 sigmoid 激活函數來產生一組權重,這些權重可以乘以原始輸入特徵,以重新校準通道響應。
    """

    def __init__(self, channel, reduction=16):
    # 初始化 (__init__ 方法):
    # 在 Python 中，類的方法（包括初始化方法 __init__）的第一個參數通常命名為 self，這是一個約定俗成的做法。
    # 接受兩個參數：channel（輸入通道數）和 reduction（可選，默認為 16，用於控制激發過程中的維度縮減）。
        super(SELayer, self).__init__()
        #super() 是一個內建函數，用於獲取父類（超類）的對象。它可以用來調用父類的方法
        # 在 super(SELayer, self) 中，SELayer 是當前類的名稱。這告訴 super() 我們希望獲取 SELayer 的父類（在這個例子中是 nn.Module）的引用。
        # 調用父類的初始化方法：這行代碼確保 SELayer 的父類 nn.Module 的初始化方法被調用，這樣 SELayer 實例就能繼承 nn.Module 的所有屬性和方法，並正確設置其狀態。
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # Squeeze
        self.fc = nn.Sequential(  # Excitation (similar to attention)
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
    # 前向傳遞 (forward 方法):
    # 接受輸入張量 x,應用擠壓操作以獲得通道統計資訊,然後將這些統計資訊傳遞給激發層。
    # 最後,它通過學習到的權重來縮放原始輸入 x。
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class FinalModel(LightningModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.subject_biases = nn.Parameter(torch.zeros(15 * 2, 2))  # pitch and yaw offset for the original and mirrored participant

        self.cnn_face = nn.Sequential(
            models.vgg16(pretrained=True).features[:9],  # first four convolutional layers of VGG16 pretrained on ImageNet
            nn.Conv2d(128, 64, kernel_size=(1, 1), stride=(1, 1), padding='same'),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding='valid', dilation=(2, 2)),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding='valid', dilation=(3, 3)),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding='valid', dilation=(5, 5)),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding='valid', dilation=(11, 11)),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(128),
        )

        self.cnn_eye = nn.Sequential(
            models.vgg16(pretrained=True).features[:9],  # first four convolutional layers of VGG16 pretrained on ImageNet
            nn.Conv2d(128, 64, kernel_size=(1, 1), stride=(1, 1), padding='same'),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding='valid', dilation=(2, 2)),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding='valid', dilation=(3, 3)),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding='valid', dilation=(4, 5)),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding='valid', dilation=(5, 11)),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(128),
        )

        self.fc_face = nn.Sequential(
            nn.Flatten(),
            nn.Linear(6 * 6 * 128, 256),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(256),
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(64),
        )

        self.cnn_eye2fc = nn.Sequential(
            SELayer(256),

            nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding='same'),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(256),

            SELayer(256),

            nn.Conv2d(256, 128, kernel_size=(3, 3), stride=(1, 1), padding='same'),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(128),

            SELayer(128),
        )

        self.fc_eye = nn.Sequential(
            nn.Flatten(),
            nn.Linear(4 * 6 * 128, 512),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),
        )

        self.fc_eyes_face = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(576, 256),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(256),
            nn.Dropout(p=0.5),
            nn.Linear(256, 2),
        )

    def forward(self, person_idx: torch.Tensor, full_face: torch.Tensor, right_eye: torch.Tensor, left_eye: torch.Tensor):
        out_cnn_face = self.cnn_face(full_face)
        out_fc_face = self.fc_face(out_cnn_face)

        out_cnn_right_eye = self.cnn_eye(right_eye)
        out_cnn_left_eye = self.cnn_eye(left_eye)
        out_cnn_eye = torch.cat((out_cnn_right_eye, out_cnn_left_eye), dim=1)

        cnn_eye2fc_out = self.cnn_eye2fc(out_cnn_eye)  # feature fusion
        out_fc_eye = self.fc_eye(cnn_eye2fc_out)

        fc_concatenated = torch.cat((out_fc_face, out_fc_eye), dim=1)
        t_hat = self.fc_eyes_face(fc_concatenated)  # subject-independent term

        return t_hat + self.subject_biases[person_idx].squeeze(1)  # t_hat + subject-dependent bias term


if __name__ == '__main__':
    model = FinalModel()
    model.summarize(max_depth=1)

    print(model.cnn_face)

    batch_size = 16
    summary(model, [
        (batch_size, 1),
        (batch_size, 3, 96, 96),  # full face
        (batch_size, 3, 64, 96),  # right eye
        (batch_size, 3, 64, 96)  # left eye
    ], dtypes=[torch.long, torch.float, torch.float, torch.float])