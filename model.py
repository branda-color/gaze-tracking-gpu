import torch
from pytorch_lightning import LightningModule
from torch import nn
from torchinfo import summary
from torchvision import models
from torchvision.models import vgg16, VGG16_Weights
from kalman_filter import KalmanFilter

class SELayer(nn.Module):

    """
    Squeeze-and-Excitation layer

    https://github.com/moskomule/senet.pytorch/blob/master/senet/se_module.py

    增強網路的表示能力
    1.Squeeze:這個操作使用全局平均池化來生成通道方向的統計資訊。它計算一個描述符來總結每個通道的全局分佈。
    2.Excitation:這一步涉及一個全連接(FC)層,它根據擠壓後的統計資訊學習每個通道的重要性。它應用 sigmoid 激活函數來產生一組權重,這些權重可以乘以原始輸入特徵,以重新校準通道響應。
    """

    def __init__(self, channel, reduction=16):
   
        super(SELayer, self).__init__()
       
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # Squeeze
       
        self.fc = nn.Sequential(  # Excitation (similar to attention)
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
            
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c) 
        y = self.fc(y).view(b, c, 1, 1) 
        return x * y.expand_as(x)  


class FinalModel(LightningModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.subject_biases = nn.Parameter(torch.zeros(15 * 2, 2))  # pitch and yaw offset for the 

        # 初始化卡爾曼濾波器
        # self.kf_pitch = KalmanFilter(
        #     initial_state=[0, 0],
        #     state_covariance=[[0.1, 0], [0, 0.1]],
        #     process_noise=[[0.05, 0], [0, 0.05]],
        #     measurement_noise=[[0.05]]
        # )
        # self.kf_yaw = KalmanFilter(
        #     initial_state=[0, 0],
        #     state_covariance=[[0.1, 0], [0, 0.1]],
        #     process_noise=[[0.05, 0], [0, 0.05]],
        #     measurement_noise=[[0.05]]
        # )

        self.cnn_face = nn.Sequential(
            vgg16(weights=VGG16_Weights.IMAGENET1K_V1).features[:9],  # first four convolutional layers of VGG16 pretrained on ImageNet
            nn.Conv2d(128, 64, kernel_size=(1, 1), stride=(1, 1), padding='same'),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding='valid', dilation=(2, 2)),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding='valid', dilation=(3, 3)),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding='valid', dilation=(5, 5)),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding='valid', dilation=(11, 11)),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.BatchNorm2d(128),
            SELayer(128)
        )

        self.cnn_eye = nn.Sequential(
            vgg16(weights=VGG16_Weights.IMAGENET1K_V1).features[:9],  # first four convolutional layers of VGG16 pretrained on ImageNet
            nn.Conv2d(128, 64, kernel_size=(1, 1), stride=(1, 1), padding='same'),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding='valid', dilation=(2, 2)),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding='valid', dilation=(3, 3)),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding='valid', dilation=(4, 5)),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding='valid', dilation=(5, 11)),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.BatchNorm2d(128),
            SELayer(128)
        )

        self.fc_face = nn.Sequential(
            nn.Flatten(),
            nn.Linear(6 * 6 * 128, 256),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.BatchNorm1d(256),
            nn.Linear(256, 64),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.BatchNorm1d(64),
        )

        self.cnn_eye2fc = nn.Sequential(
            #SELayer(256),

            nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding='same'),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.BatchNorm2d(256),

            #SELayer(256),

            nn.Conv2d(256, 128, kernel_size=(3, 3), stride=(1, 1), padding='same'),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.BatchNorm2d(128),

            #SELayer(128),
        )

        self.fc_eye = nn.Sequential(
            nn.Flatten(),
            nn.Linear(4 * 6 * 128, 512),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.BatchNorm1d(512),
        )

        self.fc_eyes_face = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(576, 256),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.BatchNorm1d(256),
            nn.Dropout(p=0.5),
            nn.Linear(256, 2),
        )

    def forward(self, person_idx: torch.Tensor, full_face: torch.Tensor, right_eye: torch.Tensor, left_eye: torch.Tensor):
        out_cnn_face = self.cnn_face(full_face)
        out_fc_face = self.fc_face(out_cnn_face)
        #print(f"注意看out_cnn_face shape: {out_cnn_face.shape}")

        out_cnn_right_eye = self.cnn_eye(right_eye)
        out_cnn_left_eye = self.cnn_eye(left_eye)
        out_cnn_eye = torch.cat((out_cnn_right_eye, out_cnn_left_eye), dim=1)

        cnn_eye2fc_out = self.cnn_eye2fc(out_cnn_eye)  # feature fusion
        out_fc_eye = self.fc_eye(cnn_eye2fc_out)

        fc_concatenated = torch.cat((out_fc_face, out_fc_eye), dim=1)
        t_hat = self.fc_eyes_face(fc_concatenated)  # subject-independent term

    # 卡爾曼濾波處理
        # filtered_pitch = []
        # filtered_yaw = []
        # for batch_idx in range(t_hat.size(0)):  # 批次處理
        #     pitch, yaw = t_hat[batch_idx].cpu().detach().numpy()
        #     filtered_pitch.append(self.kf_pitch.update(pitch)[0])  # 只提取位置
        #     filtered_yaw.append(self.kf_yaw.update(yaw)[0])

        # 將濾波後的結果轉回 Tensor 格式
        #filtered_output = torch.tensor(list(zip(filtered_pitch, filtered_yaw))).to(t_hat.device)

        #return filtered_output  + self.subject_biases[person_idx].squeeze(1)  # t_hat + subject-dependent bias term

        return t_hat + self.subject_biases[person_idx].squeeze(1)


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