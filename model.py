import torch
from pytorch_lightning import LightningModule
from torch import nn
from torchinfo import summary
from torchvision import models
from torchvision.models import vgg16, VGG16_Weights
from torchvision.models import resnet18, ResNet18_Weights
from kalman_filter import KalmanFilter
import torch.nn.functional as F

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

        #self.subject_biases = nn.Parameter(torch.zeros(15 * 2, 2))  #舊的
        
        num_subjects = 30  # 你可以改成實際用到的最大 index（含 flip）
        self.bias_mlp = nn.Sequential(
            nn.Embedding(num_subjects, 32),
            nn.ReLU(),
            nn.Linear(32, 2)
        )

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
            SELayer(256),

            nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding='same'),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),

            nn.BatchNorm2d(256),

            SELayer(256),

            nn.Conv2d(256, 128, kernel_size=(3, 3), stride=(1, 1), padding='same'),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),

            nn.BatchNorm2d(128),

            SELayer(128),
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
        #print('傳入的full_face:',full_face.shape)
        out_cnn_face = self.cnn_face(full_face)
        out_fc_face = self.fc_face(out_cnn_face)
        #print(f"注意看out_cnn_face shape: {out_cnn_face.shape}")

        out_cnn_right_eye = self.cnn_eye(right_eye)
        #print('傳入的cnn_eye:',out_cnn_right_eye.shape)

        out_cnn_left_eye = self.cnn_eye(left_eye)
        out_cnn_eye = torch.cat((out_cnn_right_eye, out_cnn_left_eye), dim=1)

        cnn_eye2fc_out = self.cnn_eye2fc(out_cnn_eye)  # feature fusion
        out_fc_eye = self.fc_eye(cnn_eye2fc_out)

        fc_concatenated = torch.cat((out_fc_face, out_fc_eye), dim=1)
        t_hat = self.fc_eyes_face(fc_concatenated)  # subject-independent term

        #return t_hat + self.subject_biases[person_idx].squeeze(1)  舊的

        b_hat = self.bias_mlp(person_idx)  # ⬅ 用 MLP 動態產生偏移
        return t_hat + b_hat
        #return t_hat


    
    def get_subject_independent_output(self, full_face: torch.Tensor, right_eye: torch.Tensor, left_eye: torch.Tensor) -> torch.Tensor:
        """
        Forward without subject bias term. Used for calibration (Eq. 3).
        """
        out_cnn_face = self.cnn_face(full_face)
        out_fc_face = self.fc_face(out_cnn_face)

        out_cnn_right_eye = self.cnn_eye(right_eye)
        out_cnn_left_eye = self.cnn_eye(left_eye)
        out_cnn_eye = torch.cat((out_cnn_right_eye, out_cnn_left_eye), dim=1)

        cnn_eye2fc_out = self.cnn_eye2fc(out_cnn_eye)
        out_fc_eye = self.fc_eye(cnn_eye2fc_out)

        fc_concatenated = torch.cat((out_fc_face, out_fc_eye), dim=1)
        t_hat = self.fc_eyes_face(fc_concatenated)

        return t_hat

if __name__ == '__main__':
    print("Initializing model...")
    model = FinalModel()
    print("Model initialized successfully!")
    model.summarize(max_depth=1)

    batch_size = 16
    summary(model, [
        (batch_size, 1),
        (batch_size, 3, 96, 96),  # full face
        (batch_size, 3, 64, 96),  # right eye
        (batch_size, 3, 64, 96)  # left eye
    ], dtypes=[torch.long, torch.float, torch.float, torch.float])