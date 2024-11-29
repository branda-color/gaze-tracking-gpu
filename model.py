import torch
from pytorch_lightning import LightningModule
from torch import nn
from torchinfo import summary
from torchvision import models
from torchvision.models import vgg16, VGG16_Weights
from torchvision.models import resnet18, ResNet18_Weights
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
        self.kf_pitch = KalmanFilter(
            initial_state=[0, 0],
            state_covariance=[[1, 0], [0, 1]],
            process_noise=[[0.01, 0], [0, 0.01]],
            measurement_noise=[[0.1]]
        )
        self.kf_yaw = KalmanFilter(
            initial_state=[0, 0],
            state_covariance=[[1, 0], [0, 1]],
            process_noise=[[0.01, 0], [0, 0.01]],
            measurement_noise=[[0.1]]
        )

        # 初始化 ResNet18
        resnet = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)

        self.cnn_face = nn.Sequential(
            resnet.conv1,  # 初始卷积层，输入 [32, 3, 96, 96] -> 输出 [32, 64, 96, 96]
            resnet.bn1,    # 批量正则化
            resnet.relu,   # 激活函数
            #resnet.maxpool,  # 最大池化，输入 [32, 64, 96, 96] -> 输出 [32, 64, 48, 48]
            resnet.layer1,  # ResNet 层1，输入 [32, 64, 48, 48] -> 输出 [32, 64, 48, 48]
            resnet.layer2,  # ResNet 层2，输入 [32, 64, 48, 48] -> 输出 [32, 128, 24, 24]
            
            # 新增卷积层和批量正则化
            nn.Conv2d(128, 64, kernel_size=(1, 1), stride=(1, 1), padding='same'),  # 降维卷积，输出 [32, 64, 24, 24]
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            
            nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding='valid', dilation=(2, 2)),  # 空洞卷积
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            
            nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding='valid', dilation=(2, 2)),  # 降低 dilation
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            
            nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding='valid', dilation=(3, 3)),  # 空洞卷积
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(128),
            
            nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding='same'),  # 移除 dilation，输出 [32, 128, 6, 6]
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(128),
            
            nn.AdaptiveAvgPool2d((6, 6))  # 固定輸出大小為 [32, 128, 6, 6]
        )


        self.cnn_eye = nn.Sequential(
            # 使用 ResNet18 的前几层
            resnet18(weights=ResNet18_Weights.IMAGENET1K_V1).conv1,
            resnet18(weights=ResNet18_Weights.IMAGENET1K_V1).bn1,
            resnet18(weights=ResNet18_Weights.IMAGENET1K_V1).relu,
            resnet18(weights=ResNet18_Weights.IMAGENET1K_V1).maxpool,
            resnet18(weights=ResNet18_Weights.IMAGENET1K_V1).layer1,
            resnet18(weights=ResNet18_Weights.IMAGENET1K_V1).layer2,
            # 保持与原始输出大小一致的调整
            nn.Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1), padding='same'),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 64, kernel_size=(3, 3), stride=(1, 1), padding='same'),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding='same'),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(128),
            nn.AdaptiveAvgPool2d((4, 6))  # 调整输出到 (batch_size, 128, 4, 6)
        )

        self.fc_face = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128*6*6, 256),  # 使用動態計算的輸入大小
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
        #print('傳入的full_face:',full_face.shape)
        out_cnn_face = self.cnn_face(full_face)
        #print('輸出的cnn_face:',out_cnn_face.shape)
        out_fc_face = self.fc_face(out_cnn_face.view(out_cnn_face.size(0), -1))
        #print("FC face output shape:", out_fc_face.shape)

        out_cnn_right_eye = self.cnn_eye(right_eye)
        #print('傳入的cnn_eye:',out_cnn_right_eye.shape)

        out_cnn_left_eye = self.cnn_eye(left_eye)
        out_cnn_eye = torch.cat((out_cnn_right_eye, out_cnn_left_eye), dim=1)

        cnn_eye2fc_out = self.cnn_eye2fc(out_cnn_eye)  # feature fusion
        out_fc_eye = self.fc_eye(cnn_eye2fc_out)

        fc_concatenated = torch.cat((out_fc_face, out_fc_eye), dim=1)
        t_hat = self.fc_eyes_face(fc_concatenated)  # subject-independent term

    # 卡爾曼濾波處理
        filtered_pitch = []
        filtered_yaw = []
        for batch_idx in range(t_hat.size(0)):  # 批次處理
            pitch, yaw = t_hat[batch_idx].cpu().detach().numpy()
            filtered_pitch.append(self.kf_pitch.update(pitch)[0])  # 只提取位置
            filtered_yaw.append(self.kf_yaw.update(yaw)[0])

        # 將濾波後的結果轉回 Tensor 格式
        filtered_output = torch.tensor(list(zip(filtered_pitch, filtered_yaw))).to(t_hat.device)

        return filtered_output  + self.subject_biases[person_idx].squeeze(1)  # t_hat + subject-dependent bias term


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