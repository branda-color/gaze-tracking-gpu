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
        #創建一個自適應平均池化層，將每個通道的特徵圖壓縮為一個單一的數值
        self.fc = nn.Sequential(  # Excitation (similar to attention)
        #創建一個序列的全連接層，用於 Excitation 操作，這裡的結構類似於注意力機制。
            nn.Linear(channel, channel // reduction, bias=False),
            #第一個全連接層，將通道數從 channel 縮減到 channel 整除 reduction，不使用偏置項。
            nn.ReLU(inplace=True),
            #使用 ReLU 激活函數，將所有負值設置為 0，並保持正值不變。inplace=True 表示直接在輸入張量上進行操作，以節省內存。
            nn.Linear(channel // reduction, channel, bias=False),
            #第二個全連接層，將通道數從 channel // reduction 重新映射回 channel，同樣不使用偏置項。
            nn.Sigmoid()
            #使用 Sigmoid 激活函數，將輸出值限制在 0 到 1 之間，這些值將作為通道的權重。
        )

    def forward(self, x):
    # 前向傳遞 (forward 方法):
    # 接受輸入張量 x,應用擠壓操作以獲得通道統計資訊,然後將這些統計資訊傳遞給激發層。
    # 最後,它通過學習到的權重來縮放原始輸入 x。
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c) #擠壓（Squeeze）：使用 avg_pool 層將每個通道的特徵圖縮減到 1x1 大小，並計算平均值。
        y = self.fc(y).view(b, c, 1, 1) #通過全連接層將擠壓後的平均值轉化為通道權重。
        return x * y.expand_as(x)   #使用學到的通道權重來調整原始輸入張量 x 的每個通道的響應。


class FinalModel(LightningModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.subject_biases = nn.Parameter(torch.zeros(15 * 2, 2))  # pitch and yaw offset for the original and mirrored participant
        #這行代碼創建了一個大小為 (30, 2) 的張量，這意味著你最多可以為 30 個參與者定義偏置（包括正向和鏡像的視角）。
        #這個參數是為每個人定義的偏置（bias），表示每個參與者（包括鏡像的參與者）的 pitch 和 yaw 偏移。nn.Parameter 表示這個張量是可學習的(nn.Parameter是特殊類型的張量>>會自動優化)。

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
        #面部特徵提取:使用 VGG16 模型的前 9 層作為面部特徵提取的基礎，並添加一些卷積層和激活函數。這些層將提取面部圖像的特徵。

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
        #眼睛特徵提取:同樣使用 VGG16 的前 9 層作為眼睛特徵提取的基礎，並添加一些卷積層和激活函數，這些層將提取眼睛圖像的特徵。

        self.fc_face = nn.Sequential(
            nn.Flatten(),
            nn.Linear(6 * 6 * 128, 256),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(256),
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(64),
        )
        #面部特徵的全連接層:將面部特徵展平並通過一系列全連接層進行處理，最終輸出 64 維的特徵。

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
        #提取強化眼部特徵:這個模塊用於處理眼睛特徵。它融合了來自左右眼的卷積輸出，並通過一系列的卷積層和 SELayer（一種注意力機制）來提取和強化眼睛特徵。

        self.fc_eye = nn.Sequential(
            nn.Flatten(),
            nn.Linear(4 * 6 * 128, 512),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),
        )
        #眼睛特徵的全連接層:將眼睛特徵展平並通過全連接層進行處理，最終輸出 512 維的特徵。

        self.fc_eyes_face = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(576, 256),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(256),
            nn.Dropout(p=0.5),
            nn.Linear(256, 2),
        )
        #特徵融合:將面部和眼睛的特徵進行融合，通過全連接層進行處理，最終輸出 2 維的預測結果。

    def forward(self, person_idx: torch.Tensor, full_face: torch.Tensor, right_eye: torch.Tensor, left_eye: torch.Tensor):
    # 整個 forward 方法的目的是從面部和眼睛圖像中提取特徵，進行融合，並結合個體的偏差來生成最終的預測結果。這段代碼的結構清晰地展示了如何處理面部和眼睛的圖像，並進行特徵融合，最終輸出一個包含 2 個數值的預測結果，這些數值可能代表頭部或視線的俯仰角（Pitch）和偏航角（Yaw）。
        out_cnn_face = self.cnn_face(full_face)
        #將完整的面部圖像傳遞給 cnn_face 模型，提取面部特徵。
        out_fc_face = self.fc_face(out_cnn_face)
        #將提取到的面部特徵傳遞給全連接層 fc_face，進行進一步的處理。

        out_cnn_right_eye = self.cnn_eye(right_eye)
        
        out_cnn_left_eye = self.cnn_eye(left_eye)
        #分別將右眼和左眼的圖像傳遞給 cnn_eye 模型，提取眼睛特徵。
        out_cnn_eye = torch.cat((out_cnn_right_eye, out_cnn_left_eye), dim=1)
        #將右眼和左眼的特徵在通道維度上進行拼接，生成 out_cnn_eye，這樣可以將兩個眼睛的特徵合併為一個張量。

        cnn_eye2fc_out = self.cnn_eye2fc(out_cnn_eye)  # feature fusion
        #將拼接後的眼睛特徵傳遞給 cnn_eye2fc 模型，進行進一步的特徵提取和融合。
        out_fc_eye = self.fc_eye(cnn_eye2fc_out)
        #將融合後的眼睛特徵傳遞給全連接層 fc_eye，進行進一步處理。

        fc_concatenated = torch.cat((out_fc_face, out_fc_eye), dim=1)
        #將處理後的面部特徵和眼睛特徵在通道維度上進行拼接，生成 fc_concatenated。
        t_hat = self.fc_eyes_face(fc_concatenated)  # subject-independent term
        #將拼接的特徵傳遞給 fc_eyes_face 全連接層，生成最終的預測結果 t_hat。這個預測結果代表了模型對注視點或頭部姿態的預測，通常是兩個數值（例如，Pitch 和 Yaw）。

        return t_hat + self.subject_biases[person_idx].squeeze(1)  # t_hat + subject-dependent bias term

        #根據 person_idx 獲取對應參與者的偏差，這是一個可學習的參數，表示每個參與者的特定偏差。
        #將預測結果 t_hat 與對應參與者的偏差相加，返回最終的預測結果。

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