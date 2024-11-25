import torch
import torch.nn as nn
import os 
import cv2
import numpy as np


class Generator(nn.Module):
    def __init__(self, latent_dim=100):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            # 第一層：從潛在向量生成 4x4 的特徵圖
            nn.ConvTranspose2d(latent_dim, 256, kernel_size=4, stride=1, padding=0),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),  # 替換為 LeakyReLU

            # 第二層：上採樣到 11x11
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),  # 替換為 LeakyReLU

            # 第三層：保持尺寸，增加深度
            nn.ConvTranspose2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),  # 替換為 LeakyReLU

            # 第四層：上採樣到 32x32
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),  # 替換為 LeakyReLU

            # 第五層：保持尺寸，增加深度
            nn.ConvTranspose2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),  # 替換為 LeakyReLU

            # 第六層：上採樣到 64x64
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),  # 替換為 LeakyReLU

            # 第七層：上採樣到 200x200
            nn.ConvTranspose2d(32, 16, kernel_size=11, stride=3, padding=1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),  # 替換為 LeakyReLU

            # 第八層：保持尺寸，增加深度
            nn.ConvTranspose2d(16, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),  # 替換為 LeakyReLU

            # 最終輸出層：生成 1 通道的灰度圖像
            nn.ConvTranspose2d(16, 1, kernel_size=5, stride=1, padding=1),
            nn.Sigmoid()  # 確保輸出範圍在 [0, 1]
        )

    def forward(self, z):
        x = self.model(z)
        binary_output = (x > 0.5).float()  # 二值化處理
        rgb_output = binary_output.repeat(1, 3, 1, 1)  # 將單通道數據複製為三通道
        return rgb_output




# 判別器 (Discriminator)
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            # 第一層：從 3 通道 (RGB) 到 64 通道，特徵提取
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),  # 200x200 -> 100x100
            nn.LeakyReLU(0.2, inplace=True),

            # 第二層：從 64 通道到 128 通道，進一步提取特徵
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),  # 100x100 -> 50x50
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            # 第三層：從 128 通道到 256 通道
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),  # 50x50 -> 25x25
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            # 第四層：從 256 通道到 512 通道
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),  # 25x25 -> 12x12
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),

            # 第五層：從 512 通道到 1 通道，縮小到 1x1
            nn.Conv2d(512, 1, kernel_size=12, stride=1, padding=0),  # 12x12 -> 1x1
            nn.Sigmoid()  # 輸出範圍 [0, 1]
        )

    def forward(self, img):
        output = self.model(img)
        return output.view(-1, 1)


# 測試模型結構
if __name__ == "__main__":
    # 測試生成器
    latent_dim = 100
    generator = Generator(latent_dim)
    z = torch.randn(1, latent_dim, 1, 1)  # 隨機噪聲輸入
    generated_img = generator(z)

    # 確認生成器輸出是否為黑白
    print(f"Generated image shape: {generated_img.shape}")  # 應為 [1, 3, 200, 200]
    print(f"Unique values in generated image: {torch.unique(generated_img)}")  # 應僅包含 [0, 1]

    # 保存生成的圖片
    generated_img_np = (generated_img[0].cpu().detach().permute(1, 2, 0).numpy() * 255).astype(np.uint8)  # [C, H, W] -> [H, W, C] 並轉換為 uint8
    save_path = os.path.join("", "generated_image.png")
    cv2.imwrite(save_path, generated_img_np)

    print(f"Generated image saved to: {save_path}")

    # 測試判別器
    discriminator = Discriminator()
    score = discriminator(generated_img)
    print(f"Discriminator output shape: {score.shape}")  # 應為 [1]
