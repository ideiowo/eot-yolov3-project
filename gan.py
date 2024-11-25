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
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            # 第二層：上採樣到 11x11
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            # 第三層：保持尺寸，增加深度
            nn.ConvTranspose2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            # 第四層：上採樣到 32x32
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            # 第五層：保持尺寸，增加深度
            nn.ConvTranspose2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            # 第六層：上採樣到 64x64
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            # 第七層：上採樣到 200x200
            nn.ConvTranspose2d(32, 16, kernel_size=11, stride=3, padding=1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            # 第八層：保持尺寸，增加深度
            nn.ConvTranspose2d(16, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            # 最終輸出層：生成 3 通道的灰度圖像
            nn.ConvTranspose2d(16, 3, kernel_size=5, stride=1, padding=1),
            nn.Sigmoid()  # 確保輸出範圍在 [0, 1]
        )

    def forward(self, z):
        x = self.model(z)
        return x  # 直接返回經過 Sigmoid 的結果

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
    # 測試判別器
    discriminator = Discriminator()

    checkpoint_epoch = 220  # 您想要加載的 epoch 數
    generator_path = f"models/generator_epoch_{checkpoint_epoch}.pth"
    discriminator_path = f"models/discriminator_epoch_{checkpoint_epoch}.pth"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if os.path.exists(generator_path) and os.path.exists(discriminator_path):
        try:
            generator.load_state_dict(torch.load(generator_path, map_location=device))
            discriminator.load_state_dict(torch.load(discriminator_path, map_location=device))
            print(f"已成功加載第 {checkpoint_epoch} 個 epoch 的生成器和判別器參數。")
        except Exception as e:
            print(f"加載模型參數時出錯: {e}")
    else:
        print(f"模型檔案未找到：{generator_path} 或 {discriminator_path}。將從頭開始訓練。")
    z = torch.randn(1, latent_dim, 1, 1)  # 隨機噪聲輸入
    generated_img = generator(z)

    # 確認生成器輸出是否為黑白
    print(f"Generated image shape: {generated_img.shape}")  # 應為 [1, 3, 200, 200]
    print(f"Unique values in generated image: {torch.unique(generated_img)}")  # 應僅包含 [0, 1]

    # 保存生成的圖片
    generated_img_np = (generated_img[0].cpu().detach().permute(1, 2, 0).numpy() * 255).astype(np.uint8)  # [C, H, W] -> [H, W, C] 並轉換為 uint8
    save_path = os.path.join("", "generated_image2.png")
    cv2.imwrite(save_path, generated_img_np)

    print(f"Generated image saved to: {save_path}")

    
    score = discriminator(generated_img)
    print(f"Discriminator output shape: {score.shape}")  # 應為 [1]
