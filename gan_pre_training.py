import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import os
import cv2
import numpy as np
from torchvision import transforms
from torchvision.utils import save_image
import matplotlib.pyplot as plt
from tqdm import tqdm  # 新增 tqdm

# ---------------------------
# 生成器和判別器類別
# ---------------------------

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

# ---------------------------
# 自訂資料集類別
# ---------------------------

class StarDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        參數:
            root_dir (string): 包含所有星形圖片的目錄。
            transform (callable, optional): 可選的轉換操作。
        """
        self.root_dir = root_dir
        self.transform = transform
        # 假設所有圖片都直接位於 root_dir 下
        self.image_paths = [os.path.join(root_dir, fname) for fname in os.listdir(root_dir)
                            if fname.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        # 使用 OpenCV 讀取圖片
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # 以灰度模式讀取
        if image is None:
            raise ValueError(f"無法加載路徑 {img_path} 的圖片。")
        # 如果圖片不是 200x200，則調整大小
        if image.shape != (200, 200):
            image = cv2.resize(image, (200, 200))
        # 應用轉換
        if self.transform:
            image = self.transform(image)
        else:
            # 預設轉換：轉為 tensor 並標準化到 [0,1]
            image = transforms.ToTensor()(image)
        # 將單通道轉為三通道
        image = image.repeat(3, 1, 1)  # [1, H, W] -> [3, H, W]
        return image  # 三通道圖片

# ---------------------------
# 訓練配置
# ---------------------------

def main():
    # 超參數
    latent_dim = 100
    batch_size = 64
    lr = 0.0002
    beta1 = 0.5
    num_epochs = 300
    sample_interval = 500  # 每 500 個 batch 保存一次生成的圖片
    model_save_interval = 10  # 每 10 個 epoch 保存一次模型

    # 設備配置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用設備: {device}")

    # 創建輸出目錄
    os.makedirs("images", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    # ---------------------------
    # 資料準備
    # ---------------------------

    # 定義轉換操作
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((200, 200)),
        transforms.ToTensor(),  # 轉換為 [0,1]
    ])

    # 初始化資料集和資料加載器
    dataset = StarDataset(root_dir="./star/star", transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

    # ---------------------------
    # 初始化模型
    # ---------------------------

    generator = Generator(latent_dim).to(device)
    discriminator = Discriminator().to(device)

    # 初始化權重
    def weights_init_normal(m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0)

    generator.apply(weights_init_normal)
    discriminator.apply(weights_init_normal)

    # ---------------------------
    # 損失函數和優化器
    # ---------------------------

    adversarial_loss = nn.BCELoss()

    optimizer_G = optim.Adam(generator.parameters(), lr=lr, betas=(beta1, 0.999))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=lr/5, betas=(beta1, 0.999))

    # ---------------------------
    # 訓練循環
    # ---------------------------

    print("開始訓練...")

    for epoch in range(1, num_epochs + 1):
        # 使用 tqdm 包裝 dataloader 以顯示進度條
        progress_bar = tqdm(enumerate(dataloader, 1), total=len(dataloader), desc=f"Epoch {epoch}/{num_epochs}")

        for i, real_imgs in progress_bar:
            real_imgs = real_imgs.to(device)
            batch_size_current = real_imgs.size(0)

            # 對抗性真實標籤
            valid = torch.ones(batch_size_current, 1, device=device)
            fake = torch.zeros(batch_size_current, 1, device=device)

            # ---------------------
            # 訓練判別器
            # ---------------------
            discriminator.zero_grad()

            # 真實圖片
            output_real = discriminator(real_imgs)
            loss_real = adversarial_loss(output_real, valid)

            # 假圖片
            z = torch.randn(batch_size_current, latent_dim, 1, 1, device=device)
            gen_imgs = generator(z)
            output_fake = discriminator(gen_imgs.detach())
            loss_fake = adversarial_loss(output_fake, fake)

            # 總的判別器損失
            loss_D = loss_real + loss_fake
            loss_D.backward()
            optimizer_D.step()

            # -----------------
            # 訓練生成器
            # -----------------
            generator.zero_grad()

            # 生成圖片
            gen_imgs = generator(z)
            output = discriminator(gen_imgs)
            loss_G = adversarial_loss(output, valid)  # 希望生成器欺騙判別器

            loss_G.backward()
            optimizer_G.step()

            # 更新進度條描述
            progress_bar.set_postfix({"D 損失": loss_D.item(), "G 損失": loss_G.item()})

            # 在指定的間隔保存生成的圖片
            if (epoch * len(dataloader) + i) % sample_interval == 0:
                save_image(gen_imgs.data[:25], f"images/{epoch}_{i}.png", nrow=5, normalize=True)

        # 保存模型檢查點
        if epoch % model_save_interval == 0:
            torch.save(generator.state_dict(), f"models/generator_epoch_{epoch}.pth")
            torch.save(discriminator.state_dict(), f"models/discriminator_epoch_{epoch}.pth")
            print(f"已保存第 {epoch} 個 epoch 的模型檢查點")

        # 可選：每個 epoch 生成並保存圖片
        with torch.no_grad():
            z = torch.randn(64, latent_dim, 1, 1, device=device)
            gen_imgs = generator(z)
            save_image(gen_imgs.data, f"images/epoch_{epoch}.png", nrow=8, normalize=True)

    print("訓練完成！")

    # ---------------------------
    # 保存最終模型
    # ---------------------------

    torch.save(generator.state_dict(), "gan_models/generator_final.pth")
    torch.save(discriminator.state_dict(), "gan_models/discriminator_final.pth")
    print("已保存最終模型檢查點。")

    # ---------------------------
    # 可選：視覺化訓練結果
    # ---------------------------

    def plot_generated_images(epoch, n_row=5, n_col=5, save_path=None):
        """保存生成圖片的網格"""
        z = torch.randn(n_row * n_col, latent_dim, 1, 1, device=device)
        gen_imgs = generator(z)
        save_image(gen_imgs.data, save_path, nrow=n_col, normalize=True)
        img = cv2.imread(save_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        plt.imshow(img, cmap='gray')
        plt.axis('off')
        plt.title(f"生成圖片 - Epoch {epoch}")
        plt.show()

    # 範例使用（訓練完成後）：
    # plot_generated_images(num_epochs, save_path="images/final_generated.png")

if __name__ == '__main__':
    main()