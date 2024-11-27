import os
import cv2
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np
from gan import Generator, Discriminator
from util.eot import eot
import json

# 確保 yolov3 的路徑在搜索路徑中
yolo3_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'yolov3'))
if yolo3_path not in sys.path:
    sys.path.append(yolo3_path)
from yolov3.models.common import DetectMultiBackend
from optimize_parametsers import personal_non_max_suppression

torch.autograd.set_detect_anomaly(True)

# 自定義 letterbox 函數
def letterbox(im, new_shape=(416, 416), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True):
    shape = im.shape[:2]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:
        r = min(r, 1.0)

    ratio = r, r
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    dw /= 2
    dh /= 2

    if shape[::-1] != new_unpad:
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return im, ratio, (dw, dh)

# 配置參數
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
latent_dim = 100  # 生成器的輸入隨機噪聲維度
batch_size = 18  # 批次大小，需為 N 的倍數
N = 4  # 每組貼片數量
img_size = (720, 1280)  # 基底圖像大小
patch_size = (220, 220)  # 貼片大小
alpha = 0.5  # 攻擊損失的權重
num_epochs = 500  # 總訓練輪數
save_interval = 5  # 每隔幾輪保存模型
learning_rate = 0.0002

# 資料集路徑
base_dataset_path = './dataset'
patch_dataset_path = './star/star'

# 創建保存資料夾
os.makedirs("final_generated_patches", exist_ok=True)
os.makedirs("final_transformed_images", exist_ok=True)
os.makedirs("attack_models", exist_ok=True)
os.makedirs("saved_patches", exist_ok=True)  # 如果資料夾不存在，則創建

# 加載 YOLO 模型
weights = 'best.pt'
yolo_model = DetectMultiBackend(weights, device=device, data=None)

# 設置YOLOv3模型為評估模式並禁用梯度追蹤
yolo_model.eval()
for param in yolo_model.parameters():
    param.requires_grad = False

target_class = 'person'  # 攻擊的目標類別
names = yolo_model.names
if target_class not in names.values():
    raise ValueError(f"Target class '{target_class}' not found in model classes: {list(names.values())}")
target_index = next(key for key, value in names.items() if value == target_class)

# 初始化生成器與判別器
generator = Generator(latent_dim).to(device)
discriminator = Discriminator().to(device)

checkpoint_epoch = 220  # 您想要加載的 epoch 數
generator_path = f"gan_models/generator_epoch_{checkpoint_epoch}.pth"
discriminator_path = f"gan_models/discriminator_epoch_{checkpoint_epoch}.pth"
    

if os.path.exists(generator_path) and os.path.exists(discriminator_path):
    try:
        generator.load_state_dict(torch.load(generator_path, map_location=device))
        discriminator.load_state_dict(torch.load(discriminator_path, map_location=device))
        print(f"已成功加載第 {checkpoint_epoch} 個 epoch 的生成器和判別器參數。")
    except Exception as e:
           print(f"加載模型參數時出錯: {e}")
else:
    print(f"模型檔案未找到：{generator_path} 或 {discriminator_path}。將從頭開始訓練。")

# 優化器
optimizer_G = optim.Adam(generator.parameters(), lr=learning_rate, betas=(0.5, 0.999))
optimizer_D = optim.Adam(discriminator.parameters(), lr=learning_rate/10, betas=(0.5, 0.999))

# 損失函數
adversarial_loss = nn.BCELoss()  # GAN 的傳統損失函數
cross_entropy_loss = nn.CrossEntropyLoss()  # 攻擊損失函數

# 加載基底圖像與真實貼片
base_images = [cv2.imread(os.path.join(base_dataset_path, f)) for f in sorted(os.listdir(base_dataset_path)) if f.endswith('.jpg')]
real_patches = [cv2.imread(os.path.join(patch_dataset_path, f)) for f in sorted(os.listdir(patch_dataset_path)) if f.endswith('.png')]

# 載入所有真實貼片
real_patches = []
for f in sorted(os.listdir(patch_dataset_path)):
    if f.endswith('.png'):
        patch = cv2.imread(os.path.join(patch_dataset_path, f))
        if patch is None:
            continue
        patch = patch.transpose(2, 0, 1)  # 調整為 [C, H, W]
        real_patches.append(patch)

# 將貼片轉換為張量
real_patches = torch.tensor(real_patches, dtype=torch.float32).to(device) / 255.0

# 平滑處理參數
real_label_smooth = 0.9  # 將真實標籤從 1.0 降為 0.9
fake_label_smooth = 0.1  # 將生成標籤從 0.0 提高到 0.1

# 儲存 epoch 訓練信息的檔案
epoch_info_path = os.path.join("training_logs", "epoch_training_info.txt")
os.makedirs("training_logs", exist_ok=True)

# 初始化變換參數
epoch_transformation_params = None

# 選擇列表中的第一張圖像進行處理
base_image = base_images[0]  # 獲取單個圖像

# 調整基底圖像尺寸以符合 YOLO 的要求，並獲取縮放比例和填充
base_image_resized, (scale, _), (dw, dh) = letterbox(base_image, new_shape=(416, 416))
base_image_tensor = base_image_resized[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, HWC to CHW
base_image_tensor = (
    torch.from_numpy(base_image_tensor.copy()).float() / 255.0  # 標準化
).unsqueeze(0).to(device)  # 添加批次維度並移動到裝置

# 通過 YOLO 獲取預測目標框
pred = yolo_model(base_image_tensor)
pred = personal_non_max_suppression(pred)

# 提取目標框 (x_min, y_min, x_max, y_max) 並轉換回原始圖像尺寸
target_boxes = []
for det in pred:
    if len(det):
        for *xyxy, conf, cls in det:
            if int(cls) == 2:  # 僅選擇目標類別的框(標誌)
                x_min, y_min, x_max, y_max = map(int, xyxy)
                # 轉換回原始圖像尺寸
                x_min_original = int((x_min - dw) / scale)
                y_min_original = int((y_min - dh) / scale)
                x_max_original = int((x_max - dw) / scale)
                y_max_original = int((y_max - dh) / scale)

                # 確保不超出原始圖像邊界
                x_min_original = max(0, x_min_original)
                y_min_original = max(0, y_min_original)
                x_max_original = min(base_image.shape[1], x_max_original)
                y_max_original = min(base_image.shape[0], y_max_original)

                target_boxes.append((x_min_original, y_min_original, x_max_original, y_max_original))
# 打印目標框，確認結果
print("Target boxes:", target_boxes)

# 設定序列化參數檔案的路徑
params_path = "eot_best_results/params_1_loss2.1772.json"

# 讀取序列化參數
with open(params_path, "r") as f:
    saved_data = json.load(f)

# 提取參數並將其擴展到 epoch_transformation_params
saved_transformation_params = saved_data["params"]  # 假設 JSON 檔案中有 "params" 欄位

# 初始化變換參數
epoch_transformation_params = []
epoch_transformation_params.extend([
    (
        param["gamma"],
        param["angle"],
        np.array(param["dst_points"]),  # 從列表還原為 NumPy 陣列
        param["x"],
        param["y"]
    ) for param in saved_transformation_params
])


with open(epoch_info_path, "w") as log_file:
    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        epoch_progress = tqdm(range(0, len(real_patches), batch_size), desc="Batch Progress", leave=True)
        
        # 初始化 epoch 累積損失
        epoch_loss_D = 0.0
        epoch_loss_G = 0.0
        epoch_attack_loss = 0.0
        num_batches = 0

        for batch_idx in epoch_progress:
            # 分批次載入真實貼片
            real_patch_batch = real_patches[batch_idx:batch_idx + batch_size]
            if real_patch_batch.shape[0] < batch_size:
                continue  # 跳過不完整 batch

            # 使用生成器生成貼片
            z = torch.randn(batch_size, latent_dim, 1, 1, device=device)  # 隨機噪聲
            generated_patches = generator(z)  # 生成貼片

            # 訓練判別器
            optimizer_D.zero_grad()
            real_labels = torch.full((len(real_patch_batch), 1), real_label_smooth, device=device)  # 真實標籤為 0.9
            real_loss = adversarial_loss(discriminator(real_patch_batch), real_labels)
            # 修改此處，使用 .detach() 將生成的假貼片與生成器的計算圖分離
            fake_labels = torch.zeros((batch_size, 1), device=device)  # 生成標籤為 0    
            fake_loss = adversarial_loss(discriminator(generated_patches.detach()), fake_labels)
            loss_D = 0.5 * (real_loss + fake_loss)
            loss_D.backward()  # 反向傳播
            #optimizer_D.step()  # 更新判別器參數

            # 訓練生成器
            optimizer_G.zero_grad()
            fake_labels_for_G = torch.full((batch_size, 1), fake_label_smooth, device=device)  # 生成標籤為 0.1
            gan_loss = adversarial_loss(discriminator(generated_patches), fake_labels_for_G)

            generated_patches = generated_patches.view(batch_size // N, N, 3, 200, 200)  # 每組 N 個貼片，貼片大小應為 200x200


            # 計算攻擊損失
            attack_loss = 0.0  
            for group_idx in range(batch_size // N):
                group_patches = generated_patches[group_idx]
                group_patches_numpy = [
                    (patch.cpu().detach().permute(1, 2, 0).numpy() * 255).astype("uint8")
                    for patch in group_patches
                ]
                for base_idx, base_image in enumerate(base_images):
                    transformed_image, _ = eot(
                        base_image,
                        group_patches_numpy,
                        patch_size=patch_size,
                        gamma_range=(0.8, 1.2),
                        max_rotation=30,
                        max_perspective_shift=0.2,
                        transformation_params=epoch_transformation_params,
                        target_boxes=target_boxes  # 傳入目標方框
                    )
                    letterbox_transformed_image = letterbox(transformed_image)[0]
                    transformed_image_tensor = letterbox_transformed_image[:, :, ::-1].transpose(2, 0, 1)
                    transformed_image_tensor = (
                        torch.from_numpy(transformed_image_tensor.copy()).float() / 255.0
                    ).unsqueeze(0).to(device)
                    pred = yolo_model(transformed_image_tensor)
                    pred = personal_non_max_suppression(pred)

                    for detections in pred:
                        if len(det):
                            for det in detections:
                                # 提取目標類別的概率
                                class_probs = det[6:6+5]  # 假設有 5 個類別
                                p_t = class_probs[target_index].item()
                                    
                                # 防止 log(0) 的情況
                                p_t = max(p_t, 1e-6)
                                    
                                # 計算交叉熵損失
                                loss = -torch.log(torch.tensor(p_t))
                                    
                                # 累加損失
                                attack_loss += loss.item()
                                    
                                #print(f": {cls.item()}, 類別概率: {class_probs.tolist()}, p_t: {p_t}, 單框損失: {loss.item():.4f}")
                                # 保存有效對抗樣本
                                
                            if int(cls) == target_index:  # 使用累計損失作為儲存條件
                                patch_save_path = os.path.join(
                                    "saved_patches", f"epoch_{epoch + 1}_group_{group_idx}_patch_{base_idx + 1}.png"
                                )
                                cv2.imwrite(patch_save_path, transformed_image)
                                print(f"Saved patch: {patch_save_path}")
                      

            attack_loss /= (len(base_images) * (batch_size // N))

            loss_G = gan_loss + alpha * attack_loss
            loss_G.backward()  # 反向傳播
            optimizer_G.step()  # 更新生成器參數

            # 累積 batch 損失
            epoch_loss_D += loss_D.item()
            epoch_loss_G += loss_G.item()
            epoch_attack_loss += attack_loss
            num_batches += 1
            # 打印 epoch 訓練信息
            print(
                f"[Epoch {epoch + 1}/{num_epochs}] Loss_D: {loss_D:.4f}, "
                f"Loss_G: {loss_G:.4f}, Attack Loss: {attack_loss:.4f}"
            )            

        # 計算 epoch 平均損失
        epoch_loss_D /= num_batches
        epoch_loss_G /= num_batches
        epoch_attack_loss /= num_batches

        # 儲存 epoch 訓練信息到檔案
        log_file.write(
            f"Epoch {epoch + 1}/{num_epochs} - Loss_D: {epoch_loss_D:.4f}, "
            f"Loss_G: {epoch_loss_G:.4f}, Attack_Loss: {epoch_attack_loss:.4f}\n"
        )
        log_file.flush()  # 強制刷新緩衝區

        # Epoch 結束後保存生成貼片和基底圖像
        if (epoch + 1) % save_interval == 1:
            # 生成並保存貼片
            z = torch.randn(N, latent_dim, 1, 1, device=device)
            final_generated_patches = generator(z)  # 直接生成貼片
            for i, patch in enumerate(final_generated_patches):
                patch_save_path = os.path.join("final_generated_patches", f"epoch_{epoch + 1}_patch_{i + 1}.png")
                patch = (patch.cpu().detach().permute(1, 2, 0).numpy() * 255).astype("uint8")  # 保留原始範圍
                cv2.imwrite(patch_save_path, patch)

            # 貼片應用到基底圖像並保存
            for base_idx, base_image in enumerate(base_images[:3]):
                transformed_image, transformation_params = eot(
                    base_image, 
                    [(patch.cpu().detach().permute(1, 2, 0).numpy() * 255).astype("uint8") for patch in final_generated_patches],
                    patch_size=patch_size,
                    gamma_range=(0.8, 1.2),
                    max_rotation=30,
                    max_perspective_shift=0.2,
                    transformation_params=epoch_transformation_params,
                    target_boxes=target_boxes  # 傳入目標方框
                )
                
                # 保存生成的圖像
                transformed_save_path = os.path.join("final_transformed_images", f"epoch_{epoch + 1}_base_{base_idx + 1}.png")
                cv2.imwrite(transformed_save_path, transformed_image)
                print(f"Saved transformed image: {transformed_save_path}")

            # 保存模型權重
            torch.save(generator.state_dict(), os.path.join("attack_models", f"generator_epoch_{epoch + 1}.pth"))
            torch.save(discriminator.state_dict(), os.path.join("attack_models", f"discriminator_epoch_{epoch + 1}.pth"))
