import os
import shutil
import random
from PIL import Image

# 原始資料集位置
base_dir = r"."
image_dir = os.path.join(base_dir, "new_images")
label_dir = os.path.join(base_dir, "new_labels")

# YOLOv3 的目標資料夾結構
output_dir = r"./yolov3/datasets"  # YOLOv3 根目錄中的資料夾
train_image_dir = os.path.join(output_dir, "images", "train")
val_image_dir = os.path.join(output_dir, "images", "val")
train_label_dir = os.path.join(output_dir, "labels", "train")
val_label_dir = os.path.join(output_dir, "labels", "val")

# 建立目錄結構
os.makedirs(train_image_dir, exist_ok=True)
os.makedirs(val_image_dir, exist_ok=True)
os.makedirs(train_label_dir, exist_ok=True)
os.makedirs(val_label_dir, exist_ok=True)

# 獲取所有圖片檔案
image_files = [f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png'))]
random.seed(42)
random.shuffle(image_files)

# 分割數據集
split_idx = int(0.8 * len(image_files))  # 80% 作為訓練集，20% 作為驗證集

# 將資料分別複製到對應的資料夾中
for idx, image_file in enumerate(image_files):
    label_file = os.path.splitext(image_file)[0] + ".txt"  # 對應的標籤檔案
    if idx < split_idx:
        shutil.copy2(os.path.join(image_dir, image_file), train_image_dir)
        shutil.copy2(os.path.join(label_dir, label_file), train_label_dir)
    else:
        shutil.copy2(os.path.join(image_dir, image_file), val_image_dir)
        shutil.copy2(os.path.join(label_dir, label_file), val_label_dir)

print("資料已成功複製到 YOLOv3 資料夾結構中！")

# YOLOv3 資料夾結構
base_dir = r"./yolov3/datasets"
label_dirs = [os.path.join(base_dir, "labels", "train"), os.path.join(base_dir, "labels", "val")]
image_dirs = [os.path.join(base_dir, "images", "train"), os.path.join(base_dir, "images", "val")]

# 處理標籤文件
for label_dir, image_dir in zip(label_dirs, image_dirs):
    # 遍歷標籤資料夾中的所有標籤文件
    for label_file in os.listdir(label_dir):
        label_path = os.path.join(label_dir, label_file)
        image_path_jpg = os.path.join(image_dir, os.path.splitext(label_file)[0] + ".jpg")
        image_path_png = os.path.join(image_dir, os.path.splitext(label_file)[0] + ".png")

        # 嘗試找到對應的圖片
        if os.path.exists(image_path_jpg):
            image_path = image_path_jpg
        elif os.path.exists(image_path_png):
            image_path = image_path_png
        else:
            print(f"Image not found for {label_file}, skipping...")
            continue

        # 獲取圖片尺寸
        with Image.open(image_path) as img:
            img_width, img_height = img.size

        # 新的標籤內容
        new_lines = []

        # 讀取並轉換標籤
        with open(label_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                # 讀取 class_id 和邊界框 (x_min, y_min, x_max, y_max)
                parts = line.strip().split()
                if len(parts) != 5:
                    print(f"Invalid label format in {label_file}: {line.strip()}")
                    continue

                class_id, x_min, y_min, x_max, y_max = map(float, parts)

                # 計算 YOLO 格式的 (x_center, y_center, width, height)
                x_center = ((x_min + x_max) / 2) / img_width
                y_center = ((y_min + y_max) / 2) / img_height
                width = (x_max - x_min) / img_width
                height = (y_max - y_min) / img_height

                # 檢查座標是否有效
                if 0 <= x_center <= 1 and 0 <= y_center <= 1 and 0 <= width <= 1 and 0 <= height <= 1:
                    new_lines.append(f"{int(class_id)} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
                else:
                    print(f"Out-of-bounds label in {label_file}: {line.strip()}")

        # 將轉換後的標籤寫回文件
        with open(label_path, 'w') as f:
            f.write("\n".join(new_lines))

print("所有標籤文件已成功轉換為 YOLO 格式！")