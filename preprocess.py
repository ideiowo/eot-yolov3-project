import os
from PIL import Image

# 原始標籤和圖片的路徑
label_dir = r"D:\個人資料夾\碩士文件\碩二上課程\可信任人工智慧\作業2\yolov3\datasets\labels\val"
image_dir = r"D:\個人資料夾\碩士文件\碩二上課程\可信任人工智慧\作業2\yolov3\datasets\images\val"

# 遍歷所有標籤文件
for label_file in os.listdir(label_dir):
    label_path = os.path.join(label_dir, label_file)
    image_path = os.path.join(image_dir, os.path.splitext(label_file)[0] + ".jpg")

    # 如果圖片文件不存在，跳過
    if not os.path.exists(image_path):
        image_path = os.path.join(image_dir, os.path.splitext(label_file)[0] + ".png")
        if not os.path.exists(image_path):
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

    # 寫回文件
    with open(label_path, 'w') as f:
        f.write("\n".join(new_lines))

print("標籤文件已轉換完成！")
