import os
import sys
import cv2
import torch
import torch.nn.functional as F
# 確保 yolov3 的路徑在搜索路徑中
yolo3_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'yolov3'))
if yolo3_path not in sys.path:
    sys.path.append(yolo3_path)

from yolov3.models.common import DetectMultiBackend
from optimize_parametsers import personal_non_max_suppression


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
weights = 'best.pt'  # 訓練後的權重檔案
base_dataset = 'saved_patches'  # 圖片資料夾路徑
device = torch.device('cuda')  # 使用 GPU（改為 'cuda:0' 或 'cpu'）
conf_threshold = 0.25
iou_threshold = 0.45
img_size = 416

# 加載模型
model = DetectMultiBackend(weights, device=device, data=None)
stride = model.stride
names = model.names
print(f"Model loaded with stride {stride}, and {len(names)} classes: {names}")

# 處理 base_dataset 資料夾
image_files = sorted([f for f in os.listdir(base_dataset) if f.endswith('.png')])
print(f"Found {len(image_files)} images in {base_dataset}")

# 定義目標類別並獲取其索引
target_class = 'person'  # 攻擊的目標類別
if target_class not in names.values():
    raise ValueError(f"Target class '{target_class}' not found in model classes: {list(names.values())}")
target_index = next(key for key, value in names.items() if value == target_class)

# 設定處理的圖片資料夾
transformed_images_dir = "./eot_best_results"

# 確保儲存目錄存在
os.makedirs("saved_patches", exist_ok=True)
# 常見的圖片副檔名
valid_image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}


# 遍歷資料夾中的每張圖片
for image_name in os.listdir(transformed_images_dir):
    # 確認副檔名是否為圖片格式
    if os.path.splitext(image_name)[1].lower() in valid_image_extensions:
        image_path = os.path.join(transformed_images_dir, image_name)
        transformed_image = cv2.imread(image_path)
    else:
        #print(f"Skipped non-image file: {image_name}")
        continue

    # 圖像預處理
    img = letterbox(transformed_image, img_size, auto=True)[0]
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB and HWC to CHW
    img = torch.from_numpy(img.copy()).float() / 255.0
    img = img.unsqueeze(0).to(device)

    # 模型推理
    pred = model(img)
         
    nms_pred = personal_non_max_suppression(pred, conf_threshold, iou_threshold)

    attack_loss = 0.0  # 初始化攻擊損失
    save_flag = False  # 初始化保存標誌

    for detections in nms_pred:
        if detections is not None and len(detections):
            print(len(detections))
            for det in detections:
                # 提取目標類別的概率
                class_probs = det[6:6+5]  # 假設有 5 個類別
                p_t = class_probs[target_index].item()
                
                cls = int(det[5].item())
                # 防止 log(0) 的情況
                p_t = max(p_t, 1e-6)
                    
                # 計算交叉熵損失
                loss = -torch.log(torch.tensor(p_t))
                # 累加損失
                attack_loss += loss.item()

                    
                print(f"類別: {cls}, 類別概率: {class_probs.tolist()}")
        else:
            print("沒有物件:",nms_pred)

    print(f"圖片 {image_name} 的總攻擊損失: {attack_loss:.4f}\n")