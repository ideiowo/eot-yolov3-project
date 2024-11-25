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
from yolov3.utils.general import non_max_suppression
from yolov3.utils.torch_utils import select_device

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
base_dataset = 'base_dataset'  # 圖片資料夾路徑
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
image_files = sorted([f for f in os.listdir(base_dataset) if f.endswith('.jpg')])
print(f"Found {len(image_files)} images in {base_dataset}")

# 設置目標類別
target_class = 'car'
if target_class not in names.values():
    raise ValueError(f"Target class '{target_class}' not found in model classes: {list(names.values())}")
target_index = next(key for key, value in names.items() if value == target_class)  # 目標類別索引

# 開始偵測
for image_file in image_files:
    image_path = os.path.join(base_dataset, image_file)
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"Error: 無法讀取圖片 {image_file}")
        continue

    # 圖像預處理
    img = letterbox(frame, img_size, auto=True)[0]
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB and HWC to CHW
    img = torch.from_numpy(img.copy()).float() / 255.0
    img = img.unsqueeze(0).to(device)

    # 模型推理
    pred = model(img)
    pred = non_max_suppression(pred, conf_threshold, iou_threshold)

    # 計算攻擊損失
    L_f = 0
    print(f"Results for {image_file}:")
    for det in pred:
        if len(det):
            for *xyxy, conf, cls in det:
                bbox = [int(coord) for coord in xyxy]  # 檢測框座標
                label = names[int(cls)]  # 預測類別名稱

                # 使用 cls 作為類別索引，創建 logits
                logits = torch.zeros(len(names), device=device)  # 初始化 logits
                logits[int(cls)] = conf  # 將置信度對應到預測類別

                # 計算交叉熵損失
                L_f += F.cross_entropy(logits.unsqueeze(0), torch.tensor([target_index], device=device))

                # 計算類別概率分佈
                probabilities = torch.softmax(logits, dim=0).tolist()
                print(f"  - BBox: {bbox}, Probabilities: {probabilities}, Class: {label}")


    print(f"  - Total Loss (L_f): {L_f.item():.4f}")

print("所有圖片處理完成。")