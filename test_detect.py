import sys
import os

# 確保 yolov3 的路徑在搜索路徑中
yolo3_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'yolov3'))
if yolo3_path not in sys.path:
    sys.path.append(yolo3_path)

import torch
import cv2
import numpy as np
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

# 自定義 scale_coords 函數
def scale_coords(img1_shape, coords, img0_shape, ratio_pad=None):
    if ratio_pad is None:
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    coords[:, [0, 2]] -= pad[0]
    coords[:, [1, 3]] -= pad[1]
    coords[:, :4] /= gain
    coords[:, 0].clip(0, img0_shape[1])
    coords[:, 1].clip(0, img0_shape[0])
    coords[:, 2].clip(0, img0_shape[1])
    coords[:, 3].clip(0, img0_shape[0])
    return coords

# 配置參數
weights = 'best.pt'
device = torch.device('cpu') 
conf_threshold = 0.3
iou_threshold = 0.45
img_size = 416
source = 2

# 加載模型
model = DetectMultiBackend(weights, device=device, data=None)
stride = model.stride
names = model.names
print(f"Model loaded with stride {stride}, and {len(names)} classes: {names}")

# 打開攝像頭
cap = cv2.VideoCapture(source)
if not cap.isOpened():
    print("Error: Unable to open camera.")
    exit()

print("Press 'ctrl + c' to quit.")
while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame, exiting...")
        break

    # 圖像預處理
    img = letterbox(frame, img_size, auto=True)[0]
    img = img[:, :, ::-1].transpose(2, 0, 1)
    img = torch.from_numpy(img.copy()).float() / 255.0
    img = img.unsqueeze(0).to(device)

    # 模型推理
    pred = model(img)
    pred = non_max_suppression(pred, conf_threshold, iou_threshold)

    # 繪製檢測框
    for det in pred:
        if len(det):
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], frame.shape).round()
            for *xyxy, conf, cls in det:
                # 根據類別設定框的顏色和粗細
                if cls == 0:  # 類別 0（person），紅色框
                    color = (0, 0, 255)  # BGR 紅色
                elif cls == 2:  # 類別 2（sign），藍色框
                    color = (255, 0, 0)  # BGR 藍色
                else:  # 其他類別，使用綠色框
                    color = (0, 255, 0)  # BGR 綠色

                thickness = 3  # 加粗框
                fontScale = 2.0  # 大字體
                fontThickness = 2  # 字體加粗

                label = f"{names[int(cls)]} {conf:.2f}"
                cv2.rectangle(frame, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), color, thickness)
                cv2.putText(frame, label, (int(xyxy[0]), int(xyxy[1] - 10)), cv2.FONT_HERSHEY_SIMPLEX, fontScale, color, fontThickness)

    # 顯示結果
    cv2.imshow("YOLOv3 Detection", frame)

    # 按下 'q' 鍵退出
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

