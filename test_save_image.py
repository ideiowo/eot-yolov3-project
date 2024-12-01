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
from yolov3.utils.general import non_max_suppression  # 確保這是正確的路徑和函數名稱

# 自定義 letterbox 函數
def letterbox(im, new_shape=(416, 416), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True):
    shape = im.shape[:2]  # 當前圖像尺寸 [高度, 寬度]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # 計算縮放比例
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:
        r = min(r, 1.0)

    # 計算新的未填充尺寸
    ratio = r, r  # 寬度比例, 高度比例
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    dw /= 2  # 左右填充
    dh /= 2  # 上下填充

    if shape[::-1] != new_unpad:
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return im, ratio, (dw, dh)

# 修正 scale_coords 函數，添加更多檢查
def scale_coords(img1_shape, coords, img0_shape, ratio_pad=None):
    """
    Scales coords (xyxy) from img1_shape to img0_shape.
    """
    if ratio_pad is None:
        # Compute the scaling ratio and padding
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    # Adjust coordinates with scaling and padding
    coords[:, [0, 2]] -= pad[0]  # x padding
    coords[:, [1, 3]] -= pad[1]  # y padding
    coords[:, :4] /= gain
    clip_coords(coords, img0_shape)  # Clip coordinates to image size
    return coords

# 添加錯誤處理和檢查
def clip_coords(boxes, shape):
    """
    Clip bounding xyxy bounding boxes to image shape (height, width).
    """
    boxes[:, 0].clamp_(0, shape[1])  # x1
    boxes[:, 1].clamp_(0, shape[0])  # y1
    boxes[:, 2].clamp_(0, shape[1])  # x2
    boxes[:, 3].clamp_(0, shape[0])  # y2

# 配置參數
weights = 'best.pt'  # 訓練後的權重檔案
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 使用 GPU 或 CPU
conf_threshold = 0.25  # 信心閾值
iou_threshold = 0.45    # IoU 閾值
img_size = 416          # 圖像尺寸

# 加載模型
model = DetectMultiBackend(weights, device=device, data=None)
stride = model.stride
names = model.names
print(f"模型已加載，步幅為 {stride}，共有 {len(names)} 個類別：{names}")

# 定義目標類別並獲取其索引
target_class = 'person'  # 攻擊的目標類別
if target_class not in names.values():
    raise ValueError(f"目標類別 '{target_class}' 未在模型類別中找到: {list(names.values())}")
target_index = next(key for key, value in names.items() if value == target_class)

# 設定處理的圖片資料夾
transformed_images_dir = "./final_transformed_images"
# 設定儲存帶框圖片的資料夾
output_dir = "./attacked_images"

# 確保儲存目錄存在
os.makedirs(transformed_images_dir, exist_ok=True)
os.makedirs(output_dir, exist_ok=True)

# 常見的圖片副檔名
valid_image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}

# 遍歷資料夾中的每張圖片
for image_name in os.listdir(transformed_images_dir):
    # 確認副檔名是否為圖片格式
    if os.path.splitext(image_name)[1].lower() in valid_image_extensions:
        image_path = os.path.join(transformed_images_dir, image_name)
        transformed_image = cv2.imread(image_path)
        if transformed_image is None:
            print(f"無法讀取圖片: {image_path}")
            continue
    else:
        #print(f"跳過非圖片檔案: {image_name}")
        continue

    # 圖像預處理
    img, ratio, (dw, dh) = letterbox(transformed_image, img_size, auto=True)
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR 轉 RGB 並調整為 CHW
    img = torch.from_numpy(img.copy()).float() / 255.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)  # 增加 batch 維度
    img = img.to(device)

    # 模型推理
    with torch.no_grad():
        pred = model(img)

    # 非極大值抑制
    preds = non_max_suppression(pred, conf_threshold, iou_threshold, agnostic=False)

    # 確保縮放處理的輸入參數正確
    for det in preds:  # 遍歷每張圖片的檢測結果
        if det is not None and len(det):
            # 檢查原始圖像形狀是否正確傳遞
            if transformed_image.shape[:2] == (0, 0):
                print("錯誤：原始圖像尺寸無效")
                continue
            # 修正邊界框縮放
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], transformed_image.shape).round()

            # 繪製邊界框時的顏色和線條粗細設定
            for *xyxy, conf, cls in reversed(det):
                cls = int(cls)  # 確保類別是整數
                label = f"{names[cls]} {conf:.2f}"

                # 根據類別設定框的顏色和粗細
                if cls == 0:  # 類別 0（person），紅色框
                    color = (0, 0, 255)  # BGR 紅色
                elif cls == 2:  # 類別 2（sign），藍色框
                    color = (255, 0, 0)  # BGR 藍色
                else:  # 其他類別，使用綠色框
                    color = (0, 255, 0)  # BGR 綠色

                thickness = 3        # 加粗框
                fontScale = 2.0      # 大字體
                fontThickness = 2    # 字體加粗

                # 繪製矩形框
                cv2.rectangle(transformed_image, 
                            (int(xyxy[0]), int(xyxy[1])), 
                            (int(xyxy[2]), int(xyxy[3])), 
                            color, thickness)

                # 繪製標籤背景
                (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, fontScale, fontThickness)
                cv2.rectangle(transformed_image, 
                            (int(xyxy[0]), int(xyxy[1]) - text_height - baseline), 
                            (int(xyxy[0]) + text_width, int(xyxy[1])), 
                            color, -1)  # 填充背景

                # 繪製標籤文字
                cv2.putText(transformed_image, label, 
                            (int(xyxy[0]), int(xyxy[1]) - baseline), 
                            cv2.FONT_HERSHEY_SIMPLEX, fontScale, (0, 0, 0), fontThickness)
                
    # 儲存帶有邊界框的圖片
    output_path = os.path.join(output_dir, image_name)
    cv2.imwrite(output_path, transformed_image)
    print(f"已儲存帶框圖片: {output_path}")

print("所有圖片處理完成！")
