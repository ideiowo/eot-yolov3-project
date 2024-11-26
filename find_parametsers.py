import numpy as np
import cv2
import random
import os
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

# 將參數轉換為可序列化格式的函數
def make_serializable(params):
    serializable_params = []
    for gamma, angle, dst_points, x, y in params:
        serializable_params.append({
            "gamma": float(gamma),
            "angle": float(angle),
            "dst_points": np.array(dst_points).tolist(),  # 將 NumPy 陣列轉為列表
            "x": int(x),
            "y": int(y)
        })
    return serializable_params

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

def eot(base_image, patches, patch_size=(60, 60), gamma_range=(0.8, 1.2), max_rotation=30, max_perspective_shift=0.2, transformation_params=None, target_boxes=None):
    """
    執行期望轉換技術 (EOT)，對多個貼片進行隨機位置擺放及變換，並將其應用於基底圖像。

    參數:
    - base_image (np.ndarray): 基底圖像（即背景圖像）。
    - patches (list of np.ndarray): 貼片列表（每個貼片為影像）。
    - patch_size (tuple): 貼片的目標大小 (寬, 高)。
    - gamma_range (tuple): 伽瑪校正值的範圍。
    - max_rotation (int): 最大旋轉角度（以度為單位）。
    - max_perspective_shift (float): 透視變換的最大偏移比例（相對於貼片寬高的比例）。
    - transformation_params (list): 傳入固定的變換參數 (選填)。
    - target_boxes (list): 目標方框範圍 [(x_min, y_min, x_max, y_max), ...]。

    返回:
    - transformed_image (np.ndarray): 含變換貼片的基底圖像。
    - transformation_params (list): 使用的變換參數列表。
    """
    H, W, _ = base_image.shape

    # 初始化變換參數列表
    if transformation_params is None:
        transformation_params = []

    transformed_image = base_image.copy()  # 複製基底圖像
    for idx, patch in enumerate(patches):
        # 1. 縮放貼片
        patch = cv2.resize(patch, patch_size)

        # 使用固定變換參數或生成隨機參數
        if idx < len(transformation_params):
            gamma, angle, dst_points, x, y = transformation_params[idx]
            # 初始化 src_points，因為 dst_points 已固定
            width, height = patch_size
            src_points = np.float32([[0, 0], [width, 0], [0, height], [width, height]])
        else:
            # 生成新的隨機參數
            gamma = random.uniform(*gamma_range)
            angle = random.uniform(-max_rotation, max_rotation)
            width, height = patch_size
            shift_x = int(max_perspective_shift * width)
            shift_y = int(max_perspective_shift * height)
            src_points = np.float32([[0, 0], [width, 0], [0, height], [width, height]])
            dst_points = np.float32([
                [random.randint(-shift_x, shift_x), random.randint(-shift_y, shift_y)],
                [width + random.randint(-shift_x, shift_x), random.randint(-shift_y, shift_y)],
                [random.randint(-shift_x, shift_x), height + random.randint(-shift_y, shift_y)],
                [width + random.randint(-shift_x, shift_x), height + random.randint(-shift_y, shift_y)]
            ])
            
            # 在目標方框內隨機生成貼片位置
            if target_boxes:
                x_min, y_min, x_max, y_max = random.choice(target_boxes)
                x = random.randint(x_min, max(x_min, x_max - patch_size[0]))
                y = random.randint(y_min, max(y_min, y_max - patch_size[1]))
            else:
                x = random.randint(0, W - patch_size[0])
                y = random.randint(0, H - patch_size[1])

            # 添加到變換參數列表
            transformation_params.append((gamma, angle, dst_points, x, y))

        # 2. 伽瑪校正
        patch = np.clip(np.power(patch / 255.0, gamma) * 255.0, 0, 255).astype(np.uint8)

        # 3. 旋轉貼片
        center = (patch_size[0] // 2, patch_size[1] // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        patch = cv2.warpAffine(patch, rotation_matrix, patch_size, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))

        # 4. 透視變換
        perspective_matrix = cv2.getPerspectiveTransform(src_points, dst_points)
        patch = cv2.warpPerspective(patch, perspective_matrix, patch_size, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))

        # 5. 隨機擺放
        # 去除白色背景
        threshold = 100
        mask = (patch < threshold).astype(np.uint8)

        # 平滑遮罩邊界
        blurred_mask = cv2.GaussianBlur(mask.astype(np.float32), (5, 5), 0)
        blurred_mask = np.clip(blurred_mask, 0, 1)

        # 如果 mask 是單通道，擴展為三通道
        if mask.shape[-1] != 3:
            blurred_mask = np.stack([blurred_mask] * 3, axis=-1)

        # 遮罩混合貼片與基底
        transformed_image[y:y + patch_size[1], x:x + patch_size[0]] = (
            transformed_image[y:y + patch_size[1], x:x + patch_size[0]] * (1 - blurred_mask) +
            patch * blurred_mask
        )

    return transformed_image, transformation_params



import json

if __name__ == "__main__":
    # 加載基底圖像
    base_image = cv2.imread("./base_dataset/33.jpg")  # 基底圖像
    patch_folder = "./final_generated_patches"
    patches = [cv2.imread(os.path.join(patch_folder, f"epoch_21_patch_{i}.png")) for i in range(1, 7)]

    device = torch.device('cuda')
    model = DetectMultiBackend('best.pt', device=device, data=None)
    stride = model.stride
    names = model.names
    conf_threshold = 0.25
    iou_threshold = 0.45
    img_size = 416
    target_index = 3

    num_iterations = 5000  # 執行 EOT 的次數
    top_k = 5  # 篩選的最佳結果數量
    results = []  # 用於存儲 (loss, params, transformed_image) 的列表

    print(f"已加載 {len(patches)} 個貼片。")

    for i in range(num_iterations):
        # 執行 EOT
        result_image, transformation_params = eot(
            base_image,
            patches,
            patch_size=(140, 140),
            gamma_range=(0.8, 1.2),
            max_rotation=30,
            max_perspective_shift=0.2,
            target_boxes=[(83, 436, 575, 1200)]
        )

        # 圖像預處理
        img = letterbox(result_image, img_size, auto=True)[0]
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB and HWC to CHW
        img = torch.from_numpy(img.copy()).float() / 255.0
        img = img.unsqueeze(0).to(device)

        # 模型推理
        pred = model(img)
        pred = non_max_suppression(pred, conf_threshold, iou_threshold)
        
        # 計算攻擊損失
        L_f = torch.tensor(0.0, device=device)
        has_target = False  # 檢查是否有目標類別
        for det in pred:
            if len(det):
                for *xyxy, conf, cls in det:
                    label = names[int(cls)]  # 預測類別名稱
                    logits = torch.zeros(len(names), device=device)  # 初始化 logits
                    logits[int(cls)] = conf  # 將置信度對應到預測類別

                    # 計算交叉熵損失
                    L_f += F.cross_entropy(logits.unsqueeze(0), torch.tensor([target_index], device=device))
                    # 判定是否為目標類別
                    if int(cls) == target_index:
                        has_target = True
                        print(f"Iteration {i + 1}/{num_iterations}: Predicted Target Class ({target_index}).")

        total_loss = L_f.item()
        print(f"pred:{pred}")
        print(f"Iteration {i + 1}/{num_iterations}: Total Loss = {total_loss:.4f}")
        
        # 儲存結果（包括特殊情況）
        if total_loss > 0 or has_target:
            results.append((total_loss, transformation_params, result_image.copy()))

    # 排序並篩選最小損失的結果
    results.sort(key=lambda x: x[0])  # 按 Total Loss 排序
    best_results = results[:top_k]  # 選擇前 k 項

    # 儲存最優結果
    output_dir = "eot_best_results"
    os.makedirs(output_dir, exist_ok=True)
    # 儲存最佳結果
    for idx, (loss, params, image) in enumerate(best_results):
        if loss > 0:
            # 將參數轉換為可序列化格式
            serializable_params = make_serializable(params)

            # 儲存變換參數
            params_path = os.path.join(output_dir, f"params_{idx + 1}_loss{loss:.4f}.json")
            with open(params_path, 'w') as f:
                json.dump({"loss": loss, "params": serializable_params}, f, indent=4)

            # 儲存結果圖像
            image_path = os.path.join(output_dir, f"result_image_{idx + 1}.png")
            cv2.imwrite(image_path, image)
            print(f"Saved Result {idx + 1}: Loss = {loss:.4f}, Params saved to {params_path}, Image saved to {image_path}")
