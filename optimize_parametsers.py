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
from yolov3.utils.general import xywh2xyxy, box_iou
from yolov3.utils.torch_utils import select_device
import torchvision
import time

def personal_non_max_suppression(
    prediction,
    conf_thres=0.25,
    iou_thres=0.45,
    classes=None,
    agnostic=False,
    multi_label=False,
    labels=(),
    max_det=300,
    nm=0,  # number of masks
):
    """
    Non-Maximum Suppression (NMS) on inference results to reject overlapping detections.

    Returns:
         list of detections, on (n, 6 + nc + nm) tensor per image [xyxy, conf, cls, class_probs]
         If no detections, includes class probabilities.
    """
    # Checks
    assert 0 <= conf_thres <= 1, f"Invalid Confidence threshold {conf_thres}, valid values are between 0.0 and 1.0"
    assert 0 <= iou_thres <= 1, f"Invalid IoU {iou_thres}, valid values are between 0.0 and 1.0"
    if isinstance(prediction, (list, tuple)):  # YOLOv3 model in validation model, output = (inference_out, loss_out)
        prediction = prediction[0]  # select only inference output

    device = prediction.device
    mps = "mps" in device.type  # Apple MPS
    if mps:  # MPS not fully supported yet, convert tensors to CPU before NMS
        prediction = prediction.cpu()
    bs = prediction.shape[0]  # batch size
    nc = prediction.shape[2] - nm - 5  # number of classes
    xc = prediction[..., 4] > conf_thres  # candidates

    # Settings
    max_wh = 7680  # (pixels) maximum box width and height
    max_nms = 30000  # maximum number of boxes into torchvision.ops.nms()
    time_limit = 0.5 + 0.05 * bs  # seconds to quit after
    redundant = True  # require redundant detections
    multi_label &= nc > 1  # multiple labels per box (adds 0.5ms/img)
    merge = False  # use merge-NMS

    t = time.time()
    mi = 5 + nc  # mask start index
    output = [torch.zeros((0, 6 + nc + nm), device=prediction.device)] * bs  # Initialize output tensors

    for xi, x in enumerate(prediction):  # image index, image inference
        # Apply constraints
        x = x[xc[xi]]  # confidence

        # Cat apriori labels if autolabelling
        if labels and len(labels[xi]):
            lb = labels[xi]
            v = torch.zeros((len(lb), nc + nm + 5), device=x.device)
            v[:, :4] = lb[:, 1:5]  # box
            v[:, 4] = 1.0  # conf
            v[range(len(lb)), lb[:, 0].long() + 5] = 1.0  # cls
            x = torch.cat((x, v), 0)

        # If none remain process next image
        if not x.shape[0]:
            # Compute class probabilities from the raw predictions
            raw_x = prediction[xi]
            raw_conf = raw_x[:, 4]
            raw_cls = raw_x[:, 5:mi]
            # Option 1: Take the maximum class probability across all predictions
            class_probs, _ = raw_cls.max(0)
            # Option 2: Take the average class probability
            # class_probs = raw_cls.mean(0)

            # Normalize class probabilities
            class_probs = class_probs / class_probs.sum()

            # Create a dummy detection
            # Format: [x1, y1, x2, y2, conf, cls, class_probs..., masks...]
            # Set box coordinates and conf to zero, cls to -1 to indicate no detection
            dummy_box = torch.zeros(4, device=device)
            dummy_conf = torch.zeros(1, device=device)
            dummy_cls = -1 * torch.ones(1, device=device)
            dummy_mask = torch.zeros(nm, device=device) if nm > 0 else torch.tensor([], device=device)
            dummy_detection = torch.cat([dummy_box, dummy_conf, dummy_cls, class_probs, dummy_mask], dim=0).unsqueeze(0)

            output[xi] = dummy_detection
            continue

        # Compute conf
        x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf

        # Box/Mask
        box = xywh2xyxy(x[:, :4])  # center_x, center_y, width, height) to (x1, y1, x2, y2)
        mask = x[:, mi:]  # zero columns if no masks

        # Detections matrix nx(6 + nc) [xyxy, conf, cls, class_probs]
        if multi_label:
            i, j = (x[:, 5:mi] > conf_thres).nonzero(as_tuple=False).T
            all_class_probs = x[i, 5:mi]
            x = torch.cat((box[i], x[i, 4:5], j[:, None].float(), all_class_probs, mask[i]), 1)
        else:  # best class only
            conf, j = x[:, 5:mi].max(1, keepdim=True)
            all_class_probs = x[:, 5:mi]
            x = torch.cat((box, conf, j.float(), all_class_probs, mask), 1)[conf.view(-1) > conf_thres]

        # Filter by class
        if classes is not None:
            x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]

        # Apply finite constraint (optional, enable if needed)
        # if not torch.isfinite(x).all():
        #     x = x[torch.isfinite(x).all(1)]

        # Check shape
        n = x.shape[0]  # number of boxes
        if not n:  # no boxes after class filtering
            # Compute class probabilities from the raw predictions
            raw_x = prediction[xi]
            raw_conf = raw_x[:, 4]
            raw_cls = raw_x[:, 5:mi]
            # Option 1: Take the maximum class probability across all predictions
            class_probs, _ = raw_cls.max(0)
            # Option 2: Take the average class probability
            # class_probs = raw_cls.mean(0)

            # Normalize class probabilities
            class_probs = class_probs / class_probs.sum()

            # Create a dummy detection
            dummy_box = torch.zeros(4, device=device)
            dummy_conf = torch.zeros(1, device=device)
            dummy_cls = -1 * torch.ones(1, device=device)
            dummy_mask = torch.zeros(nm, device=device) if nm > 0 else torch.tensor([], device=device)
            dummy_detection = torch.cat([dummy_box, dummy_conf, dummy_cls, class_probs, dummy_mask], dim=0).unsqueeze(0)

            output[xi] = dummy_detection
            continue

        x = x[x[:, 4].argsort(descending=True)[:max_nms]]  # sort by confidence and remove excess boxes

        # Batched NMS
        c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
        boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
        i = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS
        i = i[:max_det]  # limit detections

        if merge and (1 < n < 3e3):  # Merge NMS (boxes merged using weighted mean)
            # update boxes as boxes(i,4) = weights(i,n) * boxes(n,4)
            iou = box_iou(boxes[i], boxes) > iou_thres  # iou matrix
            weights = iou * scores[None]  # box weights
            x[i, :4] = torch.mm(weights, x[:, :4]).float() / weights.sum(1, keepdim=True)  # merged boxes
            if redundant:
                i = i[iou.sum(1) > 1]  # require redundancy

        # 保存結果並包含所有類別概率
        output[xi] = x[i]
        if mps:
            output[xi] = output[xi].to(device)
        if (time.time() - t) > time_limit:
            print(f"WARNING ⚠️ NMS time limit {time_limit:.3f}s exceeded")
            break  # time limit exceeded

    return output

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

def extract_class_probs(pred):
    """
    從非極大值抑制 (NMS) 後的輸出中提取類別概率
    """
    class_probs_list = []  # 存放每張圖片的類別概率
    for det in pred:  # 遍歷每張圖片的檢測結果
        if det is not None and len(det) > 0:  # 確保有檢測結果
            det = det.to(device)  # 確保檢測結果在正確的設備上
            # det 的形狀為 (num_detections, 6) -> [x1, y1, x2, y2, conf, class_id]
            # 提取類別概率分佈
            for *xyxy, conf, cls in det:
                class_prob = torch.zeros(len(names), device=device)  # 初始化類別概率
                class_prob[int(cls)] = conf  # 將置信度分配給預測類別
                class_probs_list.append(class_prob)
    return class_probs_list  # 返回類別概率列表



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
             
        nms_pred = personal_non_max_suppression(pred, conf_threshold, iou_threshold)
        
        for idx, detections in enumerate(nms_pred):
            if detections is not None and len(detections):
                attack_loss = 0.0  # 初始化攻擊損失
                for det in detections:
                    xyxy = det[:4]  # 框的座標
                    conf = det[4]   # 置信度
                    cls = det[5]    # 類別編號
                    class_probs = det[6:6+5]  # 5 個類別的概率 (假設有 5 個類別)
                    
                    # 提取目標類別的概率
                    p_t = class_probs[target_index].item()
                    
                    # 防止 log(0) 的情況
                    p_t = max(p_t, 1e-6)
                    
                    # 計算交叉熵損失
                    loss = -torch.log(torch.tensor(p_t))
                    
                    # 累加損失
                    attack_loss += loss.item()
                    
                    print(f"類別: {cls.item()}, 類別概率: {class_probs.tolist()}, p_t: {p_t}, 單框損失: {loss.item():.4f}")
                
                print(f"圖片 {idx} 的總攻擊損失: {attack_loss:.4f}\n")
                # 儲存損失、參數、圖像到 results
                results.append((attack_loss, transformation_params, result_image.copy()))


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