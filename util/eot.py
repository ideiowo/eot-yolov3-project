import numpy as np
import cv2
import random

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
        threshold = 200
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


if __name__ == "__main__":
    # 加載基底圖像
    base_image = cv2.imread("./base_dataset/33.jpg")

    # 加載多個貼片影像
    #patch1 = cv2.imread("./star/star/1.png")
    #patch2 = cv2.imread("./star/star/100.png")
    patch1 = cv2.imread("./final_generated_patches/epoch_2_patch_2.png")
    patch2 = cv2.imread("./final_generated_patches/epoch_2_patch_2.png")
    patches = [patch1, patch2]

    # 執行 EOT，解包返回的結果
    result_image, _ = eot(
        base_image,
        patches,
        patch_size=(100, 100),
        gamma_range=(0.8, 1.2),
        max_rotation=30,
        max_perspective_shift=0.2
    )

    # 儲存結果
    cv2.imwrite("result_image.jpg", result_image)


