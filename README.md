# YOLOv3 微調專案

本專案展示如何在自訂資料集上進行 YOLOv3 模型的微調，並使用已訓練的模型進行即時物件檢測。

---

## 專案功能

- 支援 YOLOv3 模型的微調，適用於自訂資料集
- 包含資料預處理與模型訓練腳本
- 提供即時物件檢測腳本，可使用攝像頭進行測試

---

## 專案結構

```
.
├── yolov3/                 # YOLOv3 模型與工具檔案
├── new_images/             # 自訂資料集圖片
├── new_labels/             # 自訂資料集標籤
├── gan/
│   ├── generator.py        # 生成器
│   └── discriminator.py    # 判別器
├── utils/
│   ├── eot.py              # EOT 相關函數
│   ├── datasets.py         # 自訂資料集類
│   └── attack.py           # 攻擊相關函數
├── preprocess.py           # YOLOv3資料預處理腳本
├── test.py                 # 即時物件檢測腳本
├── best.pt                 # 微調後的 YOLOv3 模型權重
└── README.md               # 專案說明文件
```

---

## 快速開始

### 1. 環境請自行準備

---

### 2. 微調 YOLOv3 模型

#### 資料集準備

1. **將圖片放置於指定資料夾**
   - 將所有自訂圖片放置於 `new_images/` 資料夾。
   
2. **將標籤檔案放置於指定資料夾**
   - 將所有對應的標籤檔案放置於 `new_labels/` 資料夾，標籤需符合 YOLO 格式。

3. **執行資料預處理（如有需要）**
   - 若需調整資料格式或分割資料集，可運行以下腳本：
     ```bash
     python preprocess.py
     ```

#### 開始訓練

1. **運行訓練腳本**
   ```bash
   cd yolov3
   python train.py --data <custom_data.yaml> --cfg <odels/yolov3-tiny-custom.yaml> --epochs 50 --batch-size 16 --img-size 416    
   ```
   - `<custom_data.yaml>`：自訂資料集的配置檔案。
   - `<yolov3-tiny-custom.cfg>`：自訂模型的配置檔案。

2. **檢查輸出**
   訓練完成後，最佳權重檔案將儲存為 `best.pt`。

---

### 3. 測試模型

1. **執行即時物件檢測腳本**
   ```bash
   python test.py
   ```

2. **條件與功能**
   - 確保 `best.pt` 位於專案根目錄下。
   - 腳本會啟動攝像頭進行即時檢測，並在畫面上顯示檢測框與物件標籤。

3. **結束檢測**
   - 按下 `ctrl + c` 可退出即時檢測。

---

## 注意事項

1. **資料集格式**
   - 圖片需為 `.jpg` 或 `.png` 格式。
   - 標籤文件需為 YOLO 標籤格式：每行包含 `<class> <x_center> <y_center> <width> <height>`，且值需歸一化至 [0, 1]。

2. **測試腳本**
   - 預設從攝像頭讀取影像，若需測試單張圖片或影片，可修改 `test.py` 的 `source` 參數。

---

## 支援類別

專案已設定以下檢測類別，並可根據需要進行調整：

1. Person（人）
2. Text（文字）
3. Sign（標誌）
4. Car（汽車）
5. Bicycle（自行車）

---

## 授權

本專案基於 [YOLOv3](https://github.com/ultralytics/yolov3) 並遵循 AGPL-3.0 授權協議。

