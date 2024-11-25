import cv2
import os

# 影片路徑與輸出資料夾
video_path = "video.mp4"
output_dir = "dataset"
os.makedirs(output_dir, exist_ok=True)

# 打開影片
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("Error: 無法打開影片檔案")
    exit()

# 取得影片資訊
fps = cap.get(cv2.CAP_PROP_FPS)  # 幀率
start_time = 36.  # 起始時間 (秒)
end_time = 37.5    # 結束時間 (秒)
start_frame = int(start_time * fps)  # 起始幀
end_frame = int(end_time * fps)      # 結束幀

print(f"影片幀率: {fps:.2f} FPS，範圍幀數: {start_frame} 到 {end_frame}")

# 設置影片讀取幀數
cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

# 提取範圍內的每幀，命名從 1.jpg 開始
frame_index = 1

while cap.get(cv2.CAP_PROP_POS_FRAMES) <= end_frame:
    ret, frame = cap.read()
    if not ret:
        print(f"Error: 無法讀取影片幀，結束提取")
        break

    # 儲存幀為圖像
    frame_name = f"{frame_index}.jpg"
    cv2.imwrite(os.path.join(output_dir, frame_name), frame)
    print(f"已儲存幀至 {frame_name}")
    frame_index += 1

# 釋放資源
cap.release()
print(f"範圍內的幀已提取並儲存到資料夾: {output_dir}")
