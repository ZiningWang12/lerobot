from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.cameras.opencv.camera_opencv import OpenCVCamera
from lerobot.cameras.configs import ColorMode, Cv2Rotation
import cv2
import os
import time

# 可测试的分辨率列表（720p及以下）
resolutions = [
    (1280, 720), #10Hz
    (800, 600), #15Hz
    (640, 480), #25Hz
]

# 相机设备路径
camera_id = "/dev/video4"

# 输出目录
output_dir = "outputs/test_opencv_frames"
os.makedirs(output_dir, exist_ok=True)

frames_per_resolution = 20

for width, height in resolutions:
    print(f"\n=== 测试分辨率: {width}x{height} ===")
    config = OpenCVCameraConfig(
        index_or_path=camera_id,
        fps=25,
        width=width,
        height=height,
        color_mode=ColorMode.RGB,
        rotation=Cv2Rotation.NO_ROTATION
    )
    camera = OpenCVCamera(config)
    camera.connect()
    timestamps = []
    try:
        for i in range(frames_per_resolution):
            t0 = time.time()
            frame = camera.async_read(timeout_ms=500)
            t1 = time.time()
            timestamps.append(t1)
            print(f"分辨率 {width}x{height} - 帧 {i} shape: {frame.shape}  采集耗时: {t1-t0:.3f}s")
            # 保存每一帧到本地
            bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            out_path = os.path.join(output_dir, f"frame_{width}x{height}_{i:02d}.jpg")
            cv2.imwrite(out_path, bgr)
        # 保存时间戳，便于后续分析帧率
        ts_path = os.path.join(output_dir, f"timestamps_{width}x{height}.txt")
        with open(ts_path, 'w') as f:
            for ts in timestamps:
                f.write(f"{ts}\n")
        print(f"已保存所有帧和时间戳到: {output_dir}")
    finally:
        camera.disconnect()

# ========== 绘制帧间延迟曲线 ==========
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
for width, height in resolutions:
    ts_path = os.path.join(output_dir, f"timestamps_{width}x{height}.txt")
    if not os.path.exists(ts_path):
        continue
    with open(ts_path, 'r') as f:
        timestamps = [float(line.strip()) for line in f if line.strip()]
    if len(timestamps) < 2:
        continue
    delays = [t2 - t1 for t1, t2 in zip(timestamps[:-1], timestamps[1:])]
    plt.plot(delays, marker='o', label=f"{width}x{height}")

plt.title("OpenCV Camera Frame Delay (Interval)")
plt.xlabel("frame index")
plt.ylabel("frame delay (s)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()