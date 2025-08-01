import os
import time
import threading
import cv2
import numpy as np
from datetime import datetime
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.cameras.opencv.camera_opencv import OpenCVCamera
from lerobot.cameras.configs import ColorMode, Cv2Rotation
from PIL import Image
import base64
import requests
import torch
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
import sys
import supervision as sv

# 添加Grounded-SAM-2到路径
grounded_sam2_path = os.path.join(os.path.dirname(__file__), '../../Grounded-SAM-2')
if grounded_sam2_path not in sys.path:
    sys.path.append(grounded_sam2_path)

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

# ========== 配置 ==========
CAMERA_ID = "/dev/video4"
WIDTH, HEIGHT = 1280, 720
FPS = 25

BOX_THRESHOLD = 0.2
TEXT_THRESHOLD = 0.2
TEXT_PROMPT = "HDMI plug. HDMI port. HDMI connector. HDMI"
GROUNDING_DINO_MODEL_ID = "IDEA-Research/grounding-dino-base"

# SAM2配置
SAM2_CHECKPOINT = os.path.join(os.path.dirname(__file__), '../../Grounded-SAM-2/checkpoints/sam2.1_hiera_large.pt')
SAM2_CONFIG = "configs/sam2.1/sam2.1_hiera_l.yaml"

API_KEY = os.environ.get("GEMINI_API_KEY") or "AIzaSyDijBhdh2Hj-D6C-iaTatB_5zvX9C5YRi0"

# 加载参考图片并编码为base64
def load_reference_images():
    ref_dir = os.path.join(os.path.dirname(__file__), 'reference_images')
    ref_images = {}
    for ref_type in ['yes1', 'no1', 'unknown1']:
        ref_path = os.path.join(ref_dir, f'{ref_type}.jpg')
        if os.path.exists(ref_path):
            with open(ref_path, 'rb') as f:
                ref_images[ref_type] = base64.b64encode(f.read()).decode()
        else:
            print(f"Warning: Reference image {ref_path} not found")
            ref_images[ref_type] = ""
    return ref_images

# 加载参考图片
reference_images = load_reference_images()

VLM_PROMPT = f"""Please determine if the nearest HDMI plug/Connector is inserted into a HDMI port. Only answer Yes or No or Unknown.

Reference examples:
- YES: HDMI plug is physically connected and inserted into HDMI port, with the plug's metal connector fully or partially inside the port's opening
- NO: HDMI plug and HDMI port are clearly visible but separate, with visible gap between them, or plug is held near but not inserted into port  
- UNKNOWN: Cannot clearly determine the insertion status due to poor visibility, occlusion, or ambiguous positioning

Look for physical contact and insertion depth between the HDMI plug and port.

Reference images:
YES example: <image>{reference_images.get('yes1', '')}</image>
NO example: <image>{reference_images.get('no1', '')}</image>
UNKNOWN example: <image>{reference_images.get('unknown1', '')}</image>"""

#VLM_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-pro:generateContent?key={API_KEY}"
VLM_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-lite-preview-06-17:generateContent?key={API_KEY}"

OUTPUT_DIR = "outputs/realtime_detection"
os.makedirs(OUTPUT_DIR, exist_ok=True)

grounding_processor = AutoProcessor.from_pretrained(GROUNDING_DINO_MODEL_ID)
grounding_model = AutoModelForZeroShotObjectDetection.from_pretrained(GROUNDING_DINO_MODEL_ID).to("cuda:0" if torch.cuda.is_available() else "cpu")

# 初始化SAM2模型
print("Loading SAM2 model...")
device = "cuda:0" if torch.cuda.is_available() else "cpu"
sam2_model = build_sam2(SAM2_CONFIG, SAM2_CHECKPOINT, device=device)
sam2_predictor = SAM2ImagePredictor(sam2_model)
print("SAM2 model loaded successfully")

config = OpenCVCameraConfig(
    index_or_path=CAMERA_ID,
    fps=FPS,
    width=WIDTH,
    height=HEIGHT,
    color_mode=ColorMode.RGB,
    rotation=Cv2Rotation.NO_ROTATION
)
camera = OpenCVCamera(config)
camera.connect()

latest_detection_img = None
latest_detection_vis = None
latest_detection_boxes = None
latest_detection_labels = None
latest_detection_masks = None
lock = threading.Lock()

# ========== HDMI检测+supervision可视化 ==========
def run_grounded_sam2_detection(frame, save_path=None):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    image_pil = Image.fromarray(frame)
    inputs = grounding_processor(images=image_pil, text=TEXT_PROMPT, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = grounding_model(**inputs)
    results = grounding_processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        box_threshold=BOX_THRESHOLD,
        text_threshold=TEXT_THRESHOLD,
        target_sizes=[image_pil.size[::-1]]
    )[0]
    boxes = results["boxes"].cpu().numpy()
    labels = results["labels"]
    confidences = results["scores"].cpu().numpy()
    # 过滤面积
    img_area = frame.shape[0] * frame.shape[1]
    area_thresh = 0.25 * img_area
    filtered_boxes, filtered_labels, filtered_confidences = [], [], []
    for bbox, label, conf in zip(boxes, labels, confidences):
        x1, y1, x2, y2 = map(int, bbox)
        area = (x2 - x1) * (y2 - y1)
        if area < area_thresh:
            filtered_boxes.append([x1, y1, x2, y2])
            filtered_labels.append(label)
            filtered_confidences.append(conf)
    if not filtered_boxes:
        vis = frame.copy()
        if save_path:
            # 转换为BGR格式保存
            vis_bgr = cv2.cvtColor(vis, cv2.COLOR_RGB2BGR)
            cv2.imwrite(save_path, vis_bgr)
        return vis, [], [], None
    
    # 使用真正的SAM2生成分割掩码
    print(f"Running SAM2 segmentation for {len(filtered_boxes)} detected objects...")
    sam2_predictor.set_image(frame)
    
    # 为每个检测框生成分割掩码
    masks = []
    for bbox in filtered_boxes:
        x1, y1, x2, y2 = bbox
        # 使用边界框作为提示
        mask, score, logits = sam2_predictor.predict(
            point_coords=None,
            point_labels=None,
            box=np.array([x1, y1, x2, y2]),
            multimask_output=False,
        )
        masks.append(mask)
    
    # 将掩码堆叠成(n, H, W)格式
    if masks:
        masks = np.stack(masks, axis=0).astype(bool)
        # 确保维度正确：如果masks是4D，需要squeeze
        if masks.ndim == 4:
            masks = masks.squeeze(1)
    else:
        masks = np.zeros((len(filtered_boxes), HEIGHT, WIDTH), dtype=bool)
    
    # supervision可视化
    detection_labels = [f"{label} {conf:.2f}" for label, conf in zip(filtered_labels, filtered_confidences)]
    
    # 创建不带mask的detections用于边界框和标签
    detections_sv = sv.Detections(
        xyxy=np.array(filtered_boxes),
        confidence=np.array(filtered_confidences),
        class_id=np.arange(len(filtered_boxes))
    )
    
    box_annotator = sv.BoxAnnotator()
    label_annotator = sv.LabelAnnotator()
    mask_annotator = sv.MaskAnnotator()
    annotated_image = frame.copy()
    
    # 添加边界框
    annotated_image = box_annotator.annotate(
        scene=annotated_image,
        detections=detections_sv
    )
    
    # 添加标签
    annotated_image = label_annotator.annotate(
        scene=annotated_image,
        detections=detections_sv,
        labels=detection_labels
    )
    
    # 添加分割掩码（使用带mask的detections）
    detections_with_masks = sv.Detections(
        xyxy=np.array(filtered_boxes),
        confidence=np.array(filtered_confidences),
        class_id=np.arange(len(filtered_boxes)),
        mask=masks
    )
    
    annotated_image = mask_annotator.annotate(
        scene=annotated_image,
        detections=detections_with_masks
    )
    if save_path:
        # 转换为BGR格式保存
        annotated_bgr = cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(save_path, annotated_bgr)
    return annotated_image, filtered_boxes, filtered_labels, masks

# ========== VLM判断函数 ==========
def run_vlm_judge(img):
    try:
        # 确保图像是RGB格式发送给API
        _, img_encoded = cv2.imencode('.jpg', img)
        img_b64 = base64.b64encode(img_encoded.tobytes()).decode()
        
        # 构建包含参考图片的parts
        parts = [
            {"text": "Please determine if the nearest HDMI plug/Connector is inserted into a HDMI port. Only answer Yes or No or Unknown.\n\nReference examples:\n- YES: HDMI plug is physically connected and inserted into HDMI port, with the plug's metal connector fully or partially inside the port's opening\n- NO: HDMI plug and HDMI port are clearly visible but separate, with visible gap between them, or plug is held near but not inserted into port\n- UNKNOWN: Cannot clearly determine the insertion status due to poor visibility, occlusion, or ambiguous positioning\n\nLook for physical contact and insertion depth between the HDMI plug and port.\n\nReference images:"}
        ]
        
        # 添加参考图片
        for ref_type, ref_b64 in reference_images.items():
            if ref_b64:
                parts.append({"inline_data": {"mime_type": "image/jpeg", "data": ref_b64}})
                parts.append({"text": f"\n{ref_type.upper()} example:"})
        
        # 添加当前输入图片
        parts.append({"text": "\n\nCurrent image to analyze:"})
        parts.append({"inline_data": {"mime_type": "image/jpeg", "data": img_b64}})
        
        data = {
            "contents": [
                {
                    "role": "user",
                    "parts": parts
                }
            ]
        }
        
        headers = {"Content-Type": "application/json"}
        response = requests.post(VLM_URL, headers=headers, json=data, timeout=15)
        # Is this timeout in seconds? or in miliseconds?
        result_text = "Error"
        if response.status_code == 200:
            result = response.json()
            try:
                text = result['candidates'][0]['content']['parts'][0]['text']
                result_text = text.strip()
            except Exception as e:
                result_text = f"ParseError: {e}"
        else:
            result_text = f"HTTP {response.status_code}"
    except requests.exceptions.RequestException as e:
        result_text = f"NetworkError: {type(e).__name__}"
    except Exception as e:
        result_text = f"UnexpectedError: {type(e).__name__}"
    
    # 可视化加文字
    vis = img.copy()
    cv2.putText(vis, f"VLM: {result_text}", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,0,0), 4)
    return result_text, vis

# ========== 检测线程 ==========
def detection_loop():
    global latest_detection_img, latest_detection_vis, latest_detection_boxes, latest_detection_labels, latest_detection_masks
    while True:
        frame = camera.async_read(timeout_ms=500)
        vis, boxes, labels, masks = run_grounded_sam2_detection(frame)
        with lock:
            latest_detection_img = frame.copy()
            latest_detection_vis = vis.copy()
            latest_detection_boxes = boxes
            latest_detection_labels = labels
            latest_detection_masks = masks
        time.sleep(0.5)  # 2Hz

# ========== VLM线程 ==========
def vlm_loop():
    iter_idx = 0
    while True:
        try:
            with lock:
                img = latest_detection_img.copy() if latest_detection_img is not None else None
                vis = latest_detection_vis.copy() if latest_detection_vis is not None else None
                boxes = latest_detection_boxes.copy() if latest_detection_boxes is not None else None
                labels = latest_detection_labels.copy() if latest_detection_labels is not None else None
            if img is not None and vis is not None and boxes is not None and labels is not None:
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                
                # 创建VLM输入图像：原始图像 + plug/port标注框
                vlm_input_img = img.copy()
                for bbox, label in zip(boxes, labels):
                    x1, y1, x2, y2 = bbox
                    label_lower = label.lower()
                    if 'plug' in label_lower or 'hdmi' in label_lower or 'port' in label_lower or 'connector' in label_lower:
                        cv2.rectangle(vlm_input_img, (x1, y1), (x2, y2), (0, 255, 0), 4)  # 绿色

                # VLM判断
                result, vlm_img = run_vlm_judge(vlm_input_img)
                
                # 拼接：左图是完整检测可视化，右图是VLM输入+VLM回答
                concat_img = np.concatenate([vis, vlm_img], axis=1)
                
                # 实时显示（需要转换为BGR格式）
                concat_bgr = cv2.cvtColor(concat_img, cv2.COLOR_RGB2BGR)
                cv2.imshow('Real-time HDMI Detection & VLM', concat_bgr)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):  # 按q退出
                    print("用户按q退出...")
                    return
                
                # 保存图像（转换为BGR格式）
                concat_path = os.path.join(OUTPUT_DIR, f"concat_{ts}_{iter_idx}.jpg")
                cv2.imwrite(concat_path, concat_bgr)
                print(f"[{ts}] VLM判断: {result} | 已保存: {concat_path}")
            else:
                print("[VLM] 等待检测结果...")
        except Exception as e:
            print(f"[VLM] 处理帧时出错: {e}")
            # 继续运行，不终止程序
        iter_idx += 1
        time.sleep(3)

threading.Thread(target=detection_loop, daemon=True).start()
vlm_thread = threading.Thread(target=vlm_loop, daemon=True)
vlm_thread.start()

print("实时检测已启动，按 Ctrl+C 或按 'q' 键退出...")
try:
    while vlm_thread.is_alive():
        time.sleep(1)
except KeyboardInterrupt:
    print("收到Ctrl+C，退出实时检测.")
finally:
    cv2.destroyAllWindows()
    camera.disconnect() 