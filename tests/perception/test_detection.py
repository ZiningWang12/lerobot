# My private Gemini API key: AIzaSyDijBhdh2Hj-D6C-iaTatB_5zvX9C5YRi0
# Test detection of HDMI plug and HDMI port using various perception models

# First test Gemini-2.5 Flash lite (ID:gemini-2.5-flash-lite-preview-06-17)

import os
import requests
import base64
import json
import cv2
import matplotlib.pyplot as plt

# Read API KEY from environment variable
API_KEY = "AIzaSyDijBhdh2Hj-D6C-iaTatB_5zvX9C5YRi0"

# Image path
IMG_PATH = os.path.join(os.path.dirname(__file__), 'frame_1280x720_00.jpg')

# Read image and encode as base64
with open(IMG_PATH, 'rb') as f:
    img_b64 = base64.b64encode(f.read()).decode()

# Gemini 2.5 API endpoint
url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-lite-preview-06-17:generateContent?key={API_KEY}"

headers = {"Content-Type": "application/json"}
data = {
    "contents": [
        {
            "role": "user",
            "parts": [
                {"text": "Detect HDMI plug and HDMI port in the image. Return the center coordinates as a JSON list of objects with keys 'center' (x, y) and 'label'. Do not return box2d."},
                {"inline_data": {"mime_type": "image/jpeg", "data": img_b64}}
            ]
        }
    ]
}

print("Calling Gemini 2.5 API for object detection (center only)...")
response = requests.post(url, headers=headers, json=data)
if response.status_code == 200:
    print("Gemini 2.5 detection result:")
    result = response.json()
    print(result)
    # Parse detection result
    try:
        text = result['candidates'][0]['content']['parts'][0]['text']
        if text.startswith('```json'):
            text = text.split('```json')[-1]
        if text.startswith('\n'):
            text = text[1:]
        if text.endswith('```'):
            text = text[:-3]
        detections = json.loads(text)
    except Exception as e:
        print(f"Failed to parse detection result: {e}")
        detections = []
else:
    print(f"Request failed, status code: {response.status_code}")
    print(response.text)
    detections = []

# Try multiple coordinate interpretations
def draw_and_save(img, coords, labels, mode_desc, out_name):
    img_draw = img.copy()
    for (x, y), label in zip(coords, labels):
        if 0 <= int(x) < img.shape[1] and 0 <= int(y) < img.shape[0]:
            cv2.circle(img_draw, (int(x), int(y)), 8, (0, 255, 0), -1)
            cv2.putText(img_draw, label, (int(x)+10, int(y)-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            print(f"Warning: ({x},{y}) out of bounds for mode {mode_desc}")
    out_path = os.path.join(os.path.dirname(__file__), out_name)
    cv2.imwrite(out_path, img_draw)
    print(f"Saved {out_path} using mode: {mode_desc}")
    return img_draw

if detections:
    img = cv2.imread(IMG_PATH)
    h, w = img.shape[:2]
    centers = [det['center'] for det in detections]
    labels = [det['label'] for det in detections]

    # 1. (x, y) = (center[0], center[1])
    coords1 = [(c[0], c[1]) for c in centers]
    draw_and_save(img, coords1, labels, '(x, y) = (center[0], center[1])', 'detected_result_mode1.jpg')

    print("Tried all common coordinate interpretations. Please check the output images to see which one matches the real object positions.")
else:
    print("No detection results to visualize.")

# Second test Grounded-SAM2 with Local Grounding DINO
print("\n" + "="*50)
print("Starting Grounded-SAM-2 (Local) test...")
print("="*50)

try:
    import sys
    import torch
    import numpy as np
    from PIL import Image
    import supervision as sv
    
    # Add Grounded-SAM-2 to path
    grounded_sam2_path = os.path.join(os.path.dirname(__file__), '../../Grounded-SAM-2')
    if grounded_sam2_path not in sys.path:
        sys.path.append(grounded_sam2_path)
    
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
    from utils.supervision_utils import CUSTOM_COLOR_MAP
    
    # Configuration for HuggingFace Grounding DINO
    TEXT_PROMPT = "HDMI plug. HDMI port. HDMI connector. HDMI"  # HuggingFace format
    BOX_THRESHOLD = 0.2  # Lower threshold
    TEXT_THRESHOLD = 0.2  # Lower threshold
    
    # SAM2 Configuration  
    SAM2_CHECKPOINT = os.path.join(grounded_sam2_path, "checkpoints/sam2.1_hiera_large.pt")
    SAM2_CONFIG = "configs/sam2.1/sam2.1_hiera_l.yaml"
    
    # HuggingFace Grounding DINO Configuration
    GROUNDING_DINO_MODEL_ID = "IDEA-Research/grounding-dino-base"  # Use larger model
    
    # Set device
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Initialize HuggingFace Grounding DINO
    print("Loading HuggingFace Grounding DINO model...")
    grounding_processor = AutoProcessor.from_pretrained(GROUNDING_DINO_MODEL_ID)
    grounding_model = AutoModelForZeroShotObjectDetection.from_pretrained(GROUNDING_DINO_MODEL_ID).to(device)
    print("HuggingFace Grounding DINO model loaded successfully")
    
    # Initialize SAM2
    print("Loading SAM2 model...")
    sam2_model = build_sam2(SAM2_CONFIG, SAM2_CHECKPOINT, device=device)
    sam2_predictor = SAM2ImagePredictor(sam2_model)
    print("Loaded local SAM2 model successfully")
    
    # Load and process image
    print("Processing image...")
    image_pil = Image.open(IMG_PATH).convert('RGB')
    image_source = np.array(image_pil)
    
    # Run HuggingFace Grounding DINO for object detection
    print("Running HuggingFace Grounding DINO object detection...")
    
    inputs = grounding_processor(images=image_pil, text=TEXT_PROMPT, return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = grounding_model(**inputs)
    
    # Post-process results
    results = grounding_processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        box_threshold=BOX_THRESHOLD,
        text_threshold=TEXT_THRESHOLD,
        target_sizes=[image_pil.size[::-1]]  # (height, width)
    )[0]
    
    boxes = results["boxes"].cpu().numpy()
    labels = results["labels"]
    confidences = results["scores"].cpu().numpy()
    
    print(f"HuggingFace Grounding DINO detected {len(boxes)} objects")
    
    if len(boxes) > 0:
        # Extract detection results
        bboxes = boxes.tolist()
        confidences = confidences.tolist()
        
        for i, (bbox, label, confidence) in enumerate(zip(bboxes, labels, confidences)):
            print(f"Detected: {label} (conf: {confidence:.3f}) at bbox: {bbox}")
        
        # Set image for SAM2 predictor
        print("Running SAM2 segmentation...")
        sam2_predictor.set_image(image_source)
        
        # Enable mixed precision for better performance
        torch.autocast(device_type=device, dtype=torch.bfloat16).__enter__()
        if torch.cuda.is_available() and torch.cuda.get_device_properties(0).major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        
        # Run SAM2 with box prompts (more accurate than point prompts)
        masks, scores, logits = sam2_predictor.predict(
            point_coords=None,
            point_labels=None,
            box=np.array(bboxes),
            multimask_output=False,
        )
        
        # Convert the shape to (n, H, W) if needed
        if masks.ndim == 4:
            masks = masks.squeeze(1)
        
        print(f"SAM2 generated {len(masks)} masks with scores: {scores}")
        
        # Create labels for visualization
        detection_labels = [
            f"{label} {confidence:.2f}"
            for label, confidence in zip(labels, confidences)
        ]
        
        # Convert to supervision format
        detections_sv = sv.Detections(
            xyxy=np.array(bboxes),
            confidence=np.array(confidences),
            class_id=np.arange(len(bboxes))
        )
        
        # Visualize results using supervision
        print("Creating visualization...")
        太
        # Create annotators
        box_annotator = sv.BoxAnnotator()
        label_annotator = sv.LabelAnnotator()
        mask_annotator = sv.MaskAnnotator()
        
        # Annotate image
        annotated_image = image_source.copy()
        
        # Add bounding boxes
        annotated_image = box_annotator.annotate(
            scene=annotated_image,
            detections=detections_sv
        )
        
        # Add labels
        annotated_image = label_annotator.annotate(
            scene=annotated_image,
            detections=detections_sv,
            labels=detection_labels
        )
        
        # Add masks
        detections_with_masks = sv.Detections(
            xyxy=np.array(bboxes),
            confidence=np.array(confidences),
            class_id=np.arange(len(bboxes)),
            mask=masks.astype(bool)
        )
        
        annotated_image = mask_annotator.annotate(
            scene=annotated_image,
            detections=detections_with_masks
        )
        
        # Save result
        output_path = os.path.join(os.path.dirname(__file__), 'detected_result_grounded_sam2_hf.jpg')
        cv2.imwrite(output_path, annotated_image)
        print(f"Saved HuggingFace Grounding DINO + SAM2 result to: {output_path}")
        
        # Extract centers in same format as Gemini for comparison
        hf_grounding_centers = []
        for bbox, label, conf in zip(bboxes, labels, confidences):
            x1, y1, x2, y2 = bbox
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            hf_grounding_centers.append({
                'center': [float(center_x), float(center_y)],
                'label': f"{label} ({conf:.2f})"
            })
        
        print("HuggingFace Grounding DINO + SAM2 center coordinates:")
        print(json.dumps(hf_grounding_centers, indent=2))
        
        # Save centers using the same visualization as Gemini test
        if hf_grounding_centers:
            centers_hf = [det['center'] for det in hf_grounding_centers]
            labels_hf = [det['label'] for det in hf_grounding_centers]
            
            # Use the same coordinate interpretations
            img_orig = cv2.imread(IMG_PATH)
            coords_hf = [(c[0], c[1]) for c in centers_hf]
            draw_and_save(img_orig, coords_hf, labels_hf, 'HuggingFace Grounding DINO + SAM2 centers', 'detected_result_hf_centers.jpg')
    else:
        print("No objects detected by HuggingFace Grounding DINO")

except Exception as e:
    print(f"Error in Grounded-SAM-2 (HuggingFace) test: {e}")
    import traceback
    traceback.print_exc()

# Third test: VLM + GroundedSAM2 联合推理
print("\n" + "="*50)
print("Starting VLM + GroundedSAM2 joint test...")
print("="*50)

try:
    import cv2
    import base64
    import requests
    # 1. 过滤 plug/port 检测结果
    img = cv2.imread(IMG_PATH)
    h, w = img.shape[:2]
    img_area = h * w
    area_thresh = 0.25 * img_area
    
    # 只保留 plug/port 且面积小于阈值的框
    plug_boxes = []
    port_boxes = []
    for bbox, label in zip(bboxes, labels):
        x1, y1, x2, y2 = map(int, bbox)
        area = (x2 - x1) * (y2 - y1)
        label_lower = label.lower()
        if area < area_thresh:
            if 'plug' in label_lower:
                plug_boxes.append((x1, y1, x2, y2))
            elif 'port' in label_lower:
                port_boxes.append((x1, y1, x2, y2))
    
    # 2. 标注图片
    img_vlm = img.copy()
    for (x1, y1, x2, y2) in plug_boxes:
        cv2.rectangle(img_vlm, (x1, y1), (x2, y2), (0,255,0), 4)  # 绿色
    for (x1, y1, x2, y2) in port_boxes:
        cv2.rectangle(img_vlm, (x1, y1), (x2, y2), (0,0,255), 4)  # 红色
    vlm_img_path = os.path.join(os.path.dirname(__file__), 'grounded_sam2_vlm_input.jpg')
    cv2.imwrite(vlm_img_path, img_vlm)
    print(f"Saved VLM input image: {vlm_img_path}")
    
    # 3. VLM推理（Gemini API）
    with open(vlm_img_path, 'rb') as f:
        img_b64 = base64.b64encode(f.read()).decode()
    prompt = "这张图片中，绿色框为可能的HDMI plug/HDMI port/HMDI connector。请判断HDMI plug是否已经插入HDMI port？只回答Yes或No。"
    data = {
        "contents": [
            {
                "role": "user",
                "parts": [
                    {"text": prompt},
                    {"inline_data": {"mime_type": "image/jpeg", "data": img_b64}}
                ]
            }
        ]
    }
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-lite-preview-06-17:generateContent?key={API_KEY}"
    headers = {"Content-Type": "application/json"}
    response = requests.post(url, headers=headers, json=data)
    if response.status_code == 200:
        result = response.json()
        print("VLM (Gemini) response:")
        print(result)
        try:
            text = result['candidates'][0]['content']['parts'][0]['text']
            print("VLM判断结果:", text.strip())
        except Exception as e:
            print(f"Failed to parse VLM result: {e}")
    else:
        print(f"VLM API request failed, status code: {response.status_code}")
        print(response.text)
except Exception as e:
    print(f"Error in VLM + GroundedSAM2 test: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*50)
print("Testing completed!")
print("="*50)