"""
YOLO Visualization Module
Visualize YOLO predictions with bounding boxes compared to ground truth.
"""

from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt
import os
import numpy as np
from pathlib import Path


def draw_box(img, box, color, label, thickness=3):
    """Draw bounding box on image with label."""
    x1, y1, x2, y2 = map(int, box)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
    cv2.putText(img, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return img


def yolo_to_xyxy(yolo_box, img_width, img_height):
    """Convert YOLO format (x_center, y_center, width, height) to (x1, y1, x2, y2)."""
    x_center, y_center, width, height = yolo_box
    x1 = (x_center - width/2) * img_width
    y1 = (y_center - height/2) * img_height
    x2 = (x_center + width/2) * img_width
    y2 = (y_center + height/2) * img_height
    return [x1, y1, x2, y2]


def visualize_yolo_predictions(
    yolo_model_path,
    yolo_dataset_path,
    img_size=640,
    conf_threshold=0.5,
    num_images=1
):
    """
    Visualize YOLO predictions with ground truth comparison.
    
    Args:
        yolo_model_path: Path to trained YOLO model weights
        yolo_dataset_path: Path to YOLO dataset root (containing test/images and test/labels)
        img_size: Input image size for YOLO
        conf_threshold: Confidence threshold for predictions
        num_images: Number of test images to visualize
    """
    # Load model
    model = YOLO(str(yolo_model_path))
    
    # Setup paths
    yolo_dataset_path = Path(yolo_dataset_path)
    test_images_dir = yolo_dataset_path / "test" / "images"
    test_labels_dir = yolo_dataset_path / "test" / "labels"
    
    # Get test images
    test_images = [str(test_images_dir / img) for img in os.listdir(str(test_images_dir))[:num_images]]
    
    # Process each image
    for img_path in test_images:
        # Load image
        img = cv2.imread(img_path)
        if img is None:
            print(f"⚠️  Could not load image: {img_path}")
            continue
            
        img_height, img_width = img.shape[:2]

        # Run YOLO prediction
        results = model.predict(img_path, imgsz=img_size, conf=conf_threshold, verbose=False)

        # Draw predictions (BLUE)
        for box in results[0].boxes:
            xyxy = box.xyxy[0].cpu().numpy()
            cls = int(box.cls)
            conf = box.conf.item()
            label = f"Pred: Class {cls} ({conf:.2f})"
            img = draw_box(img, xyxy, (255, 0, 0), label)

        # Load and draw ground truth (GREEN)
        label_file = os.path.join(
            str(test_labels_dir), 
            os.path.basename(img_path).replace('.png', '.txt').replace('.jpg', '.txt')
        )
        
        if os.path.exists(label_file):
            with open(label_file, 'r') as f:
                for line in f.readlines():
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        cls_gt = int(parts[0])
                        yolo_box = [float(x) for x in parts[1:5]]
                        xyxy_gt = yolo_to_xyxy(yolo_box, img_width, img_height)
                        label_gt = f"GT: Class {cls_gt}"
                        img = draw_box(img, xyxy_gt, (0, 255, 0), label_gt, thickness=2)

        # Display
        plt.figure(figsize=(15, 10))
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.title(f"YOLO Predictions - {os.path.basename(img_path)}\nGreen = Ground Truth | Blue = Prediction")
        plt.axis('off')
        plt.show()

        print(f"\n{os.path.basename(img_path)}:")
        print(f"  Predictions: {len(results[0].boxes)} boxes detected")
