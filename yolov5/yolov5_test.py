import torch
import os
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
from utils.metrics import bbox_iou

# Load YOLOv5 model with custom weights
model = torch.load('/home/ubuntu/CS230_final_project/CS230-final-project/yolov5_training/exp2/weights/best.pt')
# Set your test folder paths
images_folder = '/home/ubuntu/CS230_final_project/CS230-final-project/drone_dataset/test/images'  # Replace with your path to test images
labels_folder = '/home/ubuntu/CS230_final_project/CS230-final-project/drone_dataset/test/labels'  # Replace with your path to label files
output_folder = '/home/ubuntu/CS230_final_project/CS230-final-project/yolov5_test'  # Folder where you want to save images with predicted boxes
os.makedirs(output_folder, exist_ok=True)

# Function to parse YOLO label files
def parse_label_file(label_path):
    with open(label_path, 'r') as f:
        lines = f.readlines()
        labels = []
        for line in lines:
            data = line.strip().split()
            cls, x, y, w, h = map(float, data)
            labels.append([cls, x, y, w, h])
    return labels

# Calculate IoU between predicted and ground-truth boxes
def calculate_iou(pred_boxes, true_boxes):
    ious = []
    for true_box in true_boxes:
        true_box_xyxy = xywh2xyxy(true_box[1:])
        for pred_box in pred_boxes:
            pred_box_xyxy = pred_box[:4]
            iou = bbox_iou(torch.tensor(pred_box_xyxy), torch.tensor(true_box_xyxy), x1y1x2y2=True)
            ious.append(iou.item())
    return ious

# Convert [x, y, w, h] format to [x1, y1, x2, y2]
def xywh2xyxy(box):
    x, y, w, h = box
    x1 = (x - w / 2) * img_width
    y1 = (y - h / 2) * img_height
    x2 = (x + w / 2) * img_width
    y2 = (y + h / 2) * img_height
    return [x1, y1, x2, y2]

# Iterate through images in the test folder
for image_file in tqdm(os.listdir(images_folder)):
    image_path = os.path.join(images_folder, image_file)
    label_path = os.path.join(labels_folder, Path(image_file).stem + '.txt')

    # Load image
    img = cv2.imread(image_path)
    img_height, img_width = img.shape[:2]

    # Run YOLOv5 inference
    results = model(img)
    pred_boxes = results.xyxy[0].cpu().numpy()  # Predicted boxes in [x1, y1, x2, y2, confidence, class]

    # Load ground-truth labels
    true_boxes = parse_label_file(label_path)

    # Calculate IoU
    ious = calculate_iou(pred_boxes, true_boxes)
    if ious:
        avg_iou = np.mean(ious)
        print(f"Image: {image_file}, Average IoU: {avg_iou:.2f}")
    else:
        print(f"Image: {image_file}, No ground truth boxes found.")

    # Draw predicted boxes on the image
    for box in pred_boxes:
        x1, y1, x2, y2, conf, cls = box
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.putText(img, f'Class: {int(cls)}, Conf: {conf:.2f}', (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # Save the output image
    output_path = os.path.join(output_folder, image_file)
    cv2.imwrite(output_path, img)

print("Testing completed.")