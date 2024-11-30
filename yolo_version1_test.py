
from yolo_version1_model import CustomYOLODataset, YOLO
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import numpy as np
from torch.utils.data import DataLoader
import os
import math
from tqdm import tqdm
from PIL import Image, ImageDraw
from sklearn.metrics import precision_score, recall_score, f1_score


def calculate_iou(box1, box2):
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    box1_x1, box1_y1 = x1 - w1 / 2, y1 - h1 / 2
    box1_x2, box1_y2 = x1 + w1 / 2, y1 + h1 / 2
    box2_x1, box2_y1 = x2 - w2 / 2, y2 - h2 / 2
    box2_x2, box2_y2 = x2 + w2 / 2, y2 + h2 / 2

    inter_x1 = max(box1_x1, box2_x1)
    inter_y1 = max(box1_y1, box2_y1)
    inter_x2 = min(box1_x2, box2_x2)
    inter_y2 = min(box1_y2, box2_y2)

    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
    box1_area = (box1_x2 - box1_x1) * (box1_y2 - box1_y1)
    box2_area = (box2_x2 - box2_x1) * (box2_y2 - box2_y1)

    union_area = box1_area + box2_area - inter_area

    return inter_area / union_area if union_area != 0 else 0

test_folder = '/home/ubuntu/CS230_final_project/CS230-final-project/drone_dataset/test'
output_folder = '/home/ubuntu/CS230_final_project/CS230-final-project/yolo_v1_output_images'
model_path = '/home/ubuntu/CS230_final_project/CS230-final-project/yolo_v1_weights/yolo_model_grid5_yololoss_lr0.00001_batch128.pth' #change path to the corresponding 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

grid_size = 5 #change grid size to 

transform = transforms.Compose([
    transforms.Resize((640, 640)),
    transforms.ToTensor(),
])


yolo = YOLO(grid_size=grid_size)
yolo.load_state_dict(torch.load(model_path))
yolo.to(device)
                       


print('Starting Testing')
test_dataset = CustomYOLODataset(image_folder=test_folder+'/images', label_folder=test_folder+'/labels', grid_size=grid_size, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
os.makedirs(output_folder, exist_ok=True)
yolo.eval()

y_true_boxes = []
y_pred_boxes = []
iou_scores = []
y_true_classes = []
y_pred_classes = []
loc_errors = []
size_errors = []

with torch.no_grad():
    for data in test_loader:
        inputs, labels, img_path = data
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = yolo(inputs)
        predictions = torch.sigmoid(outputs).view(grid_size, grid_size, 6).cpu().numpy()
        # outputs[...,0:4] = torch.log(outputs[...,0:4])
        # outputs[...,4:] = torch.sigmoid(outputs[...,4:])
        # predictions = outputs.view(grid_size, grid_size, 6).cpu().numpy()
        labels = labels.view(grid_size, grid_size, 6).cpu().numpy()

        for row in range(grid_size):
            for col in range(grid_size):
                if labels[row, col, 5] == 0:
                    true_box = labels[row, col, 0:4]
                    y_true_boxes.append(true_box)
                    y_true_classes.append(0)

                    if predictions[row, col, 4] > 0.5 and predictions[row, col, 5] < 0.5:
                        pred_box = predictions[row, col, 0:4]
                        y_pred_boxes.append(pred_box)
                        y_pred_classes.append(0)
                        iou = calculate_iou(pred_box, true_box)
                        iou_scores.append(iou)
                    else:
                        y_pred_classes.append(1)
                else:
                    y_true_classes.append(1)
                    if predictions[row, col, 4] > 0.5 and predictions[row, col, 5] < 0.5:
                        y_pred_classes.append(0)
                    else:
                        y_pred_classes.append(1)
                
                        

        # Draw boxes on the test images
        original_image = Image.open(img_path[0]).convert('RGB')
        draw = ImageDraw.Draw(original_image)

        for row in range(grid_size):
            for col in range(grid_size):
                if predictions[row, col, 4] > 0.5 and predictions[row, col, 5] < 0.5:  # If confidence score is high and classified as 0
                    x_offset, y_offset, w, h = predictions[row, col, 0:4]
                    x_center = (col + x_offset) / grid_size * 640
                    y_center = (row + y_offset) / grid_size * 640
                    box_w = w * 640
                    box_h = h * 640
                    x1 = int(x_center - box_w / 2)
                    y1 = int(y_center - box_h / 2)
                    x2 = int(x_center + box_w / 2)
                    y2 = int(y_center + box_h / 2)
                    draw.rectangle([x1, y1, x2, y2], outline='red', width=3)
                    draw.rectangle([x1, y1, x2, y2], outline='red', width=3)
                    confidence = predictions[row, col, 4]
                    draw.text((x1, y1 - 10), f"{confidence:.2f}", fill='red')
        output_path = os.path.join(output_folder, os.path.basename(img_path[0]))
        original_image.save(output_path)

mean_iou = np.mean(iou_scores) if iou_scores else 0
mean_loc_error = np.mean([math.sqrt((pred[0] - true[0]) ** 2 + (pred[1] - true[1]) ** 2) for pred, true in zip(y_pred_boxes, y_true_boxes)]) if y_true_boxes else 0
mean_size_error = np.mean([math.sqrt((pred[2] - true[2]) ** 2 + (pred[3] - true[3]) ** 2) for pred, true in zip(y_pred_boxes, y_true_boxes)]) if y_true_boxes else 0
precision = precision_score(y_true_classes, y_pred_classes, pos_label=0, labels=[0,1])
recall = recall_score(y_true_classes, y_pred_classes, pos_label=0, labels=[0,1])
f1 = f1_score(y_true_classes, y_pred_classes, pos_label=0)
print(f"Mean IoU: {mean_iou}")
print(f"Mean Box Location Error: {mean_loc_error}")
print(f"Mean Box Size Error: {mean_size_error}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 score: {f1}")