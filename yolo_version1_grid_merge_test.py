from yolo_version1_model import CustomYOLODataset, YOLO
import torch
import torchvision.transforms as transforms
import numpy as np
from torch.utils.data import DataLoader
from PIL import Image, ImageDraw
import os
import math


def calculate_iou(box1, box2):
    """
    Calculate IoU between two boxes.
    Each box is represented as [x, y, w, h] in normalized form.
    """
    x1_min = box1[0] - box1[2] / 2
    y1_min = box1[1] - box1[3] / 2
    x1_max = box1[0] + box1[2] / 2
    y1_max = box1[1] + box1[3] / 2

    x2_min = box2[0] - box2[2] / 2
    y2_min = box2[1] - box2[3] / 2
    x2_max = box2[0] + box2[2] / 2
    y2_max = box2[1] + box2[3] / 2

    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)

    inter_area = max(0, inter_x_max - inter_x_min) * max(0, inter_y_max - inter_y_min)
    box1_area = (x1_max - x1_min) * (y1_max - y1_min)
    box2_area = (x2_max - x2_min) * (y2_max - y2_min)

    union_area = box1_area + box2_area - inter_area
    return inter_area / union_area if union_area > 0 else 0


def rescale_predictions(prediction, grid_size, image_size=640):
    prediction_view = prediction.view(-1, grid_size, grid_size, 6)
    rescaled_pred = []
    for i in range(grid_size):
        for j in range(grid_size):
            x_offset, y_offset, w, h, confid, cls = prediction_view[:, i, j, :]
            x = (x_offset + j) / grid_size
            y = (y_offset + i) / grid_size
            rescaled_pred.append([x, y, w, h, confid, cls])
    return torch.tensor(rescaled_pred)


def merge_predictions(image, models, grid_sizes, iou_threshold=0.5):
    merged_pred = []
    for model, grid_size in zip(models, grid_sizes):
        with torch.no_grad():
            prediction = torch.sigmoid(model(image))
            rescaled_prediction = rescale_predictions(prediction, grid_size)
            merged_pred.append(rescaled_prediction)
    merged_predictions = torch.cat(merged_pred, dim=0)
    return merged_predictions


# Configurations
test_folder = '/path/to/test/folder'
output_folder = '/path/to/output/folder'
grid_sizes = [7, 5, 3]
model_paths = [
    '/path/to/yolo_model_grid7.pth',
    '/path/to/yolo_model_grid5.pth',
    '/path/to/yolo_model_grid3.pth'
]
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

transform = transforms.Compose([
    transforms.Resize((640, 640)),
    transforms.ToTensor(),
])

# Load models
yolo_models = [YOLO(grid_size=grid).to(device) for grid in grid_sizes]
for model, model_path in zip(yolo_models, model_paths):
    model.load_state_dict(torch.load(model_path))
    model.eval()

# Load test dataset
test_dataset = CustomYOLODataset(
    image_folder=test_folder + '/images',
    label_folder=test_folder + '/labels',
    grid_size=1,
    transform=transform
)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
os.makedirs(output_folder, exist_ok=True)

iou_threshold = 0.5  # IoU threshold for matching
true_positive = 0
false_positive = 0
false_negative = 0
iou_scores = []

# Testing loop
print('Starting Testing')
with torch.no_grad():
    for data in test_loader:
        inputs, labels, img_path = data
        inputs = inputs.to(device)
        labels = labels.view(-1, 5).cpu().numpy()  # Convert to normalized ground truth format

        combined_predictions = merge_predictions(inputs, yolo_models, grid_sizes)
        combined_predictions = combined_predictions.cpu().numpy()

        # Normalize predictions to [0, 1]
        predictions = []
        for pred in combined_predictions:
            x_center, y_center, w, h, conf, cls = pred
            if conf > 0.5:  # Confidence threshold
                predictions.append([x_center / 640, y_center / 640, w / 640, h / 640, cls])

        # IoU calculation
        matched_gt = set()
        for pred_box in predictions:
            best_iou = 0
            best_gt_idx = -1
            for idx, gt_box in enumerate(labels):
                iou = calculate_iou(pred_box[:4], gt_box[:4])
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = idx
            if best_iou >= iou_threshold and best_gt_idx not in matched_gt:
                true_positive += 1
                matched_gt.add(best_gt_idx)
                iou_scores.append(best_iou)
            else:
                false_positive += 1

        false_negative += len(labels) - len(matched_gt)

        # Annotate image
        original_image = Image.open(img_path[0]).convert('RGB')
        draw = ImageDraw.Draw(original_image)
        for pred in predictions:
            x_center, y_center, w, h, conf = pred[:5]
            x1 = int((x_center - w / 2) * 640)
            y1 = int((y_center - h / 2) * 640)
            x2 = int((x_center + w / 2) * 640)
            y2 = int((y_center + h / 2) * 640)
            draw.rectangle([x1, y1, x2, y2], outline='red', width=3)
            draw.text((x1, y1 - 10), f"{conf:.2f}", fill='red')
        output_path = os.path.join(output_folder, os.path.basename(img_path[0]))
        original_image.save(output_path)

precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0
recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0
mean_iou = np.mean(iou_scores) if iou_scores else 0

print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"Mean IoU: {mean_iou}")
print('Testing completed. Annotated images saved.')
