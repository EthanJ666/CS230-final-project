import torch
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Resize, ToTensor
from feature_object_detection_model import SimpleObjectDetector
from feature_dataset import SimpleDataset
from tqdm import tqdm

def calculate_iou(pred_box, true_box):
    """
    Calculate IoU between two boxes.
    Boxes are in [x_center, y_center, w, h] format.
    """
    pred_x1 = pred_box[0] - pred_box[2] / 2
    pred_y1 = pred_box[1] - pred_box[3] / 2
    pred_x2 = pred_box[0] + pred_box[2] / 2
    pred_y2 = pred_box[1] + pred_box[3] / 2

    true_x1 = true_box[0] - true_box[2] / 2
    true_y1 = true_box[1] - true_box[3] / 2
    true_x2 = true_box[0] + true_box[2] / 2
    true_y2 = true_box[1] + true_box[3] / 2

    inter_x1 = max(pred_x1, true_x1)
    inter_y1 = max(pred_y1, true_y1)
    inter_x2 = min(pred_x2, true_x2)
    inter_y2 = min(pred_y2, true_y1)

    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
    pred_area = (pred_x2 - pred_x1) * (pred_y2 - pred_y1)
    true_area = (true_x2 - true_x1) * (true_y2 - true_y1)

    union_area = pred_area + true_area - inter_area

    return inter_area / union_area if union_area > 0 else 0

def evaluate_model(model, dataloader, device, iou_threshold=0.5):
    model.eval()
    total_iou = 0.0
    total_precision = 0
    total_recall = 0
    num_samples = 0

    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Evaluating"):
            images = images.to(device)
            labels = labels.to(device)

            class_logits, bbox_preds = model(images)
            predicted_classes = torch.argmax(class_logits, dim=1)

            for i in range(len(images)):
                pred_box = bbox_preds[i].cpu().numpy()
                true_box = labels[i, 1:].cpu().numpy()
                true_class = labels[i, 0].item()

                # Calculate IoU
                iou = calculate_iou(pred_box, true_box)
                total_iou += iou

                # Calculate Precision and Recall
                if iou >= iou_threshold and predicted_classes[i] == true_class:
                    total_precision += 1
                    total_recall += 1
                elif predicted_classes[i] == true_class:
                    total_precision += 1

                num_samples += 1

    mean_iou = total_iou / num_samples
    precision = total_precision / num_samples
    recall = total_recall / num_samples

    return mean_iou, precision, recall

if __name__ == "__main__":
    # Configuration
    test_folder = "./drone_dataset/test"
    model_path = "./simple_object_detector.pth"
    batch_size = 8
    num_classes = 2

    # Prepare test dataset and dataloader
    transform = Compose([Resize((640, 640)), ToTensor()])
    test_dataset = DroneDataset(
        image_folder=f"{test_folder}/images",
        label_folder=f"{test_folder}/labels",
        transform=transform,
    )
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleObjectDetector(num_classes=num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)

    # Evaluate model
    mean_iou, precision, recall = evaluate_model(model, test_loader, device)

    # Print results
    print(f"Mean IoU: {mean_iou:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")

