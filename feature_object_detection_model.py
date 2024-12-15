import torch
import torch.nn as nn
from torchvision.models import resnet18
import os
from torch.utils.data import Dataset
from PIL import Image
from torchvision.transforms import transforms

class SimpleObjectDetector(nn.Module):
    def __init__(self, num_classes=2, pretrained=True):
        super(SimpleObjectDetector, self).__init__()

        # Backbone: Pretrained ResNet18
        self.backbone = resnet18(pretrained=pretrained)
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-2])  # Remove the classification head

        # Classification Head
        self.cls_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)  # Predict class probabilities
        )

        # Bounding Box Regression Head
        self.bbox_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 4)  # Predict [x, y, w, h]
        )

    def forward(self, x):
        # Extract features
        features = self.backbone(x)

        # Predict class probabilities and bounding boxes
        class_logits = self.cls_head(features)
        bbox_preds = self.bbox_head(features)

        return class_logits, bbox_preds


class DroneDataset(Dataset):
    def __init__(self, image_folder, label_folder, transform=None):
        self.image_folder = image_folder
        self.label_folder = label_folder
        self.transform = transform
        self.image_files = [f for f in os.listdir(image_folder) if f.endswith('.jpg')]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_folder, self.image_files[idx])
        label_path = os.path.join(self.label_folder, self.image_files[idx].replace('.jpg', '.txt'))

        # Load and transform the image
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        # Load bounding boxes
        bboxes = []
        with open(label_path, "r") as f:
            for line in f.readlines():
                x, y, w, h, cls = map(float, line.strip().split())
                bboxes.append([x, y, w, h, int(cls)])

        # Only supports a single bounding box for simplicity
        bbox = bboxes[0] if len(bboxes) > 0 else [0.0, 0.0, 0.0, 0.0, -1]

        return image, torch.tensor(bbox)
