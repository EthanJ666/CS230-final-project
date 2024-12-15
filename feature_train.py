from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor, Resize
import torch
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
from feature_object_detection_model import SimpleObjectDetector, DroneDataset


# Configuration
num_classes = 2
num_epochs = 50
batch_size = 32
learning_rate = 1e-4
train_folder = './drone_dataset/train'
model_output_path = './simple_object_detector.pth'

# Dataset and DataLoader
transform = Compose([Resize((640, 640)), ToTensor()])
dataset = DroneDataset(image_folder=f"{train_folder}/images", label_folder=f"{train_folder}/labels", transform=transform)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Model, optimizer, and loss
model = SimpleObjectDetector(num_classes=num_classes)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Losses
classification_loss = nn.CrossEntropyLoss()  # For class predictions
bbox_loss = nn.SmoothL1Loss()  # For bounding box regression

# Training loop
for epoch in range(num_epochs):
    model.train()
    epoch_cls_loss = 0.0
    epoch_bbox_loss = 0.0

    for images, labels in tqdm(dataloader, desc=f"Epoch [{epoch + 1}/{num_epochs}]"):
        images = images.to(device)
        bboxes = labels[:, :4].to(device)  # [x, y, w, h]
        classes = labels[:, 4].long().to(device)  # class labels

        optimizer.zero_grad()
        class_logits, bbox_preds = model(images)

        # Compute losses
        cls_loss = classification_loss(class_logits, classes)
        reg_loss = bbox_loss(bbox_preds, bboxes)

        loss = cls_loss + reg_loss
        loss.backward()
        optimizer.step()

        epoch_cls_loss += cls_loss.item()
        epoch_bbox_loss += reg_loss.item()

    print(f"Epoch {epoch + 1}: Classification Loss = {epoch_cls_loss / len(dataloader):.4f}, BBox Loss = {epoch_bbox_loss / len(dataloader):.4f}")

# Save the model
torch.save(model.state_dict(), model_output_path)
print(f"Model saved to {model_output_path}")

