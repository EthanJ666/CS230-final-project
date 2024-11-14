from yolo_version1_model import CustomYOLODataset, YOLO, YoloLoss
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import numpy as np
from torch.utils.data import DataLoader
import os
import math
from tqdm import tqdm


transform = transforms.Compose([
    transforms.Resize((640, 640)),
    transforms.ToTensor(),
])

grid_size = 11
num_epochs = 40
batch_size = 32

train_folder = '/home/ubuntu/CS230_final_project/CS230-final-project/drone_augmented/train'
weight_output_folder = '/home/ubuntu/CS230_final_project/CS230-final-project/yolo_v1_weights'

dataset = CustomYOLODataset(image_folder=train_folder+'/images', label_folder=train_folder+'/labels', grid_size=grid_size, transform=transform)
train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
yolo = YOLO(grid_size=grid_size)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
yolo = yolo.to(device)

criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(yolo.parameters(), lr=0.001)

for epoch in range(num_epochs):
    running_loss = 0.0

    for batch_idx, data in enumerate(tqdm(train_loader, desc=f"Epoch [{epoch + 1}/{num_epochs}]", unit="batch")):
        inputs, labels, p = data
        inputs = inputs.to(device)
        labels = labels.to(device)

        predictions = yolo(inputs)
        loss = criterion(predictions, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    print(f'Epoch [{epoch + 1}], Loss: {running_loss / len(train_loader)}')
print('Finished Training')
torch.save(yolo.state_dict(), weight_output_folder + '/yolo_model_epoch40_grid11_data_aug.pth')
print('Model saved')