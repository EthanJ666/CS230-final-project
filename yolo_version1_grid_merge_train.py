from yolo_version1_model import CustomYOLODataset, YOLO, YoloLoss2
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

grid_sizes = [7, 5, 3]
num_epochs = 40
batch_size = 128

train_folder = '/home/ubuntu/CS230_final_project/CS230-final-project/drone_dataset/train'
weight_output_folder = '/home/ubuntu/CS230_final_project/CS230-final-project/yolo_v1_weights'

for grid_size in grid_sizes:
    print(f"Currently training model with grid_size = {grid_size}")
    dataset = CustomYOLODataset(image_folder=train_folder+'/images', label_folder=train_folder+'/labels', grid_size=grid_size, transform=transform)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    yolo = YOLO(grid_size=grid_size)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    yolo = yolo.to(device)

    # criterion = nn.BCEWithLogitsLoss()
    criterion = YoloLoss2(grid_size=grid_size, lambda_coord=1.5, lamda_class=1.1, lambda_box=2, lambda_conf=1)
    optimizer = optim.Adam(yolo.parameters(), lr=0.0005, weight_decay=1e-4)

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
    torch.save(yolo.state_dict(), weight_output_folder + f'/yolo_model_grid{grid_size}_merge_model.pth')
    print('Model saved')