import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import numpy as np
from torch.utils.data import Dataset
import os
import math
from tqdm import tqdm
from PIL import Image
import torch.nn.functional as F


def create_conv_block(in_channels, out_channels, kernel_size, stride, padding):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.LeakyReLU(0.1),
    )

class YOLO(nn.Module):
    def __init__(self, grid_size):
        super(YOLO, self).__init__()
        self.grid_size = grid_size
        
        self.conv1 = create_conv_block(3, 16, 3, 1, 1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = create_conv_block(16, 32, 3, 1, 1)
        self.conv3 = create_conv_block(32, 64, 3, 1, 1)
        self.conv4 = create_conv_block(64, 128, 3, 1, 1)
        self.conv5 = create_conv_block(128, 256, 3, 1, 1)
        self.fc1 = nn.Linear(256 * (640 // 32) * (640 // 32), 1024)
        self.fc2 = nn.Linear(1024, grid_size * grid_size * (6))

    def forward(self, x):
        x = self.pool(self.conv1(x))
        x = self.pool(self.conv2(x))
        x = self.pool(self.conv3(x))
        x = self.pool(self.conv4(x))
        x = self.pool(self.conv5(x))
        x = x.view(-1, 256 * (640 // 32) * (640 // 32))
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class CustomYOLODataset(Dataset):
    def __init__(self, image_folder, label_folder, grid_size, transform=None):
        self.image_folder = image_folder
        self.label_folder = label_folder
        self.transform = transform
        self.image_files = [f for f in os.listdir(image_folder) if f.endswith('.jpg')]
        self.grid_size=grid_size

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_folder, self.image_files[idx])
        label_path = os.path.join(self.label_folder, self.image_files[idx].replace('.jpg', '.txt'))

        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)

        #double check this transformation
        label = torch.zeros((self.grid_size, self.grid_size, 6))
        label[..., 5] = 1
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f.readlines():
                    class_id, x, y, w, h = map(float, line.split())
                    if class_id == 0:
                        grid_x = math.floor(x * self.grid_size)
                        grid_y = math.floor(y * self.grid_size)
                        x_offset = (x * self.grid_size) - grid_x
                        y_offset = (y * self.grid_size) - grid_y
                        label[grid_y, grid_x, :] = torch.tensor([x_offset, y_offset, w, h, 1, class_id])
        else:
            raise Exception(f'{label_path} does not exist')

        return image, label.view(-1), img_path


#This loss function does not work, please review
class YoloLoss(nn.Module):
    def __init__(self, grid_size, lambda_coord=1.5, lambda_noobj=1, lamda_class=1):
        super(YoloLoss, self).__init__()
        self.mse = nn.MSELoss(reduction="mean")
        self.bce = nn.BCEWithLogitsLoss()
        self.grid_size = grid_size
        self.lambda_coord = lambda_coord
        self.lambda_noobj = lambda_noobj
        self.lambda_class = lamda_class

    def forward(self, predictions, target):
        predictions = predictions.view(-1, self.grid_size, self.grid_size, 6)
        target = target.view(-1, self.grid_size, self.grid_size, 6)
        
        pred_boxes = predictions[..., :4]  # x, y, w, h
        target_boxes = target[..., :4]
        pred_conf = predictions[..., 4]  # Confidence score
        target_conf = target[..., 4]
        pred_class = predictions[..., 5]  # Class label
        target_class = target[..., 5]

        box_loss = self.mse(pred_boxes, target_boxes) * self.lambda_coord
        conf_obj_loss = self.bce(pred_conf * target_conf, target_conf) 
        conf_noobj_loss = self.bce(pred_conf * (1 - target_conf), target_conf * (1 - target_conf)) * self.lambda_noobj
        class_loss = self.bce(pred_class, target_class) * self.lambda_class
        total_loss = box_loss + conf_obj_loss + conf_noobj_loss + class_loss
        # print(box_loss , conf_obj_loss , conf_noobj_loss , class_loss)

        return total_loss