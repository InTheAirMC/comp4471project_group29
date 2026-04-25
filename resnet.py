import os
import json
import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from torch import optim
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
from PIL import Image
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

class CropDataSet(Dataset):
    def __init__(self, image_dir, json_dir, transform=None):
        self.image_dir = image_dir
        self.json_dir = json_dir
        self.transform = transform
        self.all_imgs = sorted([f for f in os.listdir(image_dir)])

    # Return the JSON of the image
    def get_json(self, image_str):
        json_path = os.path.join(self.json_dir, f"{image_str}.json")
        with open(json_path, 'r') as f:
            data = json.load(f)
        return data

    def __len__(self):
        return len(self.all_imgs)
    
    def __getitem__(self, idx):
        image_name = self.all_imgs[idx]
        image_id = image_name.split('_')[0]
        n_item = image_name.split('_')[1].split('.')[0]
        img_path = os.path.join(self.image_dir, image_name)
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        
        json_data = self.get_json(image_id)
        category = json_data[n_item]["category_id"]

        sleeve = 1 if category in [2, 4, 11] else 0
        target_type = 1 if category >= 7 else 0
        return img, sleeve, target_type

class CropModel(nn.Module):
    def __init__(self):
        super().__init__()
        backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        self.feature_layers = nn.Sequential(*list(backbone.children())[:-1])
        self.sleeve_classifier = nn.Linear(2048, 1)
        self.type_classifier = nn.Linear(2048, 1)
        
    def forward(self, x):
        tensor = self.feature_layers(x)
        x = torch.flatten(tensor, 1)
        x = nn.Dropout(x)
        sleeve_out = self.sleeve_classifier(x)
        type_out = self.type_classifier(x)
        return sleeve_out, type_out
        
if __name__ == '__main__':
    image_dir = "stage2_crops"
    json_dir = "train/annos"
    batch_size = 16
    num_epochs = 3
    learning_rate = 0.001

    transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
    ])

    datasets = CropDataSet(image_dir, json_dir, transform=transform)
    dataloader = DataLoader(datasets, batch_size=batch_size, shuffle=True)
    model = CropModel().to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.BCEWithLogitsLoss()

    for i in range(num_epochs):
        start = time.time()
        print(f"Epoch [{i+1}]")
        model.train()
        total_loss = 0
        for images, sleeve_labels, type_labels in dataloader:
            images = images.to(device)
            sleeve_labels = sleeve_labels.float().unsqueeze(1).to(device)
            type_labels = type_labels.float().unsqueeze(1).to(device)

            optimizer.zero_grad()
            sleeve_out, type_out = model(images)
            loss_sleeve = criterion(sleeve_out, sleeve_labels)
            loss_type = criterion(type_out, type_labels)
            loss = loss_sleeve + loss_type
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(dataloader)
        print(f"time = {time.time()-start}")
        print(f"Epoch [{i+1}/{num_epochs}], Loss: {avg_loss:.4f}")