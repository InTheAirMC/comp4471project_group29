from cProfile import label
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
    def __init__(self, image_dir, json_dir, transform=None, max_samples = None):
        self.image_dir = image_dir
        self.json_dir = json_dir
        self.transform = transform
        self.all_imgs = sorted([f for f in os.listdir(image_dir)])
        if max_samples is not None:
            self.all_imgs = self.all_imgs[:max_samples]

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

        label = category - 1
        return img, label

class CropModel(nn.Module):
    def __init__(self):
        super().__init__()
        backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        for param in backbone.parameters():
            param.requires_grad = False

        for param in backbone.layer4.parameters():
            param.requires_grad = True

        self.feature_layers = nn.Sequential(*list(backbone.children())[:-1])
        self.dropout = nn.Dropout(p=0.7)
        
        self.classifier = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.Dropout(0.3),
            nn.Linear(512, 13)
        )
        
    def forward(self, x):
        x = self.feature_layers(x)
        x = torch.flatten(x, 1) 
        logits = self.classifier(x) 
        return logits
    
def evaluate(model, dataloader, criterion):
    model.eval()

    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.long().to(device)

            logits = model(images)
            loss = criterion(logits, labels)

            total_loss += loss.item()

            preds = torch.argmax(logits, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    avg_loss = total_loss / len(dataloader)
    accuracy = correct / total

    return avg_loss, accuracy
        
if __name__ == '__main__':
    image_dir = "stage2_crops"
    json_dir = "train/annos"
    val_image_dir = "val_crops"
    val_json_dir = "validation/annos"
    batch_size = 64
    num_epochs = 10
    learning_rate = 1e-4

    train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomResizedCrop(224, scale=(0.75, 1.0)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=10),
    transforms.ColorJitter(
        brightness=0.2,
        contrast=0.2,
        saturation=0.2,
        hue=0.05
    ),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
    ])

    train_datasets = CropDataSet(image_dir, json_dir, transform=train_transform)
    train_loader = DataLoader(train_datasets, batch_size=batch_size, shuffle=True, drop_last=True)
    val_datasets = CropDataSet(val_image_dir, val_json_dir, transform=val_transform)
    val_loader = DataLoader(val_datasets, batch_size=batch_size, shuffle=False)
    model = CropModel().to(device)
    optimizer = optim.AdamW([
        {"params": model.feature_layers.parameters(), "lr": 1e-4},
        {"params": model.classifier.parameters(), "lr": 1e-3},
        ], weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=3
    )

    for i in range(num_epochs):
        start = time.time()
        print(f"Epoch [{i+1}]")

        model.train()
        total_loss = 0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.long().to(device)

            optimizer.zero_grad()

            logits = model(images)
            loss = criterion(logits, labels)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            # Accuracy calculation
            preds = torch.argmax(logits, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        train_loss = total_loss / len(train_loader)
        train_acc = correct / total

        val_loss, val_acc = evaluate(model, val_loader, criterion)
        scheduler.step(val_acc)

        print(f"time = {time.time() - start:.2f}s")
        print(
            f"Epoch [{i+1}/{num_epochs}], "
            f"Train Loss: {train_loss:.4f}, "
            f"Train Acc: {train_acc * 100:.2f}%, "
            f"Val Loss: {val_loss:.4f}, "
            f"Val Acc: {val_acc * 100:.2f}%"
        )