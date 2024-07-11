import os
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import pandas as pd
from PIL import Image
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from pathlib import Path
from model_3 import EfficientNetModel

# Define a function to load the appropriate model
def load_model(model_name, num_classes):
    if model_name == 'EfficientNet':
        return EfficientNetModel(num_classes)
    else:
        raise ValueError(f"Unknown model name: {model_name}")
    
# Function to calculate accuracy
def calculate_accuracy(outputs, labels):
    _, preds = torch.max(outputs, 1)
    return torch.sum(preds == labels).item() / len(labels)

def train_model(model, train_loader, valid_loader, criterion, optimizer,scheduler, device, num_epochs=50):
    model.to(device)
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 100 == 99:
                print(f"[{epoch+1}, {i+1}] loss: {running_loss / (i+1)}") 

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader)}")
        scheduler.step()
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in valid_loader:
                images, labels = images.to(device), labels.to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_loss /= len(valid_loader)
        val_acc = 100 * correct / total
        print(f"Validation Loss: {val_loss}, Validation Accuracy: {val_acc}%")

def main():
    # Paths to the dataset
    train_dir = Path('../Snow-Covered-Roads-Dataset-main/Snow-Covered-Roads-Dataset-main/dataset/augmented_train')
    test_dir = Path('../Snow-Covered-Roads-Dataset-main/Snow-Covered-Roads-Dataset-main/dataset/test')
    
    # preprocessing for training dataset
    data_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Create datasets and dataloaders
    full_dataset = datasets.ImageFolder(train_dir, transform=data_transforms)

    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    test_dataset = datasets.ImageFolder(test_dir, transform=data_transforms)

    dataloaders = {
        'train': DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4),
        'val': DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=4),
        'test': DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=4)
    }

    dataset_sizes = {'train': len(train_dataset), 'val': len(val_dataset), 'test': len(test_dataset)}
    class_names = full_dataset.classes
    # Select the model
    model_name = 'EfficientNet'  # Change this to select different models
    model = load_model(model_name, len(class_names))

    # Define the loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.002)
    scheduler = CosineAnnealingLR(optimizer, T_max=10, eta_min=1e-6)
    # Train the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_model(model, dataloaders['train'], dataloaders['val'], criterion, optimizer, scheduler, device, num_epochs=50)

if __name__ == '__main__':
    main()