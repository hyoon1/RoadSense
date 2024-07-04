import os
import yaml
import torch
import pandas as pd
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from models_2 import RoadNet, EfficientNetModel, ImprovedEfficientNetModel, ViTModel, ImprovedViTModel

# Define a function to load the appropriate model
def load_model(model_name):
    if model_name == 'RoadNet':
        return RoadNet()
    elif model_name == 'EfficientNet':
        return EfficientNetModel()
    elif model_name == 'ImprovedEfficientNet':
        return ImprovedEfficientNetModel()
    elif model_name == 'ViT':
        return ViTModel()
    elif model_name == 'ImprovedViT':
        return ImprovedViTModel()
    else:
        raise ValueError(f"Unknown model name: {model_name}")

# Function to calculate accuracy
def calculate_accuracy(outputs, labels):
    _, preds = torch.max(outputs, 1)
    return torch.sum(preds == labels).item() / len(labels)

# Function to train the model
def train_model(model, train_loader, valid_loader, criterion, optimizer, device, num_epochs=50):
    model.to(device)
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        running_corrects = 0

        for i, (images, meta, labels) in enumerate(train_loader):
            images = images.to(device)
            meta = meta.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images, meta)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            running_corrects += calculate_accuracy(outputs, labels) * images.size(0)
            step_loss = loss.item()
            step_acc = calculate_accuracy(outputs, labels)
            print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {step_loss:.4f}, Accuracy: {step_acc:.4f}')

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = running_corrects / len(train_loader.dataset)

        # Validation phase
        model.eval()
        val_running_loss = 0.0
        val_running_corrects = 0
        with torch.no_grad():
            for images, meta, labels in valid_loader:
                images = images.to(device)
                meta = meta.to(device)
                labels = labels.to(device)

                outputs = model(images, meta)
                loss = criterion(outputs, labels)

                val_running_loss += loss.item() * images.size(0)
                val_running_corrects += calculate_accuracy(outputs, labels) * images.size(0)

        val_epoch_loss = val_running_loss / len(valid_loader.dataset)
        val_epoch_acc = val_running_corrects / len(valid_loader.dataset)

        print(f'Epoch {epoch+1}/{num_epochs}, '
              f'Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.4f}, '
              f'Val Loss: {val_epoch_loss:.4f}, Val Acc: {val_epoch_acc:.4f}')

# Define a custom dataset class for road damage data
class RoadDamageDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe
        self.transform = transform
        
    def __len__(self):
        return len(self.dataframe)
    
    def __getitem__(self, idx):
        img_path = self.dataframe.iloc[idx]['Image_Path'] # Get the image path
        image = Image.open(img_path).convert("RGB") # Open the image and convert to RGB
        label = self.dataframe.iloc[idx]['Severity_Level'] # Get the severity level label
        meta = self.dataframe.iloc[idx]['Number of Potholes'] # Get the number of potholes
        
        if self.transform:
            image = self.transform(image) # Apply the transform to the image
        
        return image, torch.tensor(meta, dtype=torch.float32), torch.tensor(label, dtype=torch.long)

def main():
    # Paths to the dataset CSV files
    train_csv_path = "../damage_assessment/train_Dataset_Info.csv"
    valid_csv_path = "../damage_assessment/test_Dataset_Info.csv"

    # Read the CSV files
    train_data = pd.read_csv(train_csv_path)
    valid_data = pd.read_csv(valid_csv_path)

    # Define data augmentation and preprocessing for training dataset
    train_transforms = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1)], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Define data preprocessing for validation dataset
    valid_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Create datasets and dataloaders
    train_dataset = RoadDamageDataset(train_data, transform=train_transforms)
    valid_dataset = RoadDamageDataset(valid_data, transform=valid_transforms)

    train_loader = DataLoader(dataset=train_dataset, batch_size=32, shuffle=True)
    valid_loader = DataLoader(dataset=valid_dataset, batch_size=32, shuffle=False)

    # Select the model
    model_name = 'RoadNet'  # Change this to select different models
    model = load_model(model_name)

    # Define the loss function and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.002)

    # Train the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_model(model, train_loader, valid_loader, criterion, optimizer, device, num_epochs=50)

if __name__ == '__main__':
    main()