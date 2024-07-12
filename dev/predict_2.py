import torch
import os
import timm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from models_2 import RoadNet, EfficientNetModel, ImprovedEfficientNetModel, ViTModel, ImprovedViTModel

# Define transformations for the images
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
# Define inverse normalization
inv_normalize = transforms.Compose([
    transforms.Normalize(mean=[0., 0., 0.], std=[1/0.229, 1/0.224, 1/0.225]),
    transforms.Normalize(mean=[-0.485, -0.456, -0.406], std=[1., 1., 1.])
])

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

# Function to load models
def load_model(model_name, model_path, device):
    if model_name == 'RoadNet':
        model = RoadNet()
    elif model_name == 'EfficientNet':
        model = EfficientNetModel()
    elif model_name == 'ImprovedEfficientNet':
        model = ImprovedEfficientNetModel()
    elif model_name == 'ViT':
        model = ViTModel()
    elif model_name == 'ImprovedViT':
        model = ImprovedViTModel()
    else:
        raise ValueError("Unknown model name")
    
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

label_mapping = {0: 'C', 1: 'B', 2: 'A', 3: 'S'}
# Function to plot predictions and ground truths
def plot_results(images, true_labels, pred_labels, num_potholes, save_path=None):
    fig, axs = plt.subplots(1, 5, figsize=(25, 5))
    
    for idx in range(5):
        ax = axs[idx]
        img = images[idx].squeeze(0)  # Remove batch dimension
        img = inv_normalize(img)  # Inverse normalize
        img = img.permute(1, 2, 0).cpu().numpy()  # Move to CPU and convert to numpy
        ax.imshow(img)
        ax.set_title(f"True: {label_mapping[true_labels[idx]]}, Pred: {label_mapping[pred_labels[idx]]}, Potholes: {num_potholes[idx]}")
        ax.axis('off')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()

def main():
    # Paths to the models
    model_paths = {
        'RoadNet': './best_roadnet_model.pth',
        'EfficientNet': './best_efficientnet_model.pth',
        'ImprovedEfficientNet': './best_efficientnet_model_2.pth',
        'ViT': './best_vit_model.pth',
        'ImprovedViT': './best_vit_model_2.pth'
    }

    # Test image directory and metadata
    test_csv_path = "../damage_assessment/test_Dataset_Info.csv"
    test_data = pd.read_csv(test_csv_path)

    # Create dataset and dataloader
    test_dataset = RoadDamageDataset(test_data, transform=transform)
    test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False)

    # Load models
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    selected_model_name = 'EfficientNet'  # Change this to the desired model name
    model_path = model_paths[selected_model_name]
    model = load_model(selected_model_name, model_path, device)
    
    images = []
    true_labels = []
    pred_labels = []
    num_potholes = []

    with torch.no_grad():
        for i, (image, potholes, label) in enumerate(test_loader):
            if i >= 5:
                break

            image = image.to(device)
            potholes = potholes.to(device)
            label = label.to(device)
            
            output = model(image, potholes)
            _, pred = torch.max(output, 1)

            images.append(image.cpu())
            true_labels.append(label.item())
            pred_labels.append(pred.item())
            num_potholes.append(potholes.item())

    # Plot results
    plot_results(images, true_labels, pred_labels, num_potholes, save_path='predictions.png')

if __name__ == '__main__':
    main()
