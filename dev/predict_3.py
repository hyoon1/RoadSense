import torch
import os
import timm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
from torchvision import datasets, transforms
from torchvision.transforms import Normalize, Compose, ToPILImage
from torch.utils.data import DataLoader, Dataset
from pathlib import Path
from model_3 import EfficientNetModel

data_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

inv_normalize = transforms.Compose([
    Normalize(mean=[0., 0., 0.], std=[1/0.229, 1/0.224, 1/0.225]),
    Normalize(mean=[-0.485, -0.456, -0.406], std=[1., 1., 1.])
])

def load_model(model_name, model_path, device, num_classes):
    if model_name == 'EfficientNet':
        model = EfficientNetModel(num_classes)
    else:
        raise ValueError("Unknown model name")
    
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

label_mapping = {0: 'clear', 1: 'light', 2: 'medium', 3: 'plowed'}
def plot_results(images, true_labels, pred_labels, save_path=None):
    fig, axs = plt.subplots(1, 5, figsize=(25, 5))
    
    for idx in range(5):
        ax = axs[idx]
        img = images[idx].squeeze(0)  # Remove batch dimension
        img = inv_normalize(img)  # Inverse normalize
        img = img.permute(1, 2, 0).cpu().numpy()  # Move to CPU and convert to numpy
        ax.imshow(img)
        ax.set_title(f"True: {label_mapping[true_labels[idx]]}, Pred: {label_mapping[pred_labels[idx]]}")
        ax.axis('off')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()

def main():
    # Paths to the models
    model_paths = {
        'EfficientNet': './snow_best_efficientnet_model.pth'
    }

    # Test image directory
    test_dir = Path('../Snow-Covered-Roads-Dataset-main/Snow-Covered-Roads-Dataset-main/dataset/test')

    # Create dataset and dataloader
    test_dataset = datasets.ImageFolder(test_dir, transform=data_transforms)
    test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False, num_workers=4)

    class_names = test_dataset.classes
    # Load models
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    selected_model_name = 'EfficientNet'  # Change this to the desired model name
    model_path = model_paths[selected_model_name]
    model = load_model(selected_model_name, model_path, device, len(class_names))
    
    images = []
    true_labels = []
    pred_labels = []

    with torch.no_grad():
        for i, (image, label) in enumerate(test_loader):
            if i >= 5:
                break

            image = image.to(device)
            label = label.to(device)
            
            output = model(image)
            _, pred = torch.max(output, 1)

            images.append(image.cpu())
            true_labels.append(label.item())
            pred_labels.append(pred.item())

    # Plot results
    plot_results(images, true_labels, pred_labels, save_path='predictions_snow.png')

if __name__ == '__main__':
    main()