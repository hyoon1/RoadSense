import torch
import os
import yaml
import wandb
from ultralytics import YOLO

def main():
    # Define label map
    label_map = {
        "D00": 0,
        "D10": 1,
        "D20": 2,
        "D40": 3
    }

    names = {v:k for k,v in label_map.items()}

    # Define directory paths for dataset organization
    root_dir = "../data-collection/Japan_filtered/filtered"
    train_imgs_dir = "../data-collection/Japan_filtered/filtered/train/images"
    valid_imgs_dir = "../data-collection/Japan_filtered/filtered/val/images"
    train_labels_dir = "../data-collection/Japan_filtered/filtered/train/labels"
    valid_labels_dir = "../data-collection/Japan_filtered/filtered/val/labels"

    assert os.path.exists(train_imgs_dir), print("Image path not found")
    assert os.path.exists(train_labels_dir), print("Annotation path not found")

    # Define YOLO format dictionary for training configuration
    yolo_format = {
        'path': os.path.abspath(root_dir),           # Root directory containing all data
        'train': os.path.abspath(train_imgs_dir),    # Directory containing training images
        'val': os.path.abspath(valid_imgs_dir),      # Directory containing validation images
        'nc': 4,                   # Number of classes
        'names': names              # Class names
    }

    # Write YOLO format dictionary to a YAML file
    with open(root_dir+'/data.yaml', 'w') as outfile:
        yaml.dump(yolo_format, outfile, default_flow_style=False)

    # Load YOLO model
    model = YOLO('yolov8n.pt')
    
    # Train the model
    model.train(data=root_dir+"/data.yaml",  # Path to the YAML training configuration file
            epochs=150,     # Number of training epochs
            patience=20,    # Patience for early stopping
            batch=32,        # Batch size
            lr0=0.002,     # Initial learning rate
            imgsz=640,
            scale=0.7,
            shear=0.01,
            perspective=0.0001,
            mosaic=0.5,
            mixup=0.1,
            copy_paste=0.05
            )

if __name__ == '__main__':
    main()