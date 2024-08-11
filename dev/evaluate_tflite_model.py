from ultralytics import YOLO
import torch
import numpy as np
import os
import argparse
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

# Set up argument parser
parser = argparse.ArgumentParser(description="Evaluate a TFLite model on test images.")
parser.add_argument('--model_path', type=str, default='./best_float16.tflite', help="Path to the TFLite model. Defaults to './best_float16.tflite'.")

args = parser.parse_args()

# Load the exported TFLite model
tflite_model = YOLO(args.model_path)

# Define the test images directory
test_imgs_dir = '../data-collection/RDD2022/Japan_filtered/filtered/test/images'

# Perform inference on the test images
with torch.no_grad():
    results = tflite_model.predict(source=test_imgs_dir)

# Get a list of all image files in the test directory
image_paths = [os.path.join(test_imgs_dir, fname) for fname in os.listdir(test_imgs_dir) if fname.lower().endswith(('.png', '.jpg', '.jpeg'))]

# Function to plot predictions
def plot_predictions(image_path, boxes, scores, classes, names, ax):
    # Load the image
    img = Image.open(image_path).convert("RGB")
    ax.imshow(img)
    
    # Iterate over each prediction
    for box, score, cls in zip(boxes, scores, classes):
        # Convert tensor to numpy if necessary
        if isinstance(box, torch.Tensor):
            box = box.cpu().numpy()
        if isinstance(score, torch.Tensor):
            score = score.cpu().numpy()
        if isinstance(cls, torch.Tensor):
            cls = cls.cpu().numpy()

        # Unpack the box coordinates
        x1, y1, x2, y2 = box
        width, height = x2 - x1, y2 - y1
        
        # Create a rectangle patch
        rect = patches.Rectangle((x1, y1), width, height, linewidth=2, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        
        # Add the class label and confidence
        label = f"{names[int(cls)]}: {score:.2f}"
        ax.text(x1, y1, label, color='white', fontsize=12, bbox=dict(facecolor='red', alpha=0.5))

    ax.axis('off')  # Turn off axis

# Plot only 8 predictions
num_plots = 8
fig, axs = plt.subplots(2, 4, figsize=(20, 10))

for idx, (img_path, result) in enumerate(zip(image_paths[:num_plots], results[:num_plots])):
    boxes = result.boxes.xyxy
    scores = result.boxes.conf
    classes = result.boxes.cls
    plot_predictions(img_path, boxes, scores, classes, tflite_model.names, axs[idx // 4, idx % 4])

plt.tight_layout()
plt.show()
