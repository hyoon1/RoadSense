import torch
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
from ultralytics import YOLO

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

# Function to read labels from a file
def read_labels(label_path, img_width, img_height):
    boxes = []
    classes = []
    with open(label_path, 'r') as file:
        for line in file.readlines():
            class_id, x_center, y_center, width, height = map(float, line.strip().split())
            classes.append(int(class_id))
            x1 = (x_center - width / 2) * img_width
            y1 = (y_center - height / 2) * img_height
            x2 = (x_center + width / 2) * img_width
            y2 = (y_center + height / 2) * img_height
            boxes.append([x1, y1, x2, y2])
    return boxes, classes

# Function to plot ground truth labels
def plot_ground_truth(image_path, label_path, names, ax):
    # Load the image
    img = Image.open(image_path).convert("RGB")
    img_width, img_height = img.size

    # Read labels
    boxes, classes = read_labels(label_path, img_width, img_height)
    
    # Plot the image and labels
    ax.imshow(img)
    
    # Iterate over each label
    for box, cls in zip(boxes, classes):
        x1, y1, x2, y2 = box
        width, height = x2 - x1, y2 - y1
        
        # Create a rectangle patch
        rect = patches.Rectangle((x1, y1), width, height, linewidth=2, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        
        # Add the class label
        label = f"{names[cls]}"
        ax.text(x1, y1, label, color='white', fontsize=12, bbox=dict(facecolor='red', alpha=0.5))

    ax.axis('off')  # Turn off axis

def load_model(model_path):
    return YOLO(model_path)  # Load the trained model

def get_image_paths(directory, extension_list=['.png', '.jpg', '.jpeg']):
    return [os.path.join(directory, fname) for fname in os.listdir(directory) if fname.lower().endswith(tuple(extension_list))]

def plot_results(image_paths, results, model, start_index=0, num_plots=8, save_path=None):
    fig, axs = plt.subplots(2, 4, figsize=(20, 10))

    for idx in range(num_plots):
        img_path = image_paths[start_index + idx]
        result = results[start_index + idx]
        boxes = result.boxes.xyxy
        scores = result.boxes.conf
        classes = result.boxes.cls
        plot_predictions(img_path, boxes, scores, classes, model.names, axs[idx // 4, idx % 4])

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    #plt.show()

def plot_ground_truths(image_paths, label_paths, model, start_index=0, num_plots=8, save_path=None):
    fig, axs = plt.subplots(2, 4, figsize=(20, 10))

    for idx in range(num_plots):
        img_path = image_paths[start_index + idx]
        label_path = label_paths[start_index + idx]
        plot_ground_truth(img_path, label_path, model.names, axs[idx // 4, idx % 4])

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    #plt.show()

def main():
    model_path = "./runs/detect/train/weights/best.pt"
    test_imgs_dir = '../data-collection/Japan_filtered/filtered/val/images'
    test_labels_dir = '../data-collection/Japan_filtered/filtered/val/labels'
    
    assert os.path.exists(test_imgs_dir), "Image path not found"
    assert os.path.exists(test_labels_dir), "Label path not found"

    model = load_model(model_path)
    
    with torch.no_grad():
        results = model.predict(source=test_imgs_dir, conf=0.2, iou=0.5)
    
    image_paths = get_image_paths(test_imgs_dir)
    label_paths = [os.path.join(test_labels_dir, os.path.splitext(os.path.basename(img_path))[0] + '.txt') for img_path in image_paths]
    
    # Start index for plotting
    start_index = 0
    
    # Plot and save prediction results
    plot_results(image_paths, results, model, start_index=start_index, num_plots=8, save_path='predictions.png')

    # Plot and save ground truth labels
    plot_ground_truths(image_paths, label_paths, model, start_index=start_index, num_plots=8, save_path='ground_truths.png')

if __name__ == '__main__':
    main()
