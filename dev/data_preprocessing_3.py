import os
import uuid
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torchvision import transforms
from PIL import Image
import shutil
import matplotlib.pyplot as plt

train_dir = Path('../Snow-Covered-Roads-Dataset-main/Snow-Covered-Roads-Dataset-main/dataset/train')
augmented_dir = train_dir.parent / 'augmented_train'
os.makedirs(augmented_dir, exist_ok=True)

class_folders = ['clear', 'light', 'medium', 'plowed']
for class_folder in class_folders:
    os.makedirs(augmented_dir / class_folder, exist_ok=True)

transform_augmented = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
    transforms.ToTensor()
])

def inverse_normalize(tensor, mean, std):
    mean = torch.tensor(mean).view(3, 1, 1)
    std = torch.tensor(std).view(3, 1, 1)
    return tensor * std + mean

class CustomImageDataset(Dataset):
    def __init__(self, image_dir, transform_augmented=None, num_augments=0):
        self.image_dir = Path(image_dir)
        self.transform_augmented = transform_augmented
        self.num_augments = num_augments

        self.image_paths = list(self.image_dir.glob('**/*.jpeg'))
        self.class_to_idx = {cls.name: idx for idx, cls in enumerate(self.image_dir.iterdir()) if cls.is_dir()}
        self.idx_to_class = {idx: cls for cls, idx in self.class_to_idx.items()}

        if not self.class_to_idx:
            self.class_to_idx = {self.image_dir.name: 0}
            self.idx_to_class = {0: self.image_dir.name}

    def __len__(self):
        return len(self.image_paths) * (self.num_augments + 1)

    def __getitem__(self, idx):
        orig_idx = idx // (self.num_augments + 1)
        aug_idx = idx % (self.num_augments + 1)

        img_path = self.image_paths[orig_idx]
        image = Image.open(img_path).convert('RGB')
        label = self.class_to_idx.get(img_path.parent.name, 0)

        if aug_idx != 0 and self.transform_augmented:
            image = self.transform_augmented(image)
            image = transforms.ToPILImage()(image)
            save_name = f"{uuid.uuid4()}.jpeg"
        else:
            image = Image.open(img_path).convert('RGB')
            save_name = img_path.name

        return image, label, save_name
    
original_dataset = CustomImageDataset(train_dir)
# Apply data augmentation only to minority classes
num_augments = 3  # number of augmentation
light_dataset = CustomImageDataset(train_dir / 'light', transform_augmented=transform_augmented, num_augments=num_augments)
plowed_dataset = CustomImageDataset(train_dir / 'plowed', transform_augmented=transform_augmented, num_augments=num_augments)

augmented_dataset = ConcatDataset([light_dataset, plowed_dataset])
original_loader = DataLoader(original_dataset, batch_size=32, shuffle=False, num_workers=2)
augmented_loader = DataLoader(augmented_dataset, batch_size=32, shuffle=True, num_workers=2)

def save_images(dataset, save_dir):
    total_images = 0
    for i in range(len(dataset)):
        img, label, save_name = dataset[i]
        class_name = dataset.idx_to_class[label]
        class_dir = save_dir / class_name
        os.makedirs(class_dir, exist_ok=True)
        save_path = class_dir / save_name
        img.save(save_path)
        total_images += 1
    print(f"Total images saved: {total_images}")

def save_original_images(original_dir, save_dir):
    total_images = 0
    for img_path in original_dir.glob('**/*.jpeg'):
        class_name = img_path.parent.name
        class_dir = save_dir / class_name
        os.makedirs(class_dir, exist_ok=True)
        save_path = class_dir / img_path.name
        shutil.copy(img_path, save_path)
        total_images += 1
    print(f"Total original images copied: {total_images}")

save_images(light_dataset, augmented_dir)
save_images(plowed_dataset, augmented_dir)

save_original_images(train_dir / 'clear', augmented_dir)
save_original_images(train_dir / 'medium', augmented_dir)
