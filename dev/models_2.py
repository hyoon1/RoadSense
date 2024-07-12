import timm
import torch
import torch.nn as nn

# Define the RoadNet model
class RoadNet(nn.Module):
    def __init__(self):
        super(RoadNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1) # First convolutional layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0) # Max pooling layer
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1) # Second convolutional layer
        self.fc1 = nn.Linear(64 * 56 * 56 + 1, 512)  # Fully connected layer (includes meta data input)
        self.fc2 = nn.Linear(512, 4)  # Output layer (4 classes for severity levels)
        self.relu = nn.ReLU() # ReLu activation function
        self.dropout = nn.Dropout(p=0.5) # Dropout layer for regularization
        
    def forward(self, image, meta):
        x = self.pool(self.relu(self.conv1(image))) # Apply conv1, ReLU, and pooling
        x = self.pool(self.relu(self.conv2(x))) # Apply conv2, ReLU, and pooling
        x = x.view(-1, 64 * 56 * 56) # Flatten the tensor
        x = torch.cat((x, meta.unsqueeze(1)), dim=1)  # Concatenate image features with meta data
        x = self.dropout(self.relu(self.fc1(x))) # Apply fc1, ReLU, and dropout
        x = self.fc2(x) # Apply fc2
        return x
    
class EfficientNetModel(nn.Module):
    def __init__(self):
        super(EfficientNetModel, self).__init__()
        # Load the pretrained EfficientNet-B0 model
        self.efficientnet = timm.create_model('efficientnet_b0', pretrained=True)
        # Replace the classifier layer with a new fully connected layer of size 512
        self.efficientnet.classifier = nn.Linear(self.efficientnet.classifier.in_features, 512)
        self.fc1 = nn.Linear(512 + 1, 256)  # 512 + 1 (meta data input size)
        self.fc2 = nn.Linear(256, 4) # 4 classes for severity levels
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.5)
        
    def forward(self, image, meta):
        x = self.efficientnet(image)
        x = torch.cat((x, meta.unsqueeze(1)), dim=1)  # Concatenate image features with meta data
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

class ImprovedEfficientNetModel(nn.Module):
    def __init__(self):
        super(EfficientNetModel, self).__init__()
        # Load the pretrained EfficientNet-B0 model
        self.efficientnet = timm.create_model('efficientnet_b0', pretrained=True)
        # Replace the classifier layer with a new fully connected layer of size 512
        self.efficientnet.classifier = nn.Linear(self.efficientnet.classifier.in_features, 512)
        
        # Additional layers
        self.fc1 = nn.Linear(512 + 1, 256)  # 512 + 1 (meta data input size)
        self.fc2 = nn.Linear(256, 128)  # Additional layer
        self.fc3 = nn.Linear(128, 4) # Final layer for 4 classes
        
        # Activation function and dropout layer
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.5)
        
    def forward(self, image, meta):
        # Forward pass through the EfficientNet model
        x = self.efficientnet(image)
        # Concatenate image features with meta data
        x = torch.cat((x, meta.unsqueeze(1)), dim=1)
        # Forward pass through additional layers with ReLU activation and dropout
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.dropout(self.relu(self.fc2(x)))  # Additional layer
        x = self.fc3(x)
        return x
    
# Define the ViT model
class ViTModel(nn.Module):
    def __init__(self):
        super(ViTModel, self).__init__()
        # Load a pre-trained Vision Transformer (ViT) model from the timm library
        self.vit = timm.create_model('vit_base_patch16_224', pretrained=True)
        # Replace the head of the ViT model with a Linear layer for feature extraction
        self.vit.head = nn.Linear(self.vit.head.in_features, 512)
        # Define a fully connected layer to combine image features with meta data
        self.fc1 = nn.Linear(512 + 1, 256)
        # Define the output layer for classifying severity levels (4 classes)
        self.fc2 = nn.Linear(256, 4)
        # Define a ReLU activation function
        self.relu = nn.ReLU()
        # Define a dropout layer for regularization
        self.dropout = nn.Dropout(p=0.5)
        
    def forward(self, image, meta):
        # Forward pass through the ViT model to extract image features
        x = self.vit(image)
        # Concatenate the image features with the meta data
        x = torch.cat((x, meta.unsqueeze(1)), dim=1)
        # Forward pass through the fully connected layer with ReLU activation and dropout
        x = self.dropout(self.relu(self.fc1(x)))
        # Forward pass through the output layer to get the class scores
        x = self.fc2(x)
        return x
    
class ImprovedViTModel(nn.Module):
    def __init__(self):
        super(ImprovedViTModel, self).__init__()
        # Load the pretrained ViT model
        self.vit = timm.create_model('vit_base_patch16_224', pretrained=True)
        self.vit.head = nn.Linear(self.vit.head.in_features, 512)
        
        # Additional layers
        self.fc1 = nn.Linear(512 + 1, 256)  # 512 + 1 (meta data input size)
        self.batch_norm1 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 128)
        self.batch_norm2 = nn.BatchNorm1d(128)
        self.fc3 = nn.Linear(128, 4)  # 4 classes for severity levels
        
        # Activation function and dropout
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, image, meta):
        # Extract image features using the ViT model
        x = self.vit(image)
        
        # Concatenate image features with meta data
        x = torch.cat((x, meta.unsqueeze(1)), dim=1)
        
        # Process through additional layers
        x = self.relu(self.batch_norm1(self.fc1(x)))
        x = self.dropout(x)
        x = self.relu(self.batch_norm2(self.fc2(x)))
        x = self.dropout(x)
        x = self.fc3(x)
        
        return x