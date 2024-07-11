import torch.nn as nn
import timm

class EfficientNetModel(nn.Module):
    def __init__(self, num_classes):
        super(EfficientNetModel, self).__init__()
        # Load the pretrained EfficientNet-B0 model
        self.efficientnet = timm.create_model('efficientnet_b0', pretrained=True)
        # Replace the classifier layer with a new fully connected layer of size 512
        self.efficientnet.classifier = nn.Linear(self.efficientnet.classifier.in_features, 512)
        
        # Additional layers
        self.fc1 = nn.Linear(512, 256)  # Additional layer
        self.fc2 = nn.Linear(256, 128)  # Additional layer
        self.fc3 = nn.Linear(128, num_classes)  # Final layer for classification
        
        # Activation function and dropout layer
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.5)
        
    def forward(self, image):
        # Forward pass through the EfficientNet model
        x = self.efficientnet(image)
        # Forward pass through additional layers with ReLU activation and dropout
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.dropout(self.relu(self.fc2(x)))  # Additional layer
        x = self.fc3(x)
        return x
