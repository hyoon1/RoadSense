import cv2
import torch
from torchvision import transforms
from torch.cuda.amp import autocast
from ultralytics import YOLO
import torch.nn as nn
import timm


POTHOLE_CLS = 3

# Initialize device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ImprovedViTModel(nn.Module):
    def __init__(self):
        super(ImprovedViTModel, self).__init__()
        # Load the pretrained ViT model
        self.vit = timm.create_model("vit_base_patch16_224", pretrained=True)
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


# Instantiate the model
vit_model_directory = "./models/best_vit_model_2.pth"
vit_model = ImprovedViTModel()
vit_model.load_state_dict(
    torch.load(vit_model_directory, map_location=torch.device("cpu"))
)
vit_model.to(device).half().eval()

# Load models

# vit_model = torch.jit.script(vit_model).to(device).half().eval()

# Define the transformation pipeline
transform = transforms.Compose(
    [
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


def process_frame(frame, model):
    # Perform YOLO inference
    # yolo_model = YOLO("./models/yolov8n_best.pt").to(device).half()
    with autocast():
        results = model.track(frame, persist=True, half=True)  # Enable tracking

    # Visualize the results on the frame
    annotated_frame = results[0].plot()
    
    js = results[0]

    # Count potholes (assuming class 0 corresponds to potholes)
    count = sum(1 for box in results[0].boxes if box.cls.item() == POTHOLE_CLS)
    # print(f"Pothole count: {count}")

    # Prepare the image for ViT model
    vit_input = transform(frame).unsqueeze(0).to(device).half()

    # Prepare metadata
    meta = torch.tensor([count], dtype=torch.float16).to(device)

    severity_level = '0'
    # Inference with ViT model
    with torch.no_grad(), autocast():
        if count > 0:
            severity = vit_model(vit_input, meta)
            # Process severity output
            severity_level = str(torch.argmax(severity, dim=1).item())
            print(f"Pothole detected with severity level: {severity_level}")
        else:
            print("Skipping classify severity.")

    return annotated_frame, js, severity_level


def detect_video(video_source):
    # Capture video stream
    cap = cv2.VideoCapture(video_source)

    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret or frame_count == 50:
            break

        processed_frame = process_frame(frame)
        cv2.imshow("Frame", processed_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        frame_count += 1
    cap.release()
    cv2.destroyAllWindows()


# if __name__ == "__main__":

#     main("../../road-video-yellow-solid.mp4")  # Use the appropriate video source
