import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import streamlit as st
from PIL import Image

class model_vgg(nn.Module):
    def __init__(self):
        super(model_vgg, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1), 
            nn.ReLU(inplace=True), 
            nn.Conv2d(256, 256, kernel_size=3, padding=1), 
            nn.ReLU(inplace=True), 
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(256, 512, kernel_size=3, padding=1), 
            nn.ReLU(inplace=True), 
            nn.Conv2d(512, 512, kernel_size=3, padding=1), 
            nn.ReLU(inplace=True), 
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(512, 512, kernel_size=3, padding=1), 
            nn.ReLU(inplace=True), 
            nn.Conv2d(512, 512, kernel_size=3, padding=1), 
            nn.ReLU(inplace=True), 
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.vgg_class = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 36),
        )

    def forward(self, x):
        x = self.layers(x)
        x = x.view(x.size(0), -1)
        x = self.vgg_class(x)
        return x

# Initialize and load the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model_vgg().to(device)
model.load_state_dict(torch.load("cnn_model_weights.pth", map_location=device))  # Ensure compatibility with CPU/GPU
model.eval()  # Set the model to evaluation mode

# Define transformations
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),  # Convert to grayscale
    transforms.Resize((224, 224)),  # Resize to match the training dimensions
    transforms.ToTensor(),  # Convert to tensor
    transforms.Normalize((0.5,), (0.5,))  # Normalize with mean=0.5, std=0.5
])

# Streamlit interface
st.title("VGG-Based Image Classifier")
uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Load and display the image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Preprocess the image
    input_tensor = transform(image).unsqueeze(0).to(device)  # Move tensor to the correct device

    # Perform inference
    with torch.no_grad():
        output = model(input_tensor)
        predicted_class = output.argmax(dim=1).item()

    # Display the prediction
    st.write(f"Predicted Class: {predicted_class}")
