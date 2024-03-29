import torch
import os
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import torchvision.models as models
from PIL import Image

# Define your CNN architecture
class BasicCNN(nn.Module):
    def __init__(self):
        super(BasicCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(32 * 56 * 56, 128)
        self.fc2 = nn.Linear(128, 2)  # 2 classes: infected or uninfected

    def forward(self, x):
        x = self.pool(nn.ReLU()(self.conv1(x)))
        x = self.pool(nn.ReLU()(self.conv2(x)))
        x = x.view(-1, 32 * 56 * 56)
        x = nn.ReLU()(self.fc1(x))
        x = self.fc2(x)
        return x

# Define transformations for data preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

infected_path = r"C:\Users\SHUBHAM RAJ\Downloads\Malaria_classification\cell_images\cell_images\Parasitized"
uninfected_path = r"C:\Users\SHUBHAM RAJ\Downloads\Malaria_classification\cell_images\cell_images\Uninfected"
# Function to load images from a directory
def load_images_from_directory(path):
    images = []
    for filename in os.listdir(path):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            img = Image.open(os.path.join(path, filename))
            img = transform(img)
            images.append(img)
    return images

# Load images from the infected and uninfected directories
infected_images = load_images_from_directory(infected_path)
uninfected_images = load_images_from_directory(uninfected_path)

# Combine the datasets
combined_images = infected_images + uninfected_images
# Create a DataLoader
batch_size = 32
data_loader = DataLoader(combined_images, batch_size=batch_size, shuffle=True)

# Initialize your CNN model
model = BasicCNN()

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Train your model
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for images, labels in data_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss / len(data_loader):.4f}")

# Save the trained model parameters
torch.save(model.state_dict(), "model.pth")
