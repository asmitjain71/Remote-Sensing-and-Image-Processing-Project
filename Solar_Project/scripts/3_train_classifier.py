# scripts/3_train_classifier.py
import torch
import torch.nn as nn
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader

# CONFIG
DATA_DIR = '../data/training_roofs'
MODEL_SAVE_PATH = '../models/resnet_classifier.pth'

def train_classifier():
    # 1. Image Augmentation
    data_transforms = transforms.Compose([
        transforms.Resize((224, 224)), # ResNet expects 224x224
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # 2. Load Data
    dataset = datasets.ImageFolder(DATA_DIR, transform=data_transforms)
    loader = DataLoader(dataset, batch_size=4, shuffle=True)
    class_names = dataset.classes # ['flat', 'gable']
    print(f"Classes found: {class_names}")

    # 3. Load Pre-trained ResNet18
    try:
        model = models.resnet18(weights='IMAGENET1K_V1')
    except:
        # Fallback for older torchvision versions
        model = models.resnet18(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2) # Binary Classification
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    # 4. Train
    print("🧠 Training Roof Classifier...")
    model.train()
    for epoch in range(10): # 10 Epochs is usually enough for 100 images
        running_loss = 0.0
        for inputs, labels in loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch {epoch+1} Loss: {running_loss/len(loader):.4f}")

    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print("✅ Classifier Saved!")

if __name__ == "__main__":
    train_classifier()
