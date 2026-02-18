# scripts/2_train_detector.py
import os
import torch
import cv2
import numpy as np
import segmentation_models_pytorch as smp
from torch.utils.data import Dataset, DataLoader

# CONFIG
TRAIN_IMG_DIR = '../data/training_sliced/images/'
TRAIN_MASK_DIR = '../data/training_sliced/masks/'
MODEL_PATH = '../models/unet_detector.pth'
EPOCHS = 5

class BuildingDataset(Dataset):
    def __init__(self, img_dir, mask_dir):
        self.images = os.listdir(img_dir)
        self.img_dir = img_dir
        self.mask_dir = mask_dir
    
    def __len__(self): return len(self.images)
    
    def __getitem__(self, i):
        img_name = self.images[i]
        
        # Load
        img = cv2.imread(self.img_dir + img_name)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(self.mask_dir + img_name, 0)
        
        # Norm & Tensor
        img = img.transpose(2,0,1).astype('float32') / 255.0
        mask = np.expand_dims(mask, 0).astype('float32') / 255.0 # (1, H, W)
        
        return torch.from_numpy(img), torch.from_numpy(mask)

def train():
    dataset = BuildingDataset(TRAIN_IMG_DIR, TRAIN_MASK_DIR)
    loader = DataLoader(dataset, batch_size=8, shuffle=True)
    
    model = smp.Unet(encoder_name="resnet18", encoder_weights="imagenet", in_channels=3, classes=1)
    loss_fn = smp.losses.DiceLoss(mode='binary')
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    
    print("🚀 Starting Training...")
    model.train()
    
    for epoch in range(EPOCHS):
        total_loss = 0
        for imgs, masks in loader:
            optimizer.zero_grad()
            output = model(imgs)
            loss = loss_fn(output, masks)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {total_loss/len(loader):.4f}")
        
    torch.save(model.state_dict(), MODEL_PATH)
    print("✅ Model Saved!")

if __name__ == "__main__":
    train()
