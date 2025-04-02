import os
import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

def get_dataloaders(data_dir="data", batch_size=32, image_size=224):
    transform = transforms.Compose([
        transforms.Resize(image_size),  # Resize to fixed size
        transforms.CenterCrop(image_size),  # Crop to keep the aspect ratio consistent
        transforms.RandomHorizontalFlip(),  # Data Augmentation
        transforms.RandomRotation(10),  # Slight rotation for robustness
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_data = ImageFolder(os.path.join(data_dir, "train"), transform=transform)
    val_data = ImageFolder(os.path.join(data_dir, "val"), transform=transform)
    test_data = ImageFolder(os.path.join(data_dir, "test"), transform=transform)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader

if __name__ == "__main__":
    train_loader, val_loader, test_loader = get_dataloaders(image_size=256)
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}")