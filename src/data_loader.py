import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os
from collections import Counter

def get_dataloaders(config):
    
    transform = transforms.Compose([
        transforms.Resize((config['data']['image_size'], config['data']['image_size'])),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    dataset = datasets.ImageFolder(config['data']['data_dir'], transform=transform)
    
    # Count total images and images per class
    print("=" * 50)
    print(" DATASET STATISTICS")
    print("=" * 50)
    print(f"Total images in dataset: {len(dataset)}")
    print(f"Number of classes: {len(dataset.classes)}")
    print(f"Class names: {dataset.classes}")
    print()
    
    # Count images per class
    class_counts = Counter()
    for _, label in dataset:
        class_counts[label] += 1
    
    print("Images per class:")
    for class_idx, class_name in enumerate(dataset.classes):
        count = class_counts[class_idx]
        print(f"   {class_name}: {count} images")
    
    print()
    print("Summary:")
    print(f"   Total images: {len(dataset)}")
    print(f"   Classes: {len(dataset.classes)}")
    print(f"   Average images per class: {len(dataset) / len(dataset.classes):.1f}")
    print("=" * 50)
    
    # Original data splitting
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=config['data']['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['data']['batch_size'], shuffle=False)
    
    print(f" Training set: {len(train_dataset)} images")
    print(f"📚 Validation set: {len(val_dataset)} images")
    print("=" * 50)

    return train_loader, val_loader
