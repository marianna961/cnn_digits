"""
обучение модели ResNet18
"""
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms, models
from torch.cuda.amp import GradScaler, autocast
from collections import Counter

def create_model(num_classes, img_size=(32, 32), pretrained=False):
    model = models.resnet18(pretrained=pretrained)
    model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

def train_digit_classifier(dataset_folder="dataset_processed", img_size=(32, 32), batch_size=64, epochs=60, device=None):
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = "cpu"

    train_transforms = transforms.Compose([
        transforms.Resize(img_size),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.RandomRotation(15),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.RandomErasing(p=0.5, scale=(0.02, 0.2)),
        transforms.Normalize((0.5,), (0.5,))
    ])

    val_transforms = transforms.Compose([
        transforms.Resize(img_size),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    train_dataset = datasets.ImageFolder(os.path.join(dataset_folder, "train"), transform=train_transforms)
    val_dataset = datasets.ImageFolder(os.path.join(dataset_folder, "val"), transform=val_transforms)

    class_counts = Counter([label for _, label in train_dataset])
    class_weights = [1.0 / count for count in class_counts.values()]
    sample_weights = [class_weights[label] for _, label in train_dataset]
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    model = create_model(num_classes=len(train_dataset.classes), img_size=img_size).to(device)

# optimizer and scheduler
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    initial_lr = optimizer.param_groups[0]['lr']
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    scaler = GradScaler()

    # logging
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []

    # train
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            with autocast():
                outputs = model(inputs)
                loss = criterion(outputs, labels)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_loss = running_loss / len(train_loader)
        train_acc = 100 * correct / total
        train_losses.append(train_loss)
        train_accs.append(train_acc)

        # validation
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_loss = val_loss / len(val_loader)
        val_acc = 100 * correct / total
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        scheduler.step(val_loss)
    torch.save(model.state_dict(), "40e_resnet18_classic.pth")
    print("done")

if __name__ == "__main__":
    train_digit_classifier(
        dataset_folder=r"C:\Users\maria\OneDrive\Documents\job\extract_data\dataset",
        img_size=(32, 32),
        batch_size=64,
        epochs=40
    )