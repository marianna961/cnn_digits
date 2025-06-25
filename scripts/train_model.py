import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.cuda.amp import GradScaler, autocast
import matplotlib.pyplot as plt

# Определение модели вне функции
class DigitClassifier(nn.Module):
    def __init__(self, num_classes, img_size=(32, 32)):
        super(DigitClassifier, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2, 2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64 * (img_size[0] // 4) * (img_size[1] // 4), 128)
        self.relu3 = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.flatten(x)
        x = self.relu3(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

def train_digit_classifier(dataset_folder="dataset_processed", img_size=(32, 32), batch_size=32, epochs=40, device=None):
    device = "cpu"
    # if device is None:
    #     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Трансформации для аугментации и нормализации
    train_transforms = transforms.Compose([
        transforms.Resize(img_size),
        transforms.Grayscale(num_output_channels=1),
        transforms.RandomRotation(10),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.RandomResizedCrop(img_size, scale=(0.9, 1.1)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    val_transforms = transforms.Compose([
        transforms.Resize(img_size),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # Загрузка датасетов
    train_dataset = datasets.ImageFolder(os.path.join(dataset_folder, "train"), transform=train_transforms)
    val_dataset = datasets.ImageFolder(os.path.join(dataset_folder, "val"), transform=val_transforms)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    print(f"Найдено {len(train_dataset)} тренировочных изображений, {len(val_dataset)} валидационных")
    print(f"Классы: {train_dataset.class_to_idx}")

    # Использование вынесенного класса
    model = DigitClassifier(num_classes=len(train_dataset.classes), img_size=img_size).to(device)
    print(model)

    # Оптимизатор и функция потерь
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scaler = GradScaler()

    # Логирование
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []

    # # Ранняя остановка
    # best_val_loss = float('inf')
    # patience = 5
    # trigger_times = 0

    # Обучение
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

        # Валидация
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

        print(f"Эпоха {epoch+1}/{epochs}, Тренировка: Loss={train_loss:.4f}, Accuracy={train_acc:.2f}%")
        print(f"Валидация: Loss={val_loss:.4f}, Accuracy={val_acc:.2f}%")

        # Ранняя остановка
        # if val_loss < best_val_loss:
        # best_val_loss = val_loss
            # trigger_times = 0
        torch.save(model.state_dict(), "digit_classifier_best.pth")
        # else:
        #     trigger_times += 1
        #     if trigger_times >= patience:
        #         print("Ранняя остановка!")
        #         break

    # Построение графиков
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Train Accuracy')
    plt.plot(val_accs, label='Val Accuracy')
    plt.legend()
    plt.show()

    # Сохранение модели
    torch.save(model.state_dict(), "20e_manual_classic.pth")
    print(f"Модель обучена и сохранена в '20e_manual_classic.pth'")

# Запуск
if __name__ == "__main__":
    train_digit_classifier(
        dataset_folder=r"C:\Users\maria\OneDrive\Documents\job\extract_data\dataset",
        img_size=(32, 32),
        batch_size=32,
        epochs=30
    )