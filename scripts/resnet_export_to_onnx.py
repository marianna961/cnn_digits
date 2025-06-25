import torch
from torchvision import models
import torch.nn as nn # Добавлено для nn.Conv2d и nn.Linear

# Определение класса DigitClassifier с предобученной моделью
class DigitClassifier(torch.nn.Module):
    def __init__(self, num_classes, pretrained_model, img_size=(32, 32)):
        super(DigitClassifier, self).__init__()
        # Загружаем предобученную модель (ResNet18)
        self.base_model = pretrained_model
        # Заменяем первый слой для grayscale (1 канал)
        self.base_model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # Замораживаем все слои базы (хотя для экспорта это не обязательно)
        for param in self.base_model.parameters():
            param.requires_grad = False
        # Заменяем последний полносвязный слой
        num_features = self.base_model.fc.in_features
        self.base_model.fc = nn.Sequential( # Изменено на nn.Sequential для соответствия
            nn.Linear(num_features, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.base_model(x)

def export_to_onnx(model_path="digit_classifier_finetuned.pth", output_onnx="digit_classifier.onnx", num_classes=10, img_size=(32, 32)):
    """
    Экспортирует модель DigitClassifier в формат ONNX.

    Args:
        model_path (str): Путь к сохранённой модели (.pth).
        output_onnx (str): Путь для сохранения ONNX-модели.
        num_classes (int): Количество классов (10 для 0-9).
        img_size (tuple): Размер входных изображений.
    """
    # Создаем экземпляр базовой модели ResNet18
    # Важно: здесь мы создаем чистую ResNet18, а не предобученную с IMAGENET1K_V1,
    # так как веса будут загружены из вашей обученной модели.
    # Если вы хотите использовать предобученные веса ImageNet для инициализации,
    # то `train_resnet.py` также должен использовать их для своей модели.
    # Сейчас `train_resnet.py` использует `pretrained=False`, что эквивалентно `weights=None`.
    base_model = models.resnet18(weights=None) # Загружаем без предобученных весов

    # Модифицируем conv1 и fc слои базовой модели, как это было сделано при обучении
    base_model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    num_features = base_model.fc.in_features
    # Важно: ваш train_resnet.py устанавливает model.fc = nn.Linear(...).
    # Здесь мы также должны установить такой же слой для корректной загрузки state_dict.
    base_model.fc = nn.Linear(num_features, num_classes) 

    # Загружаем state_dict в эту модифицированную базовую модель ResNet18
    # Это загрузит веса из digit_classifier.pth, которые соответствуют ResNet18
    # с измененными conv1 и fc слоями.
    base_model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    
    # Теперь создаем ваш класс DigitClassifier, передавая ему уже загруженную и модифицированную base_model.
    # В DigitClassifier __init__ будут повторно изменены conv1 и fc слои.
    # Если DigitClassifier должен инкапсулировать модификацию, 
    # то загрузку state_dict нужно адаптировать.
    # Однако, судя по ошибке, state_dict содержит ключи напрямую от ResNet, 
    # а не от base_model внутри DigitClassifier.

    # Оптимальное решение:
    # Ваша модель DigitClassifier внутри себя создает ResNet18 и модифицирует ее.
    # Это означает, что state_dict, который вы сохранили из train_resnet.py, 
    # НЕ совпадает со state_dict, который ожидает DigitClassifier (потому что DigitClassifier
    # ожидает 'base_model.conv1.weight' и т.д.).
    # Вам нужно загружать state_dict непосредственно в базовую модель ResNet18, 
    # а не в DigitClassifier.

    # Пересоздаем модель так, как она была создана и сохранена в train_resnet.py
    # Это ResNet18 с измененными conv1 и fc слоями.
    model = models.resnet18(weights=None) # Без предобученных весов
    model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    model.fc = nn.Linear(model.fc.in_features, num_classes) # Должен совпадать с train_resnet.py

    # Загружаем state_dict в эту модель
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()

    # Создание примера входных данных
    dummy_input = torch.randn(1, 1, img_size[0], img_size[1])

    # Экспорт в ONNX
    torch.onnx.export(
        model, # Теперь экспортируем непосредственно модифицированную ResNet18
        dummy_input,
        output_onnx,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    )
    print(f"Модель экспортирована в {output_onnx}")

if __name__ == "__main__":
    export_to_onnx(
        model_path=r"C:\Users\maria\OneDrive\Documents\job\extract_data\40e_resnet18_canny.pth",
        output_onnx=r"C:\Users\maria\OneDrive\Documents\job\extract_data\40e_resnet18_canny.onnx",
        num_classes=10,
        img_size=(32, 32)
    )