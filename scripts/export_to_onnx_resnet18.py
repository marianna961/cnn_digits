import torch
from torchvision.models import resnet18

def export_to_onnx(model_path="digit_classifier_finetuned.pth", output_onnx="digit_classifier.onnx", num_classes=10, img_size=(32, 32)):
    """Экспортирует модель ResNet-18 в ONNX."""
    # Загрузка модели ResNet-18
    model = resnet18(num_classes=num_classes)
    # Изменяем входной слой для одного канала (монохромные изображения)
    model.conv1 = torch.nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    
    # Загружаем сохранённые веса
    state_dict = torch.load(model_path, map_location=torch.device('cpu'))
    model.load_state_dict(state_dict)
    model.eval()

    # Создание примера входных данных
    dummy_input = torch.randn(1, 1, img_size[0], img_size[1])

    # Экспорт в ONNX
    torch.onnx.export(
        model,
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
        model_path=r"C:\Users\maria\OneDrive\Documents\job\extract_data\40e_resnet18_classic.pth",
        output_onnx=r"C:\Users\maria\OneDrive\Documents\job\extract_data\40e_resnet18_classic.onnx",
        num_classes=10,  # Для цифр 0-9
        img_size=(32, 32)
    )