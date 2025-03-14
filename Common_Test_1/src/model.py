import torch.nn as nn
import torchvision.models as models

def get_model(model_name, num_classes):
    if model_name == "resnet10":
        model = models.resnet18(pretrained=True)  # No official ResNet10, using ResNet18
    elif model_name == "resnet34":
        model = models.resnet34(pretrained=True)
    elif model_name == "resnet50":
        model = models.resnet50(pretrained=True)
    elif model_name == "resnet152":
        model = models.resnet152(pretrained=True)
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model
