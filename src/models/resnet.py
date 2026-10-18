import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights

def get_resnet50_model(num_classes=14, pretrained=True):
    """
    Carrega a ResNet-50 pré-treinada no ImageNet e ajusta a camada final.
    """
    weights = ResNet50_Weights.DEFAULT if pretrained else None
    model = resnet50(weights=weights)
    
    # Substitui a camada linear final para prever 14 saídas independentes
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    
    return model