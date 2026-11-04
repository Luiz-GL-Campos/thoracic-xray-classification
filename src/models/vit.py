import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import ViT_B_16_Weights

def get_vit_b_16_model(num_classes=14, pretrained=True):
    """
    Retorna a arquitetura Vision Transformer (ViT-B/16) adaptada para classificacao multirrotulo.
    """
    if pretrained:
        weights = ViT_B_16_Weights.DEFAULT
        model = models.vit_b_16(weights=weights)
    else:
        model = models.vit_b_16(weights=None)
    
    # Substitui a cabeca final de classificacao (heads.head)
    in_features = model.heads.head.in_features
    model.heads.head = nn.Linear(in_features, num_classes)
    
    return model