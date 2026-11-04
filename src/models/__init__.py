from src.models.resnet import get_resnet50_model
from src.models.vit import get_vit_b_16_model

def build_model(model_name="resnet50", num_classes=14, pretrained=True):
    """
    Fabrica de modelos para instanciar a arquitetura desejada no experimento.
    """
    model_name = model_name.lower()
    if model_name == "resnet50":
        return get_resnet50_model(num_classes=num_classes, pretrained=pretrained)
    elif model_name in ["vit", "vit_b_16", "vit-b-16"]:
        return get_vit_b_16_model(num_classes=num_classes, pretrained=pretrained)
    else:
        raise ValueError(f"Modelo desconhecido: {model_name}")