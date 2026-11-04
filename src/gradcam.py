import os
import torch
import torch.nn.functional as F
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image

from src.config import DEVICE, IMAGES_DIR, LABELS, NUM_CLASSES
from src.dataset import get_transforms
from src.models.resnet import get_resnet50_model

class GradCAM:
    """Implementacao de Grad-CAM para interpretabilidade de redes convolucionais."""
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Hooks para capturar ativacoes e gradientes
        self.target_layer.register_forward_hook(self._save_activations)
        self.target_layer.register_full_backward_hook(self._save_gradients)

    def _save_activations(self, module, input, output):
        self.activations = output

    def _save_gradients(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate_heatmap(self, input_tensor, target_class_idx):
        self.model.zero_grad()
        output = self.model(input_tensor)
        
        # Seleciona a pontuacao especifica da classe alvo
        score = output[0, target_class_idx]
        score.backward(retain_graph=True)
        
        # Processamento dos gradientes e ativacoes
        gradients = self.gradients[0].cpu().data.numpy()
        activations = self.activations[0].cpu().data.numpy()
        
        # Pesos baseados na media dos gradientes (Global Average Pooling)
        weights = np.mean(gradients, axis=(1, 2))
        
        # Combinacao ponderada dos mapas de ativacao
        cam = np.zeros(activations.shape[1:], dtype=np.float32)
        for i, w in enumerate(weights):
            cam += w * activations[i, :, :]
            
        # Aplicacao de ReLU para manter apenas influencias positivas
        cam = np.maximum(cam, 0)
        
        # Normalizacao entre 0 e 1
        if cam.max() != 0:
            cam = cam / cam.max()
            
        return cam, torch.sigmoid(output[0, target_class_idx]).item()

def sobrepor_heatmap(imagem_np, heatmap, alpha=0.4):
    """Redimensiona o heatmap e faz a sobreposicao na imagem original."""
    heatmap_resized = cv2.resize(heatmap, (imagem_np.shape[1], imagem_np.shape[0]))
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
    
    # Mistura da imagem original em tons de cinza com o heatmap colorido
    overlay = np.float32(heatmap_color) * alpha + np.float32(imagem_np) * (1 - alpha)
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)
    return overlay

def gerar_gradcam_todas_patologias(model, image_path, labels, output_path):
    """Gera um painel comparativo contendo o Grad-CAM para cada uma das 14 patologias."""
    _, val_tf = get_transforms()
    
    # Carregamento e pre-processamento da imagem
    pil_image = Image.open(image_path).convert("RGB")
    imagem_np = np.array(pil_image)
    
    input_tensor = val_tf(pil_image).unsqueeze(0).to(DEVICE)
    
    # Instancia o Grad-CAM na ultima camada convolucional da ResNet-50
    target_layer = model.layer4[-1]
    grad_cam = GradCAM(model, target_layer)
    
    # Criacao da figura com matriz 3x5 (1 original + 14 patologias)
    fig, axes = plt.subplots(3, 5, figsize=(20, 12))
    axes = axes.ravel()
    
    # Subplot 0: Imagem Original
    axes[0].imshow(imagem_np)
    axes[0].set_title("Imagem Original", fontsize=10, fontweight="bold")
    axes[0].axis("off")
    
    # Subplots 1 a 14: Grad-CAM por patologia
    for idx, label_name in enumerate(labels):
        cam, prob = grad_cam.generate_heatmap(input_tensor, idx)
        overlay = sobrepor_heatmap(imagem_np, cam)
        
        ax_idx = idx + 1
        axes[ax_idx].imshow(overlay)
        axes[ax_idx].set_title(f"{label_name}\nProb: {prob:.1%}", fontsize=9)
        axes[ax_idx].axis("off")
        
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

def main():
    print("Iniciando geracao de mapas de calor individuais por patologia...")
    
    # Carrega o modelo com os melhores pesos treinados
    model = get_resnet50_model(num_classes=NUM_CLASSES, pretrained=False).to(DEVICE)
    checkpoint_path = "checkpoints/best_resnet50.pth"
    
    if not os.path.exists(checkpoint_path):
        print(f"Erro: Checkpoint nao encontrado em {checkpoint_path}")
        return
        
    model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))
    model.eval()
    
    # Exemplo: Seleciona as primeiras imagens do conjunto de teste
    test_df_path = "data/processed/test.csv"
    if os.path.exists(test_df_path):
        import pandas as pd
        test_df = pd.read_csv(test_df_path)
        
        # Processa as 3 primeiras imagens de teste como amostra
        for i in range(min(3, len(test_df))):
            img_name = test_df.iloc[i]["Image Index"]
            img_path = os.path.join(IMAGES_DIR, img_name)
            output_file = f"results/gradcam/pathologies_{img_name}"
            
            if os.path.exists(img_path):
                gerar_gradcam_todas_patologias(model, img_path, LABELS, output_file)
                print(f"Mapa de calor salvo em: {output_file}")
            else:
                print(f"Imagem nao encontrada: {img_path}")

if __name__ == "__main__":
    main()