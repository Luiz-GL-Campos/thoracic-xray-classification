import os
import cv2
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

from src.config import DEVICE, IMAGES_DIR, LABELS, NUM_CLASSES
from src.dataset import get_transforms
from src.models.resnet import get_resnet50_model

class GradCAM:
    """
    Classe para extração de mapas de ativação por gradiente (Grad-CAM)
    na última camada convolucional da ResNet-50.
    """
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Registra hooks de leitura de ativações e gradientes
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate_heatmap(self, input_tensor, class_idx):
        self.model.eval()
        self.model.zero_grad()
        
        # Realiza a passagem direta
        output = self.model(input_tensor)
        
        # Seleciona a saída do neurônio correspondente à patologia alvo
        target_score = output[0, class_idx]
        target_score.backward()

        # Extrai gradientes e ativações em formato Numpy
        gradients = self.gradients[0].cpu().data.numpy()   # (C, H, W)
        activations = self.activations[0].cpu().data.numpy() # (C, H, W)
        
        # Média global dos gradientes por canal (pesos da importância dos mapas)
        weights = np.mean(gradients, axis=(1, 2))
        
        # Combinação ponderada dos mapas de ativação
        cam = np.zeros(activations.shape[1:], dtype=np.float32)
        for i, w in enumerate(weights):
            cam += w * activations[i]

        # Aplica ativação ReLU para manter apenas contribuições positivas
        cam = np.maximum(cam, 0)
        
        # Normalização entre 0 e 1
        if cam.max() != 0:
            cam = cam / cam.max()
            
        probabilidade = torch.sigmoid(output)[0, class_idx].item()
        return cam, probabilidade


def gerar_overlay_gradcam(original_pil_img, cam, alpha=0.5):
    """Mapeia o Grad-CAM para o tamanho original e sobrepõe como mapa de calor."""
    w, h = original_pil_img.size
    cam_resized = cv2.resize(cam, (w, h))
    
    # Aplica o mapa de cores JET (Azul = Baixo, Vermelho = Alto)
    heatmap = np.uint8(255 * cam_resized)
    heatmap_colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    
    # Prepara a imagem original em RGB
    img_np = np.array(original_pil_img.convert("RGB"))
    
    # Fusão ponderada da imagem com o mapa de calor
    overlay = np.uint8((1 - alpha) * img_np + alpha * heatmap_colored)
    return overlay


def main():
    os.makedirs("results", exist_ok=True)
    
    # 1. Carrega modelo e pesos do checkpoint
    print("Carregando o modelo ResNet-50...")
    model = get_resnet50_model(num_classes=NUM_CLASSES, pretrained=False).to(DEVICE)
    model.load_state_dict(torch.load("checkpoints/best_resnet50.pth", map_location=DEVICE))
    
    # A última camada convolucional da ResNet-50 é model.layer4[-1]
    target_layer = model.layer4[-1]
    grad_cam = GradCAM(model, target_layer)
    
    # 2. Carrega o conjunto de teste e transformações
    test_df = pd.read_csv("data/processed/test.csv")
    _, val_tf = get_transforms()
    
    # Seleciona até 4 exemplos marcantes de patologias para demonstrar
    patologias_alvo = ["Cardiomegaly", "Effusion", "Pneumothorax", "Edema"]
    amostras_selecionadas = []
    
    for patologia in patologias_alvo:
        df_patologia = test_df[test_df[patologia] == 1]
        if not df_patologia.empty:
            amostras_selecionadas.append((patologia, df_patologia.iloc[0]))

    if not amostras_selecionadas:
        print("Nenhuma amostra específica encontrada. Selecionando as primeiras imagens do teste...")
        for i in range(min(4, len(test_df))):
            amostras_selecionadas.append((LABELS[0], test_df.iloc[i]))

    # 3. Processa e gera o painel comparativo
    fig, axes = plt.subplots(len(amostras_selecionadas), 2, figsize=(10, 4 * len(amostras_selecionadas)))
    if len(amostras_selecionadas) == 1:
        axes = np.expand_dims(axes, axis=0)

    print("Gerando mapas de calor Grad-CAM...")
    for idx, (patologia, row) in enumerate(amostras_selecionadas):
        img_name = row['Image Index']
        img_path = os.path.join(IMAGES_DIR, img_name)
        
        original_img = Image.open(img_path)
        input_tensor = val_tf(original_img.convert("RGB")).unsqueeze(0).to(DEVICE)
        
        class_idx = LABELS.index(patologia)
        cam, prob = grad_cam.generate_heatmap(input_tensor, class_idx)
        
        overlay = gerar_overlay_gradcam(original_img, cam)
        
        # Plota Raio-X Original
        axes[idx, 0].imshow(original_img, cmap='gray')
        axes[idx, 0].set_title(f"Original: {img_name}\nDoença Real: {patologia}", fontsize=10)
        axes[idx, 0].axis('off')
        
        # Plota Grad-CAM
        axes[idx, 1].imshow(overlay)
        axes[idx, 1].set_title(f"Grad-CAM ({patologia})\nProbabilidade Predita: {prob*100:.1f}%", fontsize=10, fontweight='bold')
        axes[idx, 1].axis('off')

    plt.suptitle("Explicabilidade Visual com Grad-CAM (ResNet-50)", fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()
    
    output_path = "results/gradcam_exemplos.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Painel do Grad-CAM salvo com sucesso em: {output_path}")

if __name__ == "__main__":
    main()