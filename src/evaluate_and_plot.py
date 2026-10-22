import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from torch.utils.data import DataLoader

from src.config import DEVICE, IMAGES_DIR, LABELS, NUM_CLASSES
from src.dataset import NIHChestXrayDataset, get_transforms
from src.models.resnet import get_resnet50_model

@torch.no_grad()
def gerar_grafico_roc():
    # 1. Carrega dados e modelo
    test_df = pd.read_csv("data/processed/test.csv")
    _, val_tf = get_transforms()
    test_dataset = NIHChestXrayDataset(test_df, IMAGES_DIR, transform=val_tf, labels=LABELS)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)
    
    model = get_resnet50_model(num_classes=NUM_CLASSES, pretrained=False).to(DEVICE)
    model.load_state_dict(torch.load("checkpoints/best_resnet50.pth", map_location=DEVICE))
    model.eval()
    
    # 2. Coleta predições do conjunto de teste
    all_targets, all_preds = [], []
    for images, targets in test_loader:
        images = images.to(DEVICE)
        outputs = model(images)
        preds = torch.sigmoid(outputs)
        
        all_targets.append(targets.numpy())
        all_preds.append(preds.cpu().numpy())
        
    y_true = np.vstack(all_targets)
    y_pred = np.vstack(all_preds)
    
    # 3. Plota as Curvas ROC para cada uma das 14 patologias
    plt.figure(figsize=(12, 8))
    os.makedirs("results", exist_ok=True)
    
    for i, label in enumerate(LABELS):
        fpr, tpr, _ = roc_curve(y_true[:, i], y_pred[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'{label} (AUROC = {roc_auc:.3f})')
        
    plt.plot([0, 1], [0, 1], 'k--', label='Classificador Aleatório')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Taxa de Falsos Positivos (1 - Especificidade)')
    plt.ylabel('Taxa de Verdadeiros Positivos (Sensibilidade)')
    plt.title('Curvas ROC por Patologia - ResNet-50 (NIH Chest X-ray)')
    plt.legend(loc="lower right", fontsize=8)
    plt.grid(True, alpha=0.3)
    
    # Salva o gráfico
    output_path = "results/curvas_roc.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Gráficos das Curvas ROC salvos em: {output_path}")

if __name__ == "__main__":
    gerar_grafico_roc()