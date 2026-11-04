import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, confusion_matrix, classification_report
from torch.utils.data import DataLoader

from src.config import DEVICE, IMAGES_DIR, LABELS, NUM_CLASSES, MODEL_NAME, CHECKPOINT_DIR, RESULTS_DIR
from src.dataset import NIHChestXrayDataset, get_transforms
from src.models import build_model

@torch.no_grad()
def obter_predicoes_e_rotulos(model, dataloader, device):
    """Realiza inferencia no conjunto de dados e retorna rotulos reais e probabilidades."""
    model.eval()
    all_targets = []
    all_preds = []
    
    for images, targets in dataloader:
        images = images.to(device, non_blocking=True)
        with torch.amp.autocast('cuda'):
            outputs = model(images)
            preds = torch.sigmoid(outputs)
            
        all_targets.append(targets.numpy())
        all_preds.append(preds.cpu().numpy())
        
    y_true = np.vstack(all_targets)
    y_pred = np.vstack(all_preds)
    return y_true, y_pred

def plotar_curvas_roc(y_true, y_pred, labels, output_path):
    """Gera e salva o grafico contendo as Curvas ROC para as 14 patologias."""
    plt.figure(figsize=(12, 10))
    
    for i, label in enumerate(labels):
        fpr, tpr, _ = roc_curve(y_true[:, i], y_pred[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, label=f"{label} (AUC = {roc_auc:.3f})")
        
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("Taxa de Falsos Positivos (FPR)", fontsize=12)
    plt.ylabel("Taxa de Verdadeiros Positivos (TPR)", fontsize=12)
    plt.title(f"Curvas ROC por Patologia - {MODEL_NAME.upper()}", fontsize=14, fontweight="bold")
    plt.legend(loc="lower right", fontsize=9)
    plt.grid(alpha=0.3)
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

def plotar_matrizes_confusao(y_true, y_pred, labels, output_path, limiar=0.5):
    """Gera matrizes de confusao binarias para cada patologia considerando um limiar."""
    y_pred_bin = (y_pred >= limiar).astype(int)
    
    fig, axes = plt.subplots(4, 4, figsize=(16, 16))
    axes = axes.ravel()
    
    for i, label in enumerate(labels):
        cm = confusion_matrix(y_true[:, i], y_pred_bin[:, i])
        
        ax = axes[i]
        ax.matshow(cm, cmap=plt.cm.Blues, alpha=0.7)
        
        for m in range(cm.shape[0]):
            for n in range(cm.shape[1]):
                ax.text(x=n, y=m, s=cm[m, n], va="center", ha="center", size=11, weight="bold")
                
        ax.set_title(label, fontsize=11, fontweight="bold")
        ax.set_xlabel("Predito", fontsize=9)
        ax.set_ylabel("Real", fontsize=9)
        ax.set_xticklabels(["", "Neg", "Pos"])
        ax.set_yticklabels(["", "Neg", "Pos"])
        
    # Desativa os dois ultimos subplots sobrantes (16 subplots para 14 classes)
    axes[14].axis("off")
    axes[15].axis("off")
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

def main():
    print(f"Iniciando avaliacao detalhada para o modelo: {MODEL_NAME}")
    
    # 1. Carrega dados de teste
    test_df_path = "data/processed/test.csv"
    if not os.path.exists(test_df_path):
        print(f"Erro: Arquivo {test_df_path} nao encontrado.")
        return
        
    test_df = pd.read_csv(test_df_path)
    _, val_tf = get_transforms()
    test_dataset = NIHChestXrayDataset(test_df, IMAGES_DIR, transform=val_tf, labels=LABELS)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=4, pin_memory=True)
    
    # 2. Carrega arquitetura dinamica e pesos salvos
    checkpoint_path = os.path.join(CHECKPOINT_DIR, f"best_{MODEL_NAME}.pth")
    if not os.path.exists(checkpoint_path):
        print(f"Erro: Checkpoint nao encontrado em {checkpoint_path}")
        return
        
    print(f"Carregando pesos de: {checkpoint_path}")
    model = build_model(model_name=MODEL_NAME, num_classes=NUM_CLASSES, pretrained=False).to(DEVICE)
    model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))
    
    # 3. Executa inferencia
    print("Gerando predicoes no conjunto de teste...")
    y_true, y_pred = obter_predicoes_e_rotulos(model, test_loader, DEVICE)
    
    # 4. Salva graficos e relatorios dentro do RESULTS_DIR correspondente
    roc_fig_path = os.path.join(RESULTS_DIR, "curvas_roc.png")
    cm_fig_path = os.path.join(RESULTS_DIR, "matrizes_confusao.png")
    csv_report_path = os.path.join(RESULTS_DIR, "relatorio_metricas.csv")
    
    print("Gerando graficos das Curvas ROC...")
    plotar_curvas_roc(y_true, y_pred, LABELS, roc_fig_path)
    
    print("Gerando Matrizes de Confusao...")
    plotar_matrizes_confusao(y_true, y_pred, LABELS, cm_fig_path)
    
    # 5. Salva relatorio de metricas em CSV
    metricas_list = []
    for i, label in enumerate(LABELS):
        fpr, tpr, _ = roc_curve(y_true[:, i], y_pred[:, i])
        roc_auc = auc(fpr, tpr)
        metricas_list.append({"Patologia": label, "AUROC": round(float(roc_auc), 4)})
        
    df_metricas = pd.DataFrame(metricas_list)
    df_metricas.to_csv(csv_report_path, index=False)
    
    print(f"Avaliacao concluida.")
    print(f"Curvas ROC salvas em: {roc_fig_path}")
    print(f"Matrizes de Confusao salvas em: {cm_fig_path}")
    print(f"Relatorio salvo em: {csv_report_path}")

if __name__ == "__main__":
    main()