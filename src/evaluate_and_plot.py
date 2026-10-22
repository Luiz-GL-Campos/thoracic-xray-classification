import os
import json
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, multilabel_confusion_matrix, roc_auc_score
from torch.utils.data import DataLoader

from src.config import DEVICE, IMAGES_DIR, LABELS, NUM_CLASSES
from src.dataset import NIHChestXrayDataset, get_transforms
from src.models.resnet import get_resnet50_model

@torch.no_grad()
def obter_predicoes_e_rotulos():
    """Carrega o modelo treinado e extrai rótulos reais e probabilidades do teste."""
    test_df = pd.read_csv("data/processed/test.csv")
    _, val_tf = get_transforms()
    test_dataset = NIHChestXrayDataset(test_df, IMAGES_DIR, transform=val_tf, labels=LABELS)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)
    
    model = get_resnet50_model(num_classes=NUM_CLASSES, pretrained=False).to(DEVICE)
    model.load_state_dict(torch.load("checkpoints/best_resnet50.pth", map_location=DEVICE))
    model.eval()
    
    all_targets, all_preds = [], []
    for images, targets in test_loader:
        images = images.to(DEVICE)
        outputs = model(images)
        preds = torch.sigmoid(outputs)
        
        all_targets.append(targets.numpy())
        all_preds.append(preds.cpu().numpy())
        
    y_true = np.vstack(all_targets)
    y_pred = np.vstack(all_preds)
    
    return y_true, y_pred

def gerar_grafico_roc(y_true, y_pred):
    """Gera e salva o gráfico das Curvas ROC para todas as 14 patologias."""
    plt.figure(figsize=(12, 8))
    
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
    
    output_path = "results/curvas_roc.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Curvas ROC salvas em: {output_path}")

def gerar_matrizes_confusao(y_true, y_pred, threshold=0.5):
    """Calcula e salva a imagem e o JSON das matrizes de confusão."""
    y_pred_binary = (y_pred >= threshold).astype(int)
    mcm = multilabel_confusion_matrix(y_true, y_pred_binary)
    
    fig, axes = plt.subplots(4, 4, figsize=(16, 16))
    axes = axes.ravel()
    dados_matrizes_json = {}
    
    for i, label in enumerate(LABELS):
        tn, fp, fn, tp = mcm[i].ravel()
        matrix_2x2 = np.array([[tn, fp], [fn, tp]])
        
        sns.heatmap(
            matrix_2x2, 
            annot=True, 
            fmt='d', 
            cmap='Blues', 
            cbar=False,
            ax=axes[i],
            xticklabels=['Negativo', 'Positivo'],
            yticklabels=['Negativo', 'Positivo']
        )
        axes[i].set_title(f'{label}', fontsize=12, fontweight='bold')
        axes[i].set_xlabel('Predito')
        axes[i].set_ylabel('Real')
        
        dados_matrizes_json[label] = {
            "True_Negatives_VN": int(tn),
            "False_Positives_FP": int(fp),
            "False_Negatives_FN": int(fn),
            "True_Positives_VP": int(tp)
        }

    for j in range(len(LABELS), 16):
        fig.delaxes(axes[j])
        
    plt.suptitle(f'Matrizes de Confusão por Patologia (Limiar = {threshold})', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    img_output = "results/matrizes_confusao.png"
    plt.savefig(img_output, dpi=300, bbox_inches='tight')
    plt.close()
    
    json_output = "results/confusion_matrices.json"
    with open(json_output, "w") as f:
        json.dump(dados_matrizes_json, f, indent=4)
        
    print(f"✅ Matrizes de Confusão salvas em: {img_output} e {json_output}")
    return mcm

def calcular_metricas_detalhadas(y_true, y_pred, mcm, threshold=0.5):
    """
    Calcula AUROC, Acurácia, Sensibilidade, Especificidade, Precisão e F1-Score
    para cada patologia e salva em CSV e JSON.
    """
    relatorio = []
    
    for i, label in enumerate(LABELS):
        tn, fp, fn, tp = mcm[i].ravel()
        
        # AUROC
        try:
            auroc = roc_auc_score(y_true[:, i], y_pred[:, i])
        except ValueError:
            auroc = np.nan

        # Métricas calculadas com tratamento contra divisão por zero
        total = tn + fp + fn + tp
        acuracia = (tp + tn) / total if total > 0 else 0.0
        sensibilidade = tp / (tp + fn) if (tp + fn) > 0 else 0.0  # Recall / TPR
        especificidade = tn / (tn + fp) if (tn + fp) > 0 else 0.0 # TNR
        precisao = tp / (tp + fp) if (tp + fp) > 0 else 0.0       # PPV
        f1 = (2 * precisao * sensibilidade) / (precisao + sensibilidade) if (precisao + sensibilidade) > 0 else 0.0

        relatorio.append({
            "Patologia": label,
            "AUROC": round(float(auroc), 4),
            "Acuracia": round(float(acuracia), 4),
            "Sensibilidade": round(float(sensibilidade), 4),
            "Especificidade": round(float(especificidade), 4),
            "Precisao": round(float(precisao), 4),
            "F1_Score": round(float(f1), 4)
        })

    df_metricas = pd.DataFrame(relatorio)
    
    # Adiciona a linha de média global (Macro Average)
    media_row = {
        "Patologia": "MÉDIA (Macro Avg)",
        "AUROC": round(df_metricas["AUROC"].mean(), 4),
        "Acuracia": round(df_metricas["Acuracia"].mean(), 4),
        "Sensibilidade": round(df_metricas["Sensibilidade"].mean(), 4),
        "Especificidade": round(df_metricas["Especificidade"].mean(), 4),
        "Precisao": round(df_metricas["Precisao"].mean(), 4),
        "F1_Score": round(df_metricas["F1_Score"].mean(), 4)
    }
    df_metricas_com_media = pd.concat([df_metricas, pd.DataFrame([media_row])], ignore_index=True)

    # Exibe a tabela formatada no terminal
    print("\n" + "="*85)
    print(f"RELATÓRIO COMPLETO DE MÉTRICAS COMPLEMENTARES (Limiar = {threshold})")
    print("="*85)
    print(df_metricas_com_media.to_string(index=False))
    print("="*85 + "\n")

    # 1. Salva a tabela em arquivo CSV (Ideal para tabelas da Monografia/Excel)
    csv_path = "results/relatorio_metricas.csv"
    df_metricas_com_media.to_csv(csv_path, index=False, encoding='utf-8-sig')
    
    # 2. Salva em arquivo JSON
    json_path = "results/detailed_metrics.json"
    with open(json_path, "w") as f:
        json.dump(relatorio, f, indent=4)

    print(f"✅ Tabela de métricas salva em CSV: {csv_path}")
    print(f"✅ Relatório de métricas salvo em JSON: {json_path}")

def main():
    os.makedirs("results", exist_ok=True)
    print("Extraindo predições e rótulos do conjunto de teste...")
    y_true, y_pred = obter_predicoes_e_rotulos()
    
    print("Gerando gráficos e matrizes de confusão...")
    gerar_grafico_roc(y_true, y_pred)
    mcm = gerar_matrizes_confusao(y_true, y_pred, threshold=0.5)
    
    print("Calculando métricas complementares...")
    calcular_metricas_detalhadas(y_true, y_pred, mcm, threshold=0.5)

if __name__ == "__main__":
    main()