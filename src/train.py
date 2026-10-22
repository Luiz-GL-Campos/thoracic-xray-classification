import os
import time
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score

from src.config import DEVICE, IMAGES_DIR, LABELS, NUM_CLASSES
from src.dataset import NIHChestXrayDataset, get_transforms
from src.models.resnet import get_resnet50_model

def calcular_auroc(y_true, y_pred, labels):
    """Calcula a AUROC para cada uma das 14 doenças e retorna a média."""
    auroc_por_classe = {}
    for i, label in enumerate(labels):
        try:
            score = roc_auc_score(y_true[:, i], y_pred[:, i])
            auroc_por_classe[label] = score
        except ValueError:
            auroc_por_classe[label] = np.nan
    
    mean_auroc = np.nanmean(list(auroc_por_classe.values()))
    return mean_auroc, auroc_por_classe

def treinar_uma_epoca(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    
    for images, targets in dataloader:
        images, targets = images.to(device), targets.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * images.size(0)
        
    return total_loss / len(dataloader.dataset)

@torch.no_grad()
def avaliar(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_targets = []
    all_preds = []
    
    for images, targets in dataloader:
        images, targets = images.to(device), targets.to(device)
        outputs = model(images)
        loss = criterion(outputs, targets)
        
        total_loss += loss.item() * images.size(0)
        
        preds = torch.sigmoid(outputs)
        
        all_targets.append(targets.cpu().numpy())
        all_preds.append(preds.cpu().numpy())
        
    y_true = np.vstack(all_targets)
    y_pred = np.vstack(all_preds)
    
    mean_auroc, auroc_classes = calcular_auroc(y_true, y_pred, LABELS)
    epoch_loss = total_loss / len(dataloader.dataset)
    
    return epoch_loss, mean_auroc, auroc_classes

def main():
    print(f"Utilizando dispositivo: {DEVICE}")
    
    # 1. Carrega os conjuntos pré-processados
    train_df = pd.read_csv("data/processed/train.csv")
    val_df = pd.read_csv("data/processed/val.csv")
    test_df = pd.read_csv("data/processed/test.csv")
    
    # 2. Prepara os DataLoaders
    train_tf, val_tf = get_transforms()
    
    train_dataset = NIHChestXrayDataset(train_df, IMAGES_DIR, transform=train_tf, labels=LABELS)
    val_dataset = NIHChestXrayDataset(val_df, IMAGES_DIR, transform=val_tf, labels=LABELS)
    test_dataset = NIHChestXrayDataset(test_df, IMAGES_DIR, transform=val_tf, labels=LABELS)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)
    
    # 3. Instancia Modelo limpo (do zero com ImageNet), Loss e Otimizador
    print("Iniciando modelo limpo a partir de pesos pré-treinados do ImageNet...")
    model = get_resnet50_model(num_classes=NUM_CLASSES, pretrained=True).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    
    # 4. Loop de Treinamento
    epochs = 10
    best_val_auroc = 0.0
    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    historico_epocas = []
    start_train_time = time.time()

    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
        
        train_loss = treinar_uma_epoca(model, train_loader, criterion, optimizer, DEVICE)
        val_loss, val_auroc, _ = avaliar(model, val_loader, criterion, DEVICE)
        
        epoch_duration = time.time() - epoch_start
        
        print(f"Época [{epoch}/{epochs}] | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Mean AUROC: {val_auroc:.4f} | Tempo: {epoch_duration:.1f}s")
        
        # Guarda o registro completo da época
        dados_epoca = {
            "epoch": epoch,
            "train_loss": round(float(train_loss), 4),
            "val_loss": round(float(val_loss), 4),
            "val_mean_auroc": round(float(val_auroc), 4),
            "epoch_time_seconds": round(epoch_duration, 2)
        }
        historico_epocas.append(dados_epoca)
        
        # Salva o checkpoint apenas se for o MELHOR até o momento
        if val_auroc > best_val_auroc:
            best_val_auroc = val_auroc
            torch.save(model.state_dict(), "checkpoints/best_resnet50.pth")
            print(" -> Melhor modelo salvo!")

        # Atualiza os arquivos de histórico no disco a cada época
        df_hist = pd.DataFrame(historico_epocas)
        df_hist.to_csv("results/history.csv", index=False)
        with open("results/history.json", "w") as f:
            json.dump(historico_epocas, f, indent=4)
            
    train_duration = time.time() - start_train_time
    train_min, train_sec = divmod(train_duration, 60)
    train_time_str = f"{int(train_min)}m {train_sec:.2f}s"
    print(f"\n⏱️  Tempo total de Treinamento: {train_time_str}")

    # 5. Avaliação Final no Conjunto de Teste com o Melhor Modelo
    print("\n--- AVALIAÇÃO FINAL NO TESTE (Melhor Checkpoint) ---")
    model.load_state_dict(torch.load("checkpoints/best_resnet50.pth", map_location=DEVICE))

    start_test_time = time.time()
    test_loss, test_auroc, class_auroc = avaliar(model, test_loader, criterion, DEVICE)
    test_duration = time.time() - start_test_time
    test_min, test_sec = divmod(test_duration, 60)
    test_time_str = f"{int(test_min)}m {test_sec:.2f}s"

    print(f"Test Loss: {test_loss:.4f} | Test Mean AUROC: {test_auroc:.4f}")
    print(f"⏱️  Tempo total de Teste (Inferência): {test_time_str}\n")

    # Salva o relatório consolidado final
    resultados = {
        "execution_time": {
            "train_time_seconds": round(train_duration, 2),
            "train_time_formatted": train_time_str,
            "test_time_seconds": round(test_duration, 2),
            "test_time_formatted": test_time_str
        },
        "best_val_mean_auroc": float(best_val_auroc),
        "test_loss": float(test_loss),
        "test_mean_auroc": float(test_auroc),
        "auroc_per_class": {k: float(v) for k, v in class_auroc.items()},
        "epochs_history": historico_epocas
    }

    with open("results/metrics.json", "w") as f:
        json.dump(resultados, f, indent=4)

    print("✅ Histórico de todas as épocas salvo em: results/history.csv e results/history.json")
    print("✅ Métricas do teste final salvas em: results/metrics.json")

if __name__ == "__main__":
    main()