import os
import time
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

from src.config import DEVICE, IMAGES_DIR, LABELS, NUM_CLASSES, MODEL_NAME, CHECKPOINT_DIR, RESULTS_DIR
from src.dataset import NIHChestXrayDataset, get_transforms
from src.models import build_model

def calcular_auroc(y_true, y_pred, labels):
    auroc_por_classe = {}
    for i, label in enumerate(labels):
        try:
            score = roc_auc_score(y_true[:, i], y_pred[:, i])
            auroc_por_classe[label] = score
        except ValueError:
            auroc_por_classe[label] = np.nan
    
    mean_auroc = np.nanmean(list(auroc_por_classe.values()))
    return mean_auroc, auroc_por_classe

def treinar_uma_epoca(model, dataloader, criterion, optimizer, scaler, device, epoch, total_epochs):
    model.train()
    total_loss = 0.0
    
    pbar = tqdm(
        dataloader, 
        desc=f"Epoca [{epoch}/{total_epochs}] - Treinando ({MODEL_NAME})", 
        leave=False, 
        dynamic_ncols=True
    )
    
    for images, targets in pbar:
        images, targets = images.to(device, non_blocking=True), targets.to(device, non_blocking=True)
        
        optimizer.zero_grad(set_to_none=True)
        
        with torch.amp.autocast('cuda'):
            outputs = model(images)
            loss = criterion(outputs, targets)
            
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        total_loss += loss.item() * images.size(0)
        pbar.set_postfix(batch_loss=f"{loss.item():.4f}")
        
    return total_loss / len(dataloader.dataset)

@torch.no_grad()
def avaliar(model, dataloader, criterion, device, desc="Avaliando"):
    model.eval()
    total_loss = 0.0
    all_targets = []
    all_preds = []
    
    pbar = tqdm(
        dataloader, 
        desc=desc, 
        leave=False, 
        dynamic_ncols=True
    )
    
    for images, targets in pbar:
        images, targets = images.to(device, non_blocking=True), targets.to(device, non_blocking=True)
        
        with torch.amp.autocast('cuda'):
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
    print(f"Modelo selecionado: {MODEL_NAME}")
    
    train_df = pd.read_csv("data/processed/train.csv")
    val_df = pd.read_csv("data/processed/val.csv")
    test_df = pd.read_csv("data/processed/test.csv")
    
    train_tf, val_tf = get_transforms()
    
    train_dataset = NIHChestXrayDataset(train_df, IMAGES_DIR, transform=train_tf, labels=LABELS)
    val_dataset = NIHChestXrayDataset(val_df, IMAGES_DIR, transform=val_tf, labels=LABELS)
    test_dataset = NIHChestXrayDataset(test_df, IMAGES_DIR, transform=val_tf, labels=LABELS)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=64, 
        shuffle=True, 
        num_workers=4, 
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=128, 
        shuffle=False, 
        num_workers=4, 
        pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=128, 
        shuffle=False, 
        num_workers=4, 
        pin_memory=True
    )
    
    print(f"Carregando arquitetura e pesos do {MODEL_NAME}...")
    model = build_model(model_name=MODEL_NAME, num_classes=NUM_CLASSES, pretrained=True).to(DEVICE)
    print("Modelo carregado na GPU com sucesso.")
    
    criterion = nn.BCEWithLogitsLoss()
    
    # Para ViT, recomenda-se uma taxa de aprendizado ligeiramente menor (ex: 3e-5 a 1e-4) com AdamW
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-5, weight_decay=1e-2)
    scaler = torch.amp.GradScaler('cuda')

    epochs = 10
    best_val_auroc = 0.0
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    checkpoint_file = os.path.join(CHECKPOINT_DIR, f"best_{MODEL_NAME}.pth")
    historico_epocas = []
    start_train_time = time.time()

    print(f"Iniciando o treinamento ({epochs} Epocas)...")

    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
        
        train_loss = treinar_uma_epoca(model, train_loader, criterion, optimizer, scaler, DEVICE, epoch, epochs)
        val_loss, val_auroc, _ = avaliar(model, val_loader, criterion, DEVICE, desc=f"Epoca [{epoch}/{epochs}] - Validando")
        
        epoch_duration = time.time() - epoch_start
        
        print(f"Epoca [{epoch}/{epochs}] | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Mean AUROC: {val_auroc:.4f} | Tempo: {epoch_duration:.1f}s")
        
        dados_epoca = {
            "epoch": epoch,
            "train_loss": round(float(train_loss), 4),
            "val_loss": round(float(val_loss), 4),
            "val_mean_auroc": round(float(val_auroc), 4),
            "epoch_time_seconds": round(epoch_duration, 2)
        }
        historico_epocas.append(dados_epoca)
        
        if val_auroc > best_val_auroc:
            best_val_auroc = val_auroc
            torch.save(model.state_dict(), checkpoint_file)
            print("Melhor modelo salvo!")

        df_hist = pd.DataFrame(historico_epocas)
        df_hist.to_csv(os.path.join(RESULTS_DIR, "history.csv"), index=False)
        with open(os.path.join(RESULTS_DIR, "history.json"), "w") as f:
            json.dump(historico_epocas, f, indent=4)
            
    train_duration = time.time() - start_train_time
    train_min, train_sec = divmod(train_duration, 60)
    train_time_str = f"{int(train_min)}m {train_sec:.2f}s"
    print(f"Tempo total de Treinamento: {train_time_str}")

    print("--- AVALIACAO FINAL NO TESTE (Melhor Checkpoint) ---")
    model.load_state_dict(torch.load(checkpoint_file, map_location=DEVICE))

    start_test_time = time.time()
    test_loss, test_auroc, class_auroc = avaliar(model, test_loader, criterion, DEVICE, desc="Avaliando no Teste")
    test_duration = time.time() - start_test_time
    test_min, test_sec = divmod(test_duration, 60)
    test_time_str = f"{int(test_min)}m {test_sec:.2f}s"

    print(f"Test Loss: {test_loss:.4f} | Test Mean AUROC: {test_auroc:.4f}")
    print(f"Tempo total de Teste (Inferencia): {test_time_str}")

    resultados = {
        "model_name": MODEL_NAME,
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

    with open(os.path.join(RESULTS_DIR, "metrics.json"), "w") as f:
        json.dump(resultados, f, indent=4)

    print(f"Historico salvo em: {RESULTS_DIR}/history.csv e {RESULTS_DIR}/history.json")
    print(f"Metricas do teste final salvas em: {RESULTS_DIR}/metrics.json")

if __name__ == "__main__":
    main()