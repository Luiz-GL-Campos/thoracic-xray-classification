import os
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
            # Caso uma classe não possua exemplos positivos no lote
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
        
        # Aplica Sigmoid para converter logits em probabilidades
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
    
    # 3. Instancia Modelo, Loss e Otimizador
    model = get_resnet50_model(num_classes=NUM_CLASSES, pretrained=True).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    
    # 4. Loop de Treinamento
    epochs = 10
    best_val_auroc = 0.0
    os.makedirs("checkpoints", exist_ok=True)

    # Verifica se já existe um modelo salvo para continuar o treino
    checkpoint_path = "checkpoints/best_resnet50.pth"
    if os.path.exists(checkpoint_path):
        print("Checkpoint encontrado! Carregando pesos salvos para retomar o treino...")
        model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))
        
        # Avalia a AUROC do modelo carregado para atualizar best_val_auroc
        _, best_val_auroc, _ = avaliar(model, val_loader, criterion, DEVICE)
        print(f" -> Modelo carregado com Val Mean AUROC inicial de: {best_val_auroc:.4f}")
    
    for epoch in range(1, epochs + 1):
        train_loss = treinar_uma_epoca(model, train_loader, criterion, optimizer, DEVICE)
        val_loss, val_auroc, _ = avaliar(model, val_loader, criterion, DEVICE)
        
        print(f"Época [{epoch}/{epochs}] | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Mean AUROC: {val_auroc:.4f}")
        
        # Salva o melhor checkpoint com base na AUROC de validação
        if val_auroc > best_val_auroc:
            best_val_auroc = val_auroc
            torch.save(model.state_dict(), "checkpoints/best_resnet50.pth")
            print(" -> Melhor modelo salvo!")
            
    # 5. Avaliação Final no Conjunto de Teste
    print("\n--- AVALIAÇÃO FINAL NO TESTE ---")
    model.load_state_dict(torch.load("checkpoints/best_resnet50.pth"))
    test_loss, test_auroc, class_auroc = avaliar(model, test_loader, criterion, DEVICE)
    
    print(f"Test Loss: {test_loss:.4f} | Test Mean AUROC: {test_auroc:.4f}\n")
    print("AUROC por Patologia:")
    for label, score in class_auroc.items():
        print(f"  - {label}: {score:.4f}")

if __name__ == "__main__":
    main()