import os
import re
import time
import json
from datetime import datetime
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


def extrair_numero_teste(test_id):
    """Extrai o numero inteiro de uma string no formato 'testeXX'."""
    match = re.search(r"teste(\d+)", str(test_id))
    return int(match.group(1)) if match else 0


def preparar_registros_incrementais(results_dir):
    """
    Carrega os historicos existentes (CSV e JSON), migra formatos antigos se necessario
    e define o proximo identificador sequencial (teste01, teste02, ...).
    """
    history_json_path = os.path.join(results_dir, "history.json")
    metrics_json_path = os.path.join(results_dir, "metrics.json")
    history_csv_path = os.path.join(results_dir, "history.csv")

    history_json_data = {}
    metrics_json_data = {}
    df_csv_existente = pd.DataFrame()
    numeros_existentes = []

    # 1. Carrega ou migra history.json
    if os.path.exists(history_json_path):
        with open(history_json_path, "r", encoding="utf-8") as f:
            conteudo = json.load(f)
        if isinstance(conteudo, list):
            history_json_data = {"teste01": {"epochs": conteudo}}
            numeros_existentes.append(1)
        elif isinstance(conteudo, dict):
            history_json_data = conteudo
            numeros_existentes.extend([extrair_numero_teste(k) for k in conteudo.keys()])

    # 2. Carrega ou migra metrics.json
    if os.path.exists(metrics_json_path):
        with open(metrics_json_path, "r", encoding="utf-8") as f:
            conteudo_metrics = json.load(f)
        if isinstance(conteudo_metrics, dict):
            if "test_mean_auroc" in conteudo_metrics:
                metrics_json_data = {"teste01": conteudo_metrics}
                numeros_existentes.append(1)
            else:
                metrics_json_data = conteudo_metrics
                numeros_existentes.extend([extrair_numero_teste(k) for k in conteudo_metrics.keys()])

    # 3. Carrega ou migra history.csv
    if os.path.exists(history_csv_path):
        df_csv_existente = pd.read_csv(history_csv_path)
        if not df_csv_existente.empty:
            if "test_id" not in df_csv_existente.columns:
                df_csv_existente.insert(0, "test_id", "teste01")
                numeros_existentes.append(1)
            else:
                numeros_existentes.extend([extrair_numero_teste(tid) for tid in df_csv_existente["test_id"].unique()])

    proximo_num = max(numeros_existentes) + 1 if numeros_existentes else 1
    run_id = f"teste{proximo_num:02d}"

    return run_id, history_json_data, metrics_json_data, df_csv_existente


def calcular_auroc(y_true, y_pred, labels):
    """Calcula a AUROC para cada uma das 14 patologias e retorna a media."""
    auroc_por_classe = {}
    for i, label in enumerate(labels):
        try:
            score = roc_auc_score(y_true[:, i], y_pred[:, i])
            auroc_por_classe[label] = score
        except ValueError:
            auroc_por_classe[label] = np.nan

    mean_auroc = np.nanmean(list(auroc_por_classe.values()))
    return mean_auroc, auroc_por_classe


def treinar_uma_epoca(model, dataloader, criterion, optimizer, scaler, device, epoch, total_epochs, run_id):
    model.train()
    total_loss = 0.0

    pbar = tqdm(
        dataloader,
        desc=f"[{run_id}] Epoca [{epoch}/{total_epochs}] - Treinando ({MODEL_NAME})",
        leave=False,
        dynamic_ncols=True
    )

    for images, targets in pbar:
        images, targets = images.to(device, non_blocking=True), targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast("cuda"):
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

        with torch.amp.autocast("cuda"):
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
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Prepara historicos incrementais e define o ID do teste atual
    run_id, history_json_data, metrics_json_data, df_csv_existente = preparar_registros_incrementais(RESULTS_DIR)
    data_inicio = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Configuracao de hiperparametros por arquitetura
    epochs = 10
    batch_size = 64 if "vit" in MODEL_NAME.lower() else 128
    learning_rate = 3e-5 if "vit" in MODEL_NAME.lower() else 1e-4
    weight_decay = 1e-2 if "vit" in MODEL_NAME.lower() else 1e-4

    print(f"Utilizando dispositivo: {DEVICE}")
    print(f"Modelo selecionado: {MODEL_NAME}")
    print(f"Identificador da execucao: {run_id}")

    # 1. Carrega os conjuntos pre-processados
    train_df = pd.read_csv("data/processed/train.csv")
    val_df = pd.read_csv("data/processed/val.csv")
    test_df = pd.read_csv("data/processed/test.csv")

    # 2. Prepara os DataLoaders
    train_tf, val_tf = get_transforms()

    train_dataset = NIHChestXrayDataset(train_df, IMAGES_DIR, transform=train_tf, labels=LABELS)
    val_dataset = NIHChestXrayDataset(val_df, IMAGES_DIR, transform=val_tf, labels=LABELS)
    test_dataset = NIHChestXrayDataset(test_df, IMAGES_DIR, transform=val_tf, labels=LABELS)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
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

    # 3. Instancia modelo limpo (ImageNet)
    print(f"Carregando arquitetura e pesos do {MODEL_NAME}...")
    model = build_model(model_name=MODEL_NAME, num_classes=NUM_CLASSES, pretrained=True).to(DEVICE)
    print("Modelo carregado na GPU com sucesso.")

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scaler = torch.amp.GradScaler("cuda")

    best_val_auroc = 0.0
    checkpoint_latest = os.path.join(CHECKPOINT_DIR, f"best_{MODEL_NAME}.pth")
    checkpoint_run = os.path.join(CHECKPOINT_DIR, f"best_{MODEL_NAME}_{run_id}.pth")

    history_csv_path = os.path.join(RESULTS_DIR, "history.csv")
    history_json_path = os.path.join(RESULTS_DIR, "history.json")
    metrics_json_path = os.path.join(RESULTS_DIR, "metrics.json")

    hyperparams = {
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay
    }

    historico_epocas_atual = []
    start_train_time = time.time()

    print(f"Iniciando o treinamento ({epochs} Epocas) - {run_id}...")

    for epoch in range(1, epochs + 1):
        epoch_start = time.time()

        train_loss = treinar_uma_epoca(model, train_loader, criterion, optimizer, scaler, DEVICE, epoch, epochs, run_id)
        val_loss, val_auroc, _ = avaliar(model, val_loader, criterion, DEVICE, desc=f"[{run_id}] Epoca [{epoch}/{epochs}] - Validando")

        epoch_duration = time.time() - epoch_start

        print(f"[{run_id}] Epoca [{epoch}/{epochs}] | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Mean AUROC: {val_auroc:.4f} | Tempo: {epoch_duration:.1f}s")

        dados_epoca_csv = {
            "test_id": run_id,
            "timestamp": data_inicio,
            "epoch": epoch,
            "train_loss": round(float(train_loss), 4),
            "val_loss": round(float(val_loss), 4),
            "val_mean_auroc": round(float(val_auroc), 4),
            "epoch_time_seconds": round(epoch_duration, 2)
        }
        historico_epocas_atual.append(dados_epoca_csv)

        if val_auroc > best_val_auroc:
            best_val_auroc = val_auroc
            torch.save(model.state_dict(), checkpoint_run)
            torch.save(model.state_dict(), checkpoint_latest)
            print(f"Melhor modelo salvo em: {checkpoint_run}")

        # Atualiza history.csv concatenando os testes anteriores com o teste atual
        df_atual = pd.DataFrame(historico_epocas_atual)
        if df_csv_existente.empty:
            df_final = df_atual
        else:
            df_final = pd.concat([df_csv_existente, df_atual], ignore_index=True)
        df_final.to_csv(history_csv_path, index=False)

        # Atualiza history.json mantendo os testes anteriores sob suas chaves
        history_json_data[run_id] = {
            "timestamp": data_inicio,
            "model_name": MODEL_NAME,
            "hyperparameters": hyperparams,
            "epochs": [
                {
                    "epoch": item["epoch"],
                    "train_loss": item["train_loss"],
                    "val_loss": item["val_loss"],
                    "val_mean_auroc": item["val_mean_auroc"],
                    "epoch_time_seconds": item["epoch_time_seconds"]
                }
                for item in historico_epocas_atual
            ]
        }
        with open(history_json_path, "w", encoding="utf-8") as f:
            json.dump(history_json_data, f, indent=4)

    train_duration = time.time() - start_train_time
    train_min, train_sec = divmod(train_duration, 60)
    train_time_str = f"{int(train_min)}m {train_sec:.2f}s"
    print(f"Tempo total de Treinamento ({run_id}): {train_time_str}")

    # 5. Avaliacao Final no Conjunto de Teste
    print(f"--- AVALIACAO FINAL NO TESTE ({run_id}) ---")
    model.load_state_dict(torch.load(checkpoint_run, map_location=DEVICE))

    start_test_time = time.time()
    test_loss, test_auroc, class_auroc = avaliar(model, test_loader, criterion, DEVICE, desc=f"[{run_id}] Avaliando no Teste")
    test_duration = time.time() - start_test_time
    test_min, test_sec = divmod(test_duration, 60)
    test_time_str = f"{int(test_min)}m {test_sec:.2f}s"

    print(f"[{run_id}] Test Loss: {test_loss:.4f} | Test Mean AUROC: {test_auroc:.4f}")
    print(f"Tempo total de Teste (Inferencia): {test_time_str}")

    # Consolida o relatorio final do teste no metrics.json sob a chave run_id
    metrics_json_data[run_id] = {
        "timestamp": data_inicio,
        "model_name": MODEL_NAME,
        "checkpoint_file": checkpoint_run,
        "hyperparameters": hyperparams,
        "execution_time": {
            "train_time_seconds": round(train_duration, 2),
            "train_time_formatted": train_time_str,
            "test_time_seconds": round(test_duration, 2),
            "test_time_formatted": test_time_str
        },
        "best_val_mean_auroc": float(best_val_auroc),
        "test_loss": float(test_loss),
        "test_mean_auroc": float(test_auroc),
        "auroc_per_class": {k: float(v) for k, v in class_auroc.items()}
    }

    with open(metrics_json_path, "w", encoding="utf-8") as f:
        json.dump(metrics_json_data, f, indent=4)

    print(f"Historico incremental atualizado em: {history_csv_path} e {history_json_path}")
    print(f"Metricas finais adicionadas em: {metrics_json_path} sob a chave '{run_id}'")


if __name__ == "__main__":
    main()