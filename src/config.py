import os
import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

RAW_DATA_DIR = "data/raw"
PROCESSED_DATA_DIR = "data/processed"
CSV_PATH = os.path.join(RAW_DATA_DIR, "Data_Entry_2017.csv")
TEST_LIST_PATH = os.path.join(RAW_DATA_DIR, "test_list.txt")
IMAGES_DIR = os.path.join(RAW_DATA_DIR, "images")  # Ou data/processed/images_224

LABELS = [
    "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration", "Mass", "Nodule",
    "Pneumonia", "Pneumothorax", "Consolidation", "Edema", "Emphysema",
    "Fibrosis", "Pleural_Thickening", "Hernia"
]

NUM_CLASSES = len(LABELS)

# Definição do modelo ativo para o experimento (resnet50 ou vit_b_16)
MODEL_NAME = "vit_b_16"

# Diretórios dinâmicos baseados no modelo selecionado
CHECKPOINT_DIR = os.path.join("checkpoints", MODEL_NAME)
RESULTS_DIR = os.path.join("results", MODEL_NAME)