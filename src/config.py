import os
import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

IMAGES_DIR = "data/raw/images"  # Ou data/processed/images_224

LABELS = [
    "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration", "Mass", "Nodule",
    "Pneumonia", "Pneumothorax", "Consolidation", "Edema", "Emphysema",
    "Fibrosis", "Pleural_Thickening", "Hernia"
]

NUM_CLASSES = len(LABELS)

# Definicao do modelo ativo para o experimento (resnet50 ou vit_b_16)
MODEL_NAME = "vit_b_16"

# Diretorios dinamicos baseados no modelo selecionado
CHECKPOINT_DIR = os.path.join("checkpoints", MODEL_NAME)
RESULTS_DIR = os.path.join("results", MODEL_NAME)