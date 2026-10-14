import torch

# Dispositivo de execução
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Caminhos padrão dos dados
CSV_PATH = "data/raw/Data_Entry_2017.csv"
IMAGES_DIR = "data/raw/images/"

# Lista das 14 patologias alvo
LABELS = [
    "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration",
    "Mass", "Nodule", "Pneumonia", "Pneumothorax",
    "Consolidation", "Edema", "Emphysema", "Fibrosis",
    "Pleural_Thickening", "Hernia"
]

NUM_CLASSES = len(LABELS)  # 14