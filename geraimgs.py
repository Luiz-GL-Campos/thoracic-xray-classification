import os
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

df = pd.read_csv("data/raw/Data_Entry_2017.csv")
casos = [
    ("No Finding", "Sem achados anormais (No Finding)"),
    ("Cardiomegaly", "Grupo 1 — Estrutural (Cardiomegaly)"),
    ("Effusion", "Grupo 2 — Pleural/Focal (Effusion)"),
    ("Infiltration", "Grupo 3 — Parenquimatoso Difuso (Infiltration)"),
]

fig, axes = plt.subplots(1, 4, figsize=(16, 4.4), dpi=300)
for ax, (classe, titulo) in zip(axes, casos):
    linha = df[df["Finding Labels"] == classe].iloc[0]
    img_nome = linha["Image Index"]
    # Ajuste o caminho caso suas imagens estejam em subpastas images_001/images etc.
    caminho = os.path.join("data/raw/images", img_nome)
    img = Image.open(caminho).convert("L")
    ax.imshow(img, cmap="gray")
    ax.set_title(f"{titulo}\n({img_nome})", fontsize=9.5, fontweight="bold")
    ax.axis("off")

plt.tight_layout(pad=1.0)
plt.savefig("figura_2_1.png", dpi=300, bbox_inches="tight")
print("Figura 2.1 salva em figura_2_1.png")