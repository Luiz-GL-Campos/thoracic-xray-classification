# Classificação Multirrótulo de Patologias Torácicas em Radiografias de Tórax: ResNet-50 vs. Vision Transformers (ViT-B/16) com Explicabilidade Visual

Repositório oficial do Trabalho de Conclusão de Curso (TCC) em **Engenharia de Software** pela **Faculdade de Ciências e Tecnologia em Engenharia (FCTE) — Universidade de Brasília (UnB)**.

O projeto implementa um *pipeline* modular em PyTorch para a classificação multirrótulo de **14 patologias torácicas** na base pública **NIH ChestX-ray14**, comparando dois paradigmas de Aprendizado Profundo — Redes Neurais Convolucionais (**ResNet-50**) e mecanismos de auto-atenção global (**Vision Transformer — ViT-B/16**) — e preparando a auditoria interpretável via **Grad-CAM**, **Grad-CAM++** e mapas de atenção.

---

## Visão Geral do Projeto

* **Base de Dados:** NIH ChestX-ray14 ($112.120$ radiografias frontais PA/AP de $30.805$ pacientes únicos).
* **Paradigmas Comparados:**
  * **ResNet-50** (viés indutivo convolucional local, $\approx 25\text{M}$ parâmetros).
  * **ViT-B/16** (auto-atenção global sobre *patches* de $16 \times 16$ pixels, $\approx 86\text{M}$ parâmetros).
* **Aprendizado por Transferência:** Ambas as redes são inicializadas com pesos pré-treinados na **ImageNet-1K** e ajustadas (*fine-tuning*) com imagens redimensionadas para $224 \times 224 \times 3$.
* **Prevenção de Vazamento de Dados (*Data Leakage*):** Particionamento estrito por `Patient ID` entre os conjuntos de Treino, Validação e Teste, garantindo que exames longitudinais de um mesmo paciente não apareçam em divisões distintas.
* **Otimização e Desempenho Computacional:** Treinamento com função de perda `BCEWithLogitsLoss`, otimizador `AdamW`, Precisão Mista Automática (`AMP FP16`) e salvamento seletivo do *checkpoint* de maior AUROC média na validação.
* **Hardware Utilizado:** GPU NVIDIA GeForce RTX 4070 (12 GB VRAM) e processador Intel Core i7-13700.

---

## Patologias Avaliadas (14 Classes)

O classificador estima probabilidades independentes para 14 achados radiográficos, agrupados na análise do estudo em três perfis morfológicos:

| Categoria Morfológica | Patologias |
| :--- | :--- |
| **Anomalias Geométricas e Estruturais** | `Cardiomegaly`, `Emphysema`, `Hernia` |
| **Alterações Pleurais e Focais** | `Effusion`, `Pneumothorax`, `Pleural_Thickening`, `Mass`, `Nodule` |
| **Opacidades Parenquimatosas Difusas** | `Infiltration`, `Pneumonia`, `Consolidation`, `Edema`, `Atelectasis`, `Fibrosis` |

> **Nota:** Exames sem alterações registradas (*No Finding*) são representados pelo vetor nulo $[0, 0, \dots, 0] \in \{0,1\}^{14}$, seguindo o protocolo padrão da literatura.

---

## Estrutura do Repositório

```text
thoracic-xray-classification/
├── checkpoints/                  # [Ignorado pelo Git] Pesos (.pth) do melhor checkpoint de validação
├── data/
│   ├── raw/                      # [Ignorado pelo Git] Data_Entry_2017.csv e pasta images/ (112.120 PNGs)
│   └── processed/                # [Ignorado pelo Git] Divisões train.csv, val.csv e test.csv por paciente
│
├── results/                      # Artefatos quantitativos e visuais gerados pelos experimentos
│   ├── resnet50/
│   │   ├── teste01/              # Métricas e históricos da Execução 01 (ResNet-50)
│   │   ├── teste02/              # Métricas e históricos da Execução 02 (ResNet-50)
│   │   ├── history.csv
│   │   ├── history.json
│   │   └── metrics.json
│   ├── vit_b_16/
│   │   ├── teste01/              # Métricas e históricos da Execução 01 (ViT-B/16)
│   │   ├── history.csv
│   │   ├── history.json
│   │   └── metrics.json
│   ├── confusion_matrices.json   # Matrizes de confusão por patologia (limiar 0,50)
│   ├── curves_roc.png / curvas_roc.png
│   ├── detailed_metrics.json     # Relatório consolidado de métricas
│   ├── matrizes_confusao.png
│   └── relatorio_metricas.csv
│
├── src/
│   ├── models/
│   │   ├── __init__.py           # Fábrica dinâmica de modelos
│   │   ├── resnet.py             # Adaptação da ResNet-50 para 14 saídas
│   │   └── vit.py                # Adaptação do ViT-B/16 para 14 saídas
│   ├── __init__.py
│   ├── config.py                 # Hiperparâmetros, caminhos, constantes ImageNet e versionamento
│   ├── prepare_data.py           # Codificação multi-hot e particionamento estrito por Patient ID
│   ├── dataset.py                # PyTorch Dataset, Data Augmentation e DataLoaders
│   ├── train.py                  # Loop de treinamento AMP (FP16) e registro incremental
│   ├── evaluate_and_plot.py      # Avaliação no teste, cálculo de AUROC, curvas ROC e matrizes
│   ├── gradcam.py                # Extração de mapas de ativação visual (Grad-CAM / Grad-CAM++)
│   └── utils.py                  # Cálculo de AUROC por classe/média e funções auxiliares
│
├── .gitignore
├── requirements.txt
└── README.md
```

---

## Configuração Experimental e Resultados Preliminares (TCC 1)

### Hiperparâmetros de Treinamento

| Parâmetro | ResNet-50 (CNN) | ViT-B/16 (Transformer) |
| :--- | :--- | :--- |
| **Pesos Iniciais** | ImageNet-1K | ImageNet-1K |
| **Resolução de Entrada** | $224 \times 224 \times 3$ | $224 \times 224 \times 3$ (*Patches* $16 \times 16$) |
| **Otimizador** | `AdamW` | `AdamW` |
| **Taxa de Aprendizado (LR)** | $1{,}0 \times 10^{-4}$ | $3{,}0 \times 10^{-5}$ |
| **Weight Decay** | $1{,}0 \times 10^{-4}$ | $1{,}0 \times 10^{-2}$ |
| **Batch Size** | $128$ | $64$ (Treino) / $128$ (Val e Teste) |
| **Épocas / Melhor Checkpoint** | $10$ épocas (Pico na Época 5) | $10$ épocas (Pico na Época 5) |
| **Precisão Numérica** | `AMP FP16` | `AMP FP16` |

### Comparativo Global (Mean AUROC no Conjunto de Teste)

| Modelo / Referência | Paradigma | Mean AUROC |
| :--- | :--- | :---: |
| Wang et al. (2017) — *Baseline NIH* | Convolucional (ResNet-50) | `0,7381` |
| Taslimi et al. (2022) — *SwinCheX* | Auto-Atenção Hierárquica (Swin) | `0,8088` |
| **ViT-B/16 — Teste 01 (Este Trabalho)** | Auto-Atenção Global (ViT) | **`0,8192`** |
| **ResNet-50 — Teste 02 (Este Trabalho)** | Convolucional (ResNet-50) | **`0,8279`** |
| **ResNet-50 — Teste 01 (Este Trabalho)** | Convolucional (ResNet-50) | **`0,8300`** |
| Rajpurkar et al. (2017) — *CheXNet* | Convolucional (DenseNet-121) | `0,8414` |

### Principais Achados desta Etapa
1. **Patologias Estruturais vs. Focais:** Ambos os modelos superam `0,87`–`0,91` de AUROC em alterações geométricas e pleurais bem definidas (`Cardiomegaly`: `0,9099` na ResNet-50 e `0,9080` no ViT-B/16; `Emphysema`: `0,9105`; `Hernia`: `0,9144`). Em contrapartida, a tokenização em *patches* de $16 \times 16$ no ViT-B/16 reduz a sensibilidade em lesões focais pequenas como `Nodule` (`0,7201` vs. `0,7553` na ResNet-50).
2. **Teto Informacional em Classes Difusas:** Tanto a ResNet-50 quanto o ViT-B/16 mantêm desempenho limitado em `Infiltration` (`0,7069` e `0,7059`) e `Pneumonia` (`0,7330` e `0,7327`), evidenciando que o gargalo decorre principalmente do ruído de rotulagem via PNL e da sobreposição visual bidimensional, e não apenas do campo receptivo da arquitetura.

---

## Como Executar o Projeto

### 1. Clonar o repositório e configurar o ambiente

```bash
git clone https://github.com/Luiz-GL-Campos/thoracic-xray-classification.git
cd thoracic-xray-classification

python -m venv .venv
# Linux/macOS:
source .venv/bin/activate
# Windows:
.venv\Scripts\activate

pip install -r requirements.txt
```

### 2. Preparar a base NIH ChestX-ray14

Baixe os arquivos da base **NIH ChestX-ray14** (`Data_Entry_2017.csv` e as imagens em PNG), organize-os em `data/raw/images/` e execute o particionamento estrito por paciente:

```bash
python -m src.prepare_data
```

### 3. Treinar e avaliar os modelos

Selecione a arquitetura desejada (`resnet50` ou `vit_b_16`) em `src/config.py` e execute:

```bash
# Treinamento com validação época a época e salvamento do melhor checkpoint
python -m src.train

# Avaliação no conjunto de teste e geração de curvas ROC e matrizes de confusão
python -m src.evaluate_and_plot
```

---

## Próximos Passos (Etapa TCC 2 — Semestre 2027.1)

* **Reprodutibilidade e Calibração de Limiares:** Novas rodadas independentes para o ViT-B/16 e ajuste de limiares ótimos por patologia via índice de Youden (mitigando o colapso de predição positiva sob limiar fixo $0{,}50$ em classes raras como `Pneumonia`).
* **Mitigação para Classes Difusas:** Experimentos com ponderação de positivos na função de perda (`pos_weight` e *Focal Loss*) para ampliar a discriminação em `Infiltration` e `Pneumonia`.
* **Arquitetura Híbrida (ResNet-50 + ViT):** Integração de mapas de características convolucionais locais com camadas de auto-atenção global.
* **Auditoria de Explicabilidade Visual (XAI):** Aplicação comparativa de **Grad-CAM**, **Grad-CAM++** e mapas de atenção (*Attention Rollout*) sobre acertos e erros em patologias estruturais e difusas para detecção de correlações espúrias.