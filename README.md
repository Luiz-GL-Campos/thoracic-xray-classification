# Classificação Multi-Rótulo em Radiografias de Tórax usando ResNet

Este repositório contém a implementação de um modelo de Aprendizado Profundo (*Deep Learning*) baseado na arquitetura **ResNet-50** para a classificação multi-rótulo de 14 patologias torácicas a partir de radiografias de tórax. O trabalho utiliza o dataset público **NIH Chest X-ray14**.

Este projeto faz parte do Trabalho de Conclusão de Curso (TCC).

---

## 🔬 Visão Geral do Projeto

* **Arquitetura Base:** ResNet-50 (pré-treinada no ImageNet com Aprendizado por Transferência).
* **Tipo de Problema:** Classificação Multi-Rótulo (*Multi-Label Classification*). Uma única imagem pode apresentar múltiplas patologias simultaneamente ou nenhuma (*No Finding*).
* **Função de Perda:** `BCEWithLogitsLoss` (Binary Cross-Entropy com ativação Sigmoid por classe).
* **Métrica Principal:** **AUROC** (Area Under the ROC Curve) calculada individualmente para cada classe e média (*Mean AUROC*).
* **Prevenção de Data Leakage:** A divisão entre conjuntos de Treino, Validação e Teste é feita rigorosamente agrupada por `Patient ID` (imagens do mesmo paciente não são compartilhadas entre diferentes conjuntos).

---

## 🎯 Patologias Alvo (14 Classes)

O modelo avalia a probabilidade de presença das seguintes 14 patologias:

1. Atelectasis
2. Cardiomegaly
3. Effusion
4. Infiltration
5. Mass
6. Nodule
7. Pneumonia
8. Pneumothorax
9. Consolidation
10. Edema
11. Emphysema
12. Fibrosis
13. Pleural Thickening
14. Hernia

> **Nota:** Casos rotulados como *No Finding* são representados naturalmente por um vetor nulo ($[0, 0, \dots, 0]$ com 14 posições), seguindo o padrão dos benchmarks da literatura acadêmica.

---

## 📁 Estrutura do Repositório

```text
.
├── data/
│   ├── raw/                  # [Ignorado pelo Git] CSV original e pasta com as imagens
│   │   ├── Data_Entry_2017.csv
│   │   └── images/           # Todas as imagens .png em uma única pasta
│   └── processed/            # [Ignorado pelo Git] CSVs gerados (train.csv, val.csv, test.csv)
│
├── src/
│   ├── __init__.py
│   ├── config.py             # Configurações globais, caminhos e hiperparâmetros
│   ├── prepare_data.py       # Script de geração de rótulos multi-hot e split por paciente
│   ├── dataset.py            # Classe PyTorch Dataset e DataLoaders
│   ├── models/               # Arquiteturas de rede neural
│   │   ├── __init__.py
│   │   └── resnet.py         # Adaptação e definição da ResNet
│   └── utils.py              # Funções utilitárias e cálculo de métricas
│
├── .gitignore                # Bloqueia o envio de dados pesados e imagens ao Git
├── requirements.txt          # Dependências do projeto
└── README.md