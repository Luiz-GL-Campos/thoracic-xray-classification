import os
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from src.config import CSV_PATH, LABELS

def criar_colunas_multihot(df):
    """Cria as 14 colunas binárias (0 ou 1) para cada patologia."""
    for label in LABELS:
        df[label] = df['Finding Labels'].map(lambda x: 1 if label in x.split('|') else 0)
    return df

def realizar_split_por_paciente(df):
    """
    Divide o DataFrame em treino (70%), validação (10%) e teste (20%),
    garantindo que exames do mesmo paciente (Patient ID) não fiquem em conjuntos diferentes.
    """
    # 1º Split: Separa 20% para Teste
    gss_test = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
    train_val_idx, test_idx = next(gss_test.split(df, groups=df['Patient ID']))
    
    train_val_df = df.iloc[train_val_idx].reset_index(drop=True)
    test_df = df.iloc[test_idx].reset_index(drop=True)
    
    # 2º Split: Dos 80% restantes, separa 12.5% para Validação (12.5% de 80% = 10% do total)
    gss_val = GroupShuffleSplit(n_splits=1, test_size=0.125, random_state=42)
    train_idx, val_idx = next(gss_val.split(train_val_df, groups=train_val_df['Patient ID']))
    
    train_df = train_val_df.iloc[train_idx].reset_index(drop=True)
    val_df = train_val_df.iloc[val_idx].reset_index(drop=True)
    
    return train_df, val_df, test_df

def main():
    print("Lendo CSV original...")
    df = pd.read_csv(CSV_PATH)
    
    print("Gerando rótulos multi-hot...")
    df = criar_colunas_multihot(df)
    
    print("Dividindo dados por Patient ID...")
    train_df, val_df, test_df = realizar_split_por_paciente(df)
    
    # Garante que a pasta de destino existe
    os.makedirs("data/processed", exist_ok=True)
    
    # Salva os CSVs divididos
    train_df.to_csv("data/processed/train.csv", index=False)
    val_df.to_csv("data/processed/val.csv", index=False)
    test_df.to_csv("data/processed/test.csv", index=False)
    
    print(f"Sucesso!")
    print(f"Treino: {len(train_df)} amostras ({train_df['Patient ID'].nunique()} pacientes)")
    print(f"Validação: {len(val_df)} amostras ({val_df['Patient ID'].nunique()} pacientes)")
    print(f"Teste: {len(test_df)} amostras ({test_df['Patient ID'].nunique()} pacientes)")

if __name__ == "__main__":
    main()