import os
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from src.config import CSV_PATH, TEST_LIST_PATH, PROCESSED_DATA_DIR, LABELS


def carregar_test_list(test_list_path: str) -> set[str]:
    """Carrega os nomes de arquivo (Image Index) do split oficial de teste."""
    with open(test_list_path, "r", encoding="utf-8") as f:
        return {line.strip() for line in f if line.strip()}


def criar_colunas_multihot(df: pd.DataFrame) -> pd.DataFrame:
    """Cria as 14 colunas binárias (0 ou 1) para cada patologia."""
    df = df.copy()
    finding_series = df["Finding Labels"].fillna("").astype(str)
    for label in LABELS:
        df[label] = finding_series.map(lambda x: 1 if label in x.split("|") else 0)
    return df


def realizar_split_oficial_e_por_paciente(
    df: pd.DataFrame,
    test_images: set[str],
    val_size_from_train_val: float = 0.125,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Usa `test_list.txt` para o conjunto de teste e divide o restante
    por `Patient ID` entre treino e validação sem vazamento de paciente.
    """
    is_test = df["Image Index"].isin(test_images)
    test_df = df[is_test].reset_index(drop=True)
    train_val_df = df[~is_test].reset_index(drop=True)

    gss_val = GroupShuffleSplit(
        n_splits=1,
        test_size=val_size_from_train_val,
        random_state=random_state,
    )
    train_idx, val_idx = next(gss_val.split(train_val_df, groups=train_val_df["Patient ID"]))

    train_df = train_val_df.iloc[train_idx].reset_index(drop=True)
    val_df = train_val_df.iloc[val_idx].reset_index(drop=True)

    # Verificação de integridade (zero sobreposição de Patient ID)
    train_patients = set(train_df["Patient ID"])
    val_patients = set(val_df["Patient ID"])
    test_patients = set(test_df["Patient ID"])

    assert train_patients.isdisjoint(val_patients), "Leakage detectado entre Treino e Validação!"
    assert train_patients.isdisjoint(test_patients), "Leakage detectado entre Treino e Teste!"
    assert val_patients.isdisjoint(test_patients), "Leakage detectado entre Validação e Teste!"

    return train_df, val_df, test_df


def main() -> None:
    print("Lendo CSV original e test_list.txt...")
    df = pd.read_csv(CSV_PATH)
    test_images = carregar_test_list(TEST_LIST_PATH)

    print("Gerando rótulos multi-hot...")
    df = criar_colunas_multihot(df)

    print("Separando test_list oficial e dividindo treino/validação por Patient ID...")
    train_df, val_df, test_df = realizar_split_oficial_e_por_paciente(df, test_images)

    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

    train_path = os.path.join(PROCESSED_DATA_DIR, "train.csv")
    val_path = os.path.join(PROCESSED_DATA_DIR, "val.csv")
    test_path = os.path.join(PROCESSED_DATA_DIR, "test.csv")

    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)

    print("Sucesso!")
    print(f"Treino:    {len(train_df)} amostras ({train_df['Patient ID'].nunique()} pacientes)")
    print(f"Validação: {len(val_df)} amostras ({val_df['Patient ID'].nunique()} pacientes)")
    print(f"Teste:     {len(test_df)} amostras ({test_df['Patient ID'].nunique()} pacientes)")


if __name__ == "__main__":
    main()