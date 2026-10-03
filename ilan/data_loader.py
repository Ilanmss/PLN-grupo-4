import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def load_data(path):
    data = pd.read_excel(path)
    data["resp_text"] = data["resp_text"].fillna("")  # Substitui nulos por texto vazio
    data["resp_text"] = data["resp_text"].astype(str)  # Garante tipo string
    return data


def analyze_text_length(data):
    """Calcula estatísticas descritivas (min, max, média, mediana) para caracteres e palavras."""
    char_lengths = data["resp_text"].apply(len)
    word_lengths = data["resp_text"].apply(lambda x: len(x.split()))

    stats = {
        "char_min": char_lengths.min(),
        "char_max": char_lengths.max(),
        "char_mean": char_lengths.mean(),
        "char_median": char_lengths.median(),
        "word_min": word_lengths.min(),
        "word_max": word_lengths.max(),
        "word_mean": word_lengths.mean(),
        "word_median": word_lengths.median(),
    }

    print("\n--- Estatísticas de Tamanho das Sentenças ---")
    print("• Caracteres:")
    print(f"  - Mais curta: {stats['char_min']} caracteres")
    print(f"  - Mais longa: {stats['char_max']} caracteres")
    print(f"  - Média:      {stats['char_mean']:.2f} caracteres")
    print(f"  - Mediana:    {stats['char_median']:.1f} caracteres")

    print("\n• Palavras:")
    print(f"  - Menos palavras: {stats['word_min']} palavras")
    print(f"  - Mais palavras:  {stats['word_max']} palavras")
    print(f"  - Média:          {stats['word_mean']:.2f} palavras")
    print(f"  - Mediana:        {stats['word_median']:.1f} palavras")

    return stats

def has_duplicates(data, subset=None):
    """Retorna True se existirem linhas duplicadas no DataFrame."""
    return data.duplicated(subset=subset).any()


def show_info_data(data):
    print("--- Primeiras 5 linhas da base ---")
    print(data.head())

    print("\n--- Distribuição das Classes (Contagem) ---")
    print(data["clarity"].value_counts())

    print("\n--- Distribuição das Classes (Porcentagem) ---")
    print(
        (data["clarity"].value_counts(normalize=True) * 100).round(2).astype(str)
        + "%"
    )

    # 1. Analisa duplicatas 100% idênticas
    check_duplicates_and_conflicts(data)

    # 2. Estatísticas de tamanho dos textos
    analyze_text_length(data)


def check_duplicates_and_conflicts(data, text_col="resp_text", label_col="clarity"):
    """Identifica textos 100% idênticos e verifica se possuem rotulações conflituosas."""
    total_rows = len(data)

    duplicate_mask = data.duplicated(subset=[text_col], keep=False)
    total_duplicates = duplicate_mask.sum()
    unique_duplicated_texts = data[duplicate_mask][text_col].nunique()

    label_counts_per_text = data.groupby(text_col)[label_col].nunique()
    conflicting_texts = label_counts_per_text[label_counts_per_text > 1]
    conflicting_rows = data[data[text_col].isin(conflicting_texts.index)]

    print("\n--- Análise de Duplicatas Exatas (100% iguais) ---")
    print(f"• Total de linhas no dataset: {total_rows}")
    print(
        f"• Linhas com textos repetidos: {total_duplicates} ({(total_duplicates/total_rows)*100:.2f}%)"
    )
    print(
        f"• Quantidade de textos únicos que se repetem: {unique_duplicated_texts}"
    )
    print(f"• Textos únicos com NOTAS CONFLITANTES: {len(conflicting_texts)}")
    print(
        f"• Linhas afetadas por conflitos de notas: {len(conflicting_rows)} ({(len(conflicting_rows)/total_rows)*100:.2f}%)"
    )

    if len(conflicting_texts) > 0:
        print(
            "\nExemplo de textos idênticos com notas DIFERENTES (Primeiros 2 casos):"
        )
        for text in conflicting_texts.index[:2]:
            sub_df = data[data[text_col] == text][[text_col, label_col]]
            print(f'\nTexto: "{text[:100]}..."')
            print("Notas atribuídas a este texto:")
            print(sub_df[label_col].value_counts().to_string())


def split_data(data, test_size=0.3, random_state=42):
    X = data["resp_text"]
    y = data["clarity"]

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    print(f"\nTotal de exemplos de treino: {len(X_train)}")
    print(f"Total de exemplos de validação: {len(X_val)}")

    return X_train, X_val, y_train, y_val


def remove_exact_duplicates_and_save(data_or_path, output_path="data/train_no_duplicate.xlsx"):
    """
    Aceita um DataFrame ou o caminho de um arquivo, remove duplicatas e salva.
    """
    if isinstance(data_or_path, (str, Path)):
        df = pd.read_excel(data_or_path)
    else:
        df = data_or_path

    df_clean = df.drop_duplicates()
    df_clean.to_excel(output_path, index=False)

    print("\n--- Remoção de Duplicatas Exatas ---")
    print(f"• Linhas originais: {len(df)}")
    print(f"• Linhas mantidas: {len(df_clean)}")
    print(f"• Linhas removidas: {len(df) - len(df_clean)}")
    print(f"• Novo arquivo salvo em: '{output_path}'")
    
    return df_clean

# ATENÇÃO: Não coloque nenhuma chamada de função aqui no final do arquivo!