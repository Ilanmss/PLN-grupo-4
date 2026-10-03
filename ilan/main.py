from pathlib import Path
from src.data_loader import (
    has_duplicates,
    load_data,
    remove_exact_duplicates_and_save,
    show_info_data,
)

BASE_DIR = Path(__file__).resolve().parent

# Se o arquivo estiver dentro da pasta "data":
INPUT_PATH = BASE_DIR / "data" / "train.xlsx"
OUTPUT_PATH = BASE_DIR / "data" / "train_no_duplicate.xlsx"

# Se o arquivo estiver direto na raiz (junto com o main.py), use:
# INPUT_PATH = BASE_DIR / "train.xlsx"
# OUTPUT_PATH = BASE_DIR / "train_no_duplicate.xlsx"


def main():
    df = load_data(INPUT_PATH)

    if has_duplicates(df):
        print("Duplicatas encontradas! Gerando novo arquivo sem duplicatas...")
        df = remove_exact_duplicates_and_save(df, output_path=OUTPUT_PATH)
    else:
        print("Nenhuma duplicata encontrada no dataset.")

    show_info_data(df)


if __name__ == "__main__":
    main()


# --- Primeiras 5 linhas da base ---
#                                            resp_text clarity
# 0   Prezado(a) Senhor(a),   Esclarecemos que o Se...      c5
# 1   Prezada cidadã,  As informações sobre óbitos ...      c1
# 2   Prezado Senhor Julio,  A Ouvidoria-Geral da P...      c1
# 3   Prezado(a) Senhor(a),   Esclarecemos que o Se...    c234
# 4   Senhor, O Serviço de Informações ao Cidadão d...    c234

# --- Distribuição das Classes (Contagem) ---
# clarity
# c5      6892
# c234    6853
# c1      6347
# Name: count, dtype: int64

# --- Distribuição das Classes (Porcentagem) ---
# clarity
# c5       34.3%
# c234    34.11%
# c1      31.59%
# Name: proportion, dtype: str

# --- Estatísticas de Tamanho das Sentenças ---
# • Caracteres:
#   - Mais curta: 7 caracteres
#   - Mais longa: 12125 caracteres
#   - Média:      1000.38 caracteres
#   - Mediana:    709.0 caracteres

# • Palavras:
#   - Menos palavras: 1 palavras
#   - Mais palavras:  1821 palavras
#   - Média:          144.27 palavras
#   - Mediana:        101.0 palavras