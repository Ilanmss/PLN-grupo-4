import os
import re
import pandas as pd

# Descobre o caminho do diretório onde o próprio script clean_data.py está salvo (pasta ilan)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Ajusta os caminhos para buscar a pasta 'data' que está 1 nível acima (em PLN EP1)
INPUT_PATH = os.path.abspath(os.path.join(BASE_DIR, "..", "data", "train_no_duplicate.xlsx"))
OUTPUT_PATH = os.path.abspath(os.path.join(BASE_DIR, "..", "data", "train_cleaned.xlsx"))

TEXT_COL = "resp_text"
LABEL_COL = "clarity"

# Expressões regulares para identificar frases puramente operacionais/anexos
GENERIC_PATTERNS = [
    r"^(segue|em|conforme)?\s*(resposta|informaç[ãa]o|question[áa]rio|documento|pedido|of[íi]cio)?\s*(em\s*)?anexo\.?$",
    r"^resposta\s*enviada\s*por\s*(e-?mail|email)\.?$",
    r"^prezado\(?a\)?\s*(senhor\(?a\)?|sr\.?|sra\.?|cidad[ãa]o)?\,?$",
    r"^pergunta\s*(duplicada|repetida)\.?$",
    r"^pedido\s*duplicado\.?$",
    r"^solicitaç[ãa]o\s*inserida\s*no\s*e-sic\.?$",
]

PATTERNS_COMPILED = [re.compile(p, re.IGNORECASE) for p in GENERIC_PATTERNS]


def clean_spaces(text):
    """Remove múltiplos espaços, tabulações e quebras de linha."""
    if not isinstance(text, str):
        return ""
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def is_generic_or_noise(text):
    """Retorna True se o texto for um ruído sem conteúdo (ex: apenas saudação ou aviso de anexo)."""
    text_clean = text.strip()

    # 1. Menos de 3 caracteres ou numérico puro
    if len(text_clean) < 3 or text_clean.isdigit():
        return True

    # 2. Textos com menos de 6 palavras que batem com padrões genéricos
    words = text_clean.split()
    if len(words) <= 6:
        for pattern in PATTERNS_COMPILED:
            if pattern.search(text_clean):
                return True

    return False


def resolve_label_conflicts(df, threshold=0.80):
    """
    Trata textos idênticos com classes diferentes:
    - Se a classe mais frequente (moda) representar >= threshold (80%) das ocorrências,
      ela é atribuída como o rótulo oficial.
    - Se a moda representara menos de threshold (80%), o texto é removido por ambiguidade.
    """
    print(f"\n[3/4] Resolvendo conflitos de classes (Limiar da Moda: >= {int(threshold*100)}%)...")

    def get_valid_majority_label(series):
        counts = series.value_counts()
        total = len(series)
        top_count = counts.iloc[0]
        
        # Calcula a proporção da classe mais frequente
        proportion = top_count / total
        
        if proportion >= threshold:
            return counts.index[0]  # Mantém a moda se atingir o limiar de 80%
        return None  # Descarta se for ambíguo (< 80%)

    # Mapeamento de cada texto para sua classe consolidada ou None
    consolidated_labels = df.groupby(TEXT_COL)[LABEL_COL].agg(get_valid_majority_label)

    # Textos que não atingiram 80% de dominância
    ambiguous_texts = consolidated_labels[consolidated_labels.isnull()].index

    if len(ambiguous_texts) > 0:
        print(f"  • Removendo {len(ambiguous_texts)} textos únicos cujas classes tinham menos de {int(threshold*100)}% de concordância.")
        df = df[~df[TEXT_COL].isin(ambiguous_texts)].copy()

    # Aplica a classe resolvida
    df[LABEL_COL] = df[TEXT_COL].map(consolidated_labels)

    # Mantém apenas 1 ocorrência por texto único
    before_dedup = len(df)
    df = df.drop_duplicates(subset=[TEXT_COL], keep="first")
    print(f"  • Base unificada: reduzida de {before_dedup} para {len(df)} registros únicos de texto.")

    return df


def process_pipeline():
    if not os.path.exists(INPUT_PATH):
        print(f"❌ Arquivo não encontrado: {INPUT_PATH}")
        return

    print(f"Carregando base original: {INPUT_PATH}...")
    df = pd.read_excel(INPUT_PATH)
    initial_count = len(df)

    # 1. Limpeza de espaços
    print("\n[1/4] Normalizando e limpando múltiplos espaços...")
    df[TEXT_COL] = df[TEXT_COL].apply(clean_spaces)

    # 2. Filtro de ruídos de anexos / saudações / textos sem conteúdo
    print("\n[2/4] Identificando e removendo textos genéricos (sem conteúdo real)...")
    noise_mask = df[TEXT_COL].apply(is_generic_or_noise)
    noise_count = noise_mask.sum()
    df = df[~noise_mask].copy()
    print(f"  • {noise_count} textos genéricos/ruídos foram removidos.")

    # 3. Resolução de conflitos de rótulo (mínimo 80% na moda)
    df = resolve_label_conflicts(df, threshold=0.80)

    # 4. Salvar resultado
    print(f"\n[4/4] Salvando nova base limpa em: {OUTPUT_PATH}...")
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    df[[TEXT_COL, LABEL_COL]].to_excel(OUTPUT_PATH, index=False)

    print("\n" + "=" * 50)
    print("RESUMO DO TRATAMENTO:")
    print(f"• Registros originais:  {initial_count}")
    print(f"• Registros finais:     {len(df)}")
    print(f"• Total descartados:    {initial_count - len(df)}")
    print("=" * 50)
    print("✓ Tratamento concluído com sucesso!")


if __name__ == "__main__":
    process_pipeline()