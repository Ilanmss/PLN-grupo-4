# Cria data/train_dedup.xlsx: o train.xlsx com textos duplicados colapsados em uma linha
# (rótulo majoritário). Compatível com src.data_loader.load_data (colunas resp_text e clarity),
# com colunas extras para uso como peso (n_ocorrencias) ou rótulo suave (pct_*).
#
# ATENÇÃO: para validação cruzada, NÃO use este arquivo diretamente para avaliar,
# pois ele muda a distribuição dos dados de validação. Deduplique apenas o fold de treino
# (ver avaliar_deduplicacao.py) e avalie sempre em dados originais.

from src.data_loader import load_data
from src.deduplicacao import CLASSES, deduplicar

CAMINHO_DADOS = 'data/train.xlsx'
CAMINHO_SAIDA = 'data/train_dedup.xlsx'
ESTRATEGIA_EMPATE = 'c234'   # 'c234' ou 'remover'
CONCORDANCIA_MINIMA = 0.0    # ex.: 0.5 descarta textos sem maioria clara


def main():
    data = load_data(CAMINHO_DADOS)
    dedup = deduplicar(data, ESTRATEGIA_EMPATE, CONCORDANCIA_MINIMA)
    dedup.to_excel(CAMINHO_SAIDA, index=False)

    repetidos = dedup[dedup['n_ocorrencias'] > 1]
    conflitantes = dedup[dedup['concordancia'] < 1]
    print(f'Linhas originais: {len(data)}')
    print(f'Linhas após deduplicação: {len(dedup)} ({len(data) - len(dedup)} removidas)')
    print(f'Textos repetidos colapsados: {len(repetidos)} (conflitantes: {len(conflitantes)}, empates: {dedup["empate"].sum()})')
    print('\nDistribuição das classes (original -> deduplicado):')
    antes = data['clarity'].value_counts(normalize=True)
    depois = dedup['clarity'].value_counts(normalize=True)
    for c in CLASSES:
        print(f'  {c:5s} {antes[c]:.2%} -> {depois[c]:.2%}')
    print(f'\nArquivo salvo em: {CAMINHO_SAIDA}')


if __name__ == '__main__':
    main()
