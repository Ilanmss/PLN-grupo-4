# Deduplicação de textos repetidos com rótulos conflitantes
# Colapsa cada grupo de textos idênticos (ignorando diferenças de espaços/quebras de linha)
# em uma única linha, com o rótulo majoritário e colunas auxiliares
# (nº de ocorrências, distribuição das classes) para uso como peso ou rótulo suave.

import re

import pandas as pd

CLASSES = ['c1', 'c234', 'c5']


def chave_texto(texto):
    """Chave de agrupamento: textos que só diferem em espaços/quebras de linha são o mesmo texto."""
    return re.sub(r'\s+', ' ', texto).strip()


def _desempatar(linha, estrategia_empate):
    """
    Escolhe o rótulo de um grupo. Sem empate, é a classe majoritária.
    Em caso de empate, estrategia_empate='c234' usa a classe intermediária
    (mediana da escala ordinal c1 < c234 < c5); 'remover' marca o grupo para descarte.
    """
    contagens = linha[CLASSES]
    maximo = contagens.max()
    empatadas = contagens[contagens == maximo].index.tolist()
    if len(empatadas) == 1:
        return empatadas[0]
    return 'c234' if estrategia_empate == 'c234' else None


def deduplicar(data, estrategia_empate='c234', concordancia_minima=0.0):
    """
    Recebe um DataFrame com 'resp_text' e 'clarity' e devolve um DataFrame com uma linha por texto único.

    Parâmetros:
      estrategia_empate: 'c234' (classe intermediária) ou 'remover' (descarta textos empatados).
      concordancia_minima: descarta textos cuja classe majoritária tenha proporção menor que esse valor
                           (ex.: 0.5). Textos que aparecem uma única vez têm concordância 1.0 e nunca são descartados.

    Colunas de saída: resp_text, clarity, n_ocorrencias, c1, c234, c5, pct_c1, pct_c234, pct_c5,
                      concordancia, empate.
    """
    data = data.copy()
    data['chave'] = data['resp_text'].apply(chave_texto)

    contagem = pd.crosstab(data['chave'], data['clarity']).reindex(columns=CLASSES, fill_value=0)
    contagem['n_ocorrencias'] = contagem[CLASSES].sum(axis=1)
    for c in CLASSES:
        contagem[f'pct_{c}'] = (contagem[c] / contagem['n_ocorrencias']).round(4)
    contagem['concordancia'] = (contagem[CLASSES].max(axis=1) / contagem['n_ocorrencias']).round(4)
    contagem['empate'] = contagem[CLASSES].eq(contagem[CLASSES].max(axis=1), axis=0).sum(axis=1) > 1
    contagem['clarity'] = contagem.apply(_desempatar, axis=1, estrategia_empate=estrategia_empate)

    # Mantém a primeira ocorrência do texto original (com a formatação original) como representante
    representante = data.drop_duplicates('chave').set_index('chave')['resp_text']
    contagem['resp_text'] = representante.reindex(contagem.index)

    contagem = contagem[contagem['clarity'].notna()]
    contagem = contagem[contagem['concordancia'] >= concordancia_minima]

    colunas = ['resp_text', 'clarity', 'n_ocorrencias', *CLASSES,
               *[f'pct_{c}' for c in CLASSES], 'concordancia', 'empate']
    return contagem.reset_index(drop=True)[colunas]
