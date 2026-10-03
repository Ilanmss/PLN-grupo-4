# Divisões treino/avaliação usadas para comparar modelos. Todos os scripts de avaliação usam estas funções.
#
#   'agrupada' -> GroupKFold (5 folds, embaralhado, semente 42) agrupado por texto normalizado; usa o fold 1.
#                 Rigorosa: nenhuma cópia de um mesmo texto fica no treino e na avaliação ao mesmo tempo.
#   'baseline' -> train_test_split aleatório 80/20 estratificado, semente 123: a mesma divisão do baselines.py
#                 (baseline oficial armazenado em modelos/baselines). Templates repetidos aparecem dos dois lados,
#                 como provavelmente acontecerá no conjunto de teste do professor.

import numpy as np
from sklearn.model_selection import GroupKFold, train_test_split

from src.deduplicacao import chave_texto

DIVISOES = ['agrupada', 'baseline']
DESCRICAO = {
    'agrupada': 'holdout agrupado por texto (GroupKFold fold 1, semente 42)',
    'baseline': 'divisão do baseline oficial (aleatória 80/20 estratificada, semente 123)',
}


def folds_cv(data, n_folds=5, semente=42):
    """Folds da validação cruzada agrupada por texto (o fold 1 é a divisão 'agrupada')."""
    grupos = data['resp_text'].apply(chave_texto)
    return list(GroupKFold(n_splits=n_folds, shuffle=True, random_state=semente).split(data, groups=grupos))


def obter_divisao(data, nome):
    """Devolve (posições de treino, posições de avaliação) no DataFrame do train.xlsx."""
    if nome == 'agrupada':
        grupos = data['resp_text'].apply(chave_texto)
        return list(GroupKFold(n_splits=5, shuffle=True, random_state=42).split(data, groups=grupos))[0]
    if nome == 'baseline':
        posicoes = np.arange(len(data))
        treino, avaliacao = train_test_split(posicoes, test_size=0.2, random_state=123, stratify=data['clarity'])
        return treino, avaliacao
    raise ValueError(f'divisão desconhecida: {nome}')
