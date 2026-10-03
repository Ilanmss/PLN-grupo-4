# Compara ensembles em validação cruzada ALEATÓRIA (estratificada, 5 folds), que imita o conjunto de teste real:
# como no teste, cópias de um mesmo texto podem aparecer no treino e na avaliação (9,8% dos textos do test1.xlsx
# são idênticos a textos do train.xlsx). A validação agrupada do relatório proíbe essas cópias.
#
# Componentes clássicos: treinados em cada fold aleatório (veem as cópias, como na entrega).
# BERTimbau: usa as previsões out-of-fold da validação AGRUPADA (modelos/bertimbau/cv_2ep), porque treinar o BERT
# em 5 folds aleatórios custaria ~11h de GPU. Nessas previsões o BERT nunca viu cópias do texto avaliado, então os
# ensembles com BERT estão em DESVANTAGEM aqui: os valores deles são um limite inferior.
#
# Os candidatos e pesos foram fixados antes de olhar estes resultados (nada é ajustado nestes folds).
#
# Uso: python avaliar_divisao_aleatoria.py

import json
import os

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline

import gerar_relatorio as g
from src.modelos_classicos import FABRICAS

PASTA_SAIDA = 'modelos/cv_aleatoria'
CLASSICOS = ['svc_word12_char25', 'logreg_word12', 'complementnb_word12']


def main():
    os.makedirs(PASTA_SAIDA, exist_ok=True)
    data = g.load_data(g.CAMINHO_DADOS)
    X, rotulos = data['resp_text'], data['clarity']
    y = rotulos.map({c: i for i, c in enumerate(g.CLASSES)}).to_numpy()
    folds = list(StratifiedKFold(5, shuffle=True, random_state=0).split(X, y))

    # Probabilidades out-of-fold dos clássicos e do baseline nos folds aleatórios (em cache)
    modelos = {'baseline': lambda: make_pipeline(TfidfVectorizer(), LogisticRegression(
        class_weight='balanced', max_iter=3000, random_state=123)), **{n: FABRICAS[n] for n in CLASSICOS}}
    P = {}
    for nome, criar in modelos.items():
        caminho = os.path.join(PASTA_SAIDA, f'oof_{nome}.npy')
        if not os.path.exists(caminho):
            probs = np.zeros((len(y), 3))
            for tr, te in folds:
                probs[te] = criar().fit(X.iloc[tr], rotulos.iloc[tr]).predict_proba(X.iloc[te])
            np.save(caminho, probs)
            print(f'{nome}: concluído', flush=True)
        P[nome] = np.load(caminho)

    oof = g.ler_oof()
    P['bertimbau'] = g.matriz(oof, 'bertimbau', np.arange(len(y)))  # previsões da CV agrupada (sem cópias)
    receita = json.load(open(g.CAMINHO_RECEITAS, encoding='utf-8'))['bert']['pesos']

    def media(nomes):
        return sum(P[n] for n in nomes) / len(nomes)

    candidatos = {
        'Baseline TF-IDF + LogReg': P['baseline'],
        'SVC palavra+caractere + ComplementNB (média simples)': media(['svc_word12_char25', 'complementnb_word12']),
        'SVC palavra+caractere + LogReg + ComplementNB (média simples)': media(CLASSICOS),
        'BERTimbau sozinho*': P['bertimbau'],
        'ENTREGUE: BERTimbau + SVC palavra+caractere + LogReg (pesos da receita)*':
            sum(w * P[n] for n, w in receita.items() if w > 0),
        'BERTimbau + SVC palavra+caractere + LogReg + ComplementNB (média simples)*':
            media(['bertimbau'] + CLASSICOS),
    }

    # Textos de avaliação com cópia idêntica no treino do fold
    copia = np.zeros(len(y), dtype=bool)
    for tr, te in folds:
        copia[te] = X.iloc[te].isin(set(X.iloc[tr])).to_numpy()

    linhas = []
    for nome, p in candidatos.items():
        pred = p.argmax(1)
        acc = [(pred[te] == y[te]).mean() for _, te in folds]
        f1 = [f1_score(y[te], pred[te], average='macro') for _, te in folds]
        linhas.append({'modelo': nome, 'acuracia': np.mean(acc), 'acuracia_dp': np.std(acc), 'f1_macro': np.mean(f1),
                       'acc_com_copia': (pred[copia] == y[copia]).mean(), 'acc_sem_copia': (pred[~copia] == y[~copia]).mean(),
                       'acc_por_fold': ' '.join(f'{a:.4f}' for a in acc)})
    tabela = pd.DataFrame(linhas)
    tabela.to_csv(os.path.join(PASTA_SAIDA, 'resultados.csv'), index=False, encoding='utf-8-sig')
    print(f'\nValidação cruzada aleatória, 5 folds. Textos de avaliação com cópia idêntica no treino: {copia.mean():.1%}')
    print('* BERT com previsões da CV agrupada (nunca viu cópias): limite inferior para os ensembles com BERT.\n')
    print(tabela.round(4).to_string(index=False))


if __name__ == '__main__':
    main()
