# Testa tratamentos de dados para aumentar a distinção entre as classes.
# Avaliação: validação cruzada agrupada, 5 folds (os mesmos do relatório), TF-IDF palavra (1,2) + LinearSVC (C=0,1).
# Tudo que é aprendido (frases frequentes, exemplos suspeitos) usa SOMENTE o treino de cada fold.
#
# Tratamentos:
#   A. remover frases padronizadas (saudações, rodapés legais, assinaturas): frases que se repetem em muitos textos
#   B. acrescentar atributos manuais (tamanho, anexos, links, negativas etc.) ao TF-IDF
#   C. limpeza de rótulos ruidosos (confident learning): remover do treino os exemplos cujo rótulo o próprio
#      modelo (validação cruzada interna) considera mais improvável
#
# Uso (a partir da raiz do projeto): python -m analise_duplicatas.tratamentos_dados [--folds 1]

import argparse
import re
from collections import Counter

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from src.data_loader import load_data
from src.divisoes import folds_cv

CAMINHO_SAIDA = 'analise_duplicatas/tratamentos_dados.csv'


def tfidf():
    return TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)


def frases(texto):
    return [f.strip() for f in re.split(r'(?<=[.!?;:])\s+|\s{2,}', texto) if f.strip()]


def chave_frase(frase):
    return re.sub(r'\d+', '0', re.sub(r'\s+', ' ', frase.lower())).strip()


def remover_frases_padrao(textos_treino, textos_alvo, min_textos):
    """Remove frases que aparecem em pelo menos min_textos textos do TREINO."""
    contagem = Counter(k for t in textos_treino for k in {chave_frase(f) for f in frases(t)})
    padrao = {k for k, n in contagem.items() if n >= min_textos}
    limpos = [' '.join(f for f in frases(t) if chave_frase(f) not in padrao) for t in textos_alvo]
    return limpos, len(padrao)


PADROES = {
    'anexo': r'anex', 'link': r'http|www\.|link', 'negativa': r'não pode|não é possível|negad|indeferid|sigilo|impossib',
    'encaminhado': r'encaminh', 'recurso': r'recurso', 'lei': r'lei n|decreto|art\.', 'email_tel': r'@|0800|telefone',
    'desculpa': r'lament|desculp|infelizmente', 'informamos': r'informamos|esclarecemos',
}


def atributos_manuais(textos):
    linhas = []
    for t in textos:
        tl = t.lower()
        palavras = tl.split()
        linhas.append([np.log1p(len(t)), np.log1p(len(palavras)), np.log1p(len(frases(t))),
                       sum(c.isdigit() for c in t) / max(1, len(t)), sum(c.isupper() for c in t) / max(1, len(t)),
                       np.mean([len(p) for p in palavras]) if palavras else 0.0]
                      + [np.log1p(len(re.findall(p, tl))) for p in PADROES.values()])
    return np.array(linhas)


def avaliar(nome, treinar_prever, X, y, folds, linhas):
    f1s, accs = [], []
    for idx_tr, idx_val in folds:
        pred = treinar_prever([X[i] for i in idx_tr], y[idx_tr], [X[i] for i in idx_val])
        f1s.append(f1_score(y[idx_val], pred, average='macro'))
        accs.append(accuracy_score(y[idx_val], pred))
    linhas.append({'tratamento': nome, 'f1_macro': np.mean(f1s), 'f1_dp': np.std(f1s), 'accuracy': np.mean(accs)})
    print(f'{nome:58s} F1 {np.mean(f1s):.4f} ± {np.std(f1s):.4f} | Acc {np.mean(accs):.4f}', flush=True)


def main():
    p = argparse.ArgumentParser(description='Testa tratamentos de dados')
    p.add_argument('--folds', type=int, default=5, help='quantos dos 5 folds usar (1 = teste rápido só no fold 1)')
    args = p.parse_args()
    data = load_data('data/train.xlsx')
    X, y = data['resp_text'].tolist(), data['clarity'].to_numpy()
    folds = folds_cv(data)[:args.folds]
    print(f'Avaliando em {len(folds)} fold(s)', flush=True)
    linhas = []

    def referencia(X_tr, y_tr, X_val):
        return make_pipeline(tfidf(), LinearSVC(C=0.1)).fit(X_tr, y_tr).predict(X_val)
    avaliar('Referência (sem tratamento)', referencia, X, y, folds, linhas)

    for minimo in (50, 20, 5):
        def sem_padrao(X_tr, y_tr, X_val, minimo=minimo):
            tr, _ = remover_frases_padrao(X_tr, X_tr, minimo)
            val, _ = remover_frases_padrao(X_tr, X_val, minimo)
            return make_pipeline(tfidf(), LinearSVC(C=0.1)).fit(tr, y_tr).predict(val)
        avaliar(f'A. Remover frases presentes em >= {minimo} textos', sem_padrao, X, y, folds, linhas)

    def com_atributos(X_tr, y_tr, X_val):
        v, e = tfidf(), StandardScaler()
        A_tr = hstack([v.fit_transform(X_tr), csr_matrix(e.fit_transform(atributos_manuais(X_tr)) * 0.3)]).tocsr()
        A_val = hstack([v.transform(X_val), csr_matrix(e.transform(atributos_manuais(X_val)) * 0.3)]).tocsr()
        return LinearSVC(C=0.1).fit(A_tr, y_tr).predict(A_val)
    avaliar('B. TF-IDF + atributos manuais', com_atributos, X, y, folds, linhas)

    def so_atributos(X_tr, y_tr, X_val):
        return make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)).fit(
            atributos_manuais(X_tr), y_tr).predict(atributos_manuais(X_val))
    avaliar('B. Só atributos manuais (15 atributos, sem TF-IDF)', so_atributos, X, y, folds, linhas)

    for fracao in (0.1, 0.2, 0.3):
        def limpar_rotulos(X_tr, y_tr, X_val, fracao=fracao):
            # Probabilidade que um modelo dá ao rótulo de cada exemplo de treino, sem tê-lo visto (CV interna)
            interno = make_pipeline(tfidf(), LogisticRegression(C=2.0, max_iter=2000))
            probs = cross_val_predict(interno, X_tr, y_tr, cv=StratifiedKFold(3, shuffle=True, random_state=0),
                                      method='predict_proba')
            classes = np.unique(y_tr)
            p_rotulo = probs[np.arange(len(y_tr)), np.searchsorted(classes, y_tr)]
            manter = p_rotulo > np.quantile(p_rotulo, fracao)
            return make_pipeline(tfidf(), LinearSVC(C=0.1)).fit(
                [x for x, m in zip(X_tr, manter) if m], y_tr[manter]).predict(X_val)
        avaliar(f'C. Remover os {int(fracao * 100)}% de rótulos mais suspeitos do treino', limpar_rotulos, X, y, folds, linhas)

    pd.DataFrame(linhas).to_csv(CAMINHO_SAIDA, index=False, encoding='utf-8-sig')
    print(f'\nResultados salvos em {CAMINHO_SAIDA}')


if __name__ == '__main__':
    main()
