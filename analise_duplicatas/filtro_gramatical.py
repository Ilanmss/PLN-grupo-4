# Teste rápido: filtrar o texto pela classe gramatical (POS) ou lematizar antes de treinar modelos simples.
#
# Etiquetagem com spaCy (pt_core_news_sm). Divisão única, sem validação cruzada: a mesma do baseline oficial
# (aleatória 80/20 estratificada, semente 123). A etiquetagem não usa rótulos, então pode ser feita uma vez para
# todos os textos (fica em cache em analise_duplicatas/cache_pos.pkl).
#
# Uso (a partir da raiz do projeto): python -m analise_duplicatas.filtro_gramatical

import os
import pickle
import time

import pandas as pd
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.naive_bayes import ComplementNB, MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.svm import LinearSVC

from src.data_loader import load_data
from src.divisoes import obter_divisao

CAMINHO_CACHE = 'analise_duplicatas/cache_pos.pkl'
CAMINHO_SAIDA = 'analise_duplicatas/filtro_gramatical.csv'

VERBOS = {'VERB', 'AUX'}
FUNCIONAIS = {'DET', 'ADP', 'PRON', 'CCONJ', 'SCONJ', 'AUX', 'PUNCT', 'SYM', 'SPACE'}
CONTEUDO = {'NOUN', 'PROPN', 'ADJ', 'VERB', 'ADV'}

# nome -> função que recebe a lista de (texto, POS, lema) de um documento e devolve o texto filtrado
VARIANTES = {
    'Original': lambda toks: ' '.join(t for t, p, l in toks),
    'Lematizado': lambda toks: ' '.join(l for t, p, l in toks),
    'Sem verbos (remove predicados: VERB, AUX)': lambda toks: ' '.join(t for t, p, l in toks if p not in VERBOS),
    'Só verbos (VERB, AUX)': lambda toks: ' '.join(t for t, p, l in toks if p in VERBOS),
    'Sem palavras funcionais (DET, ADP, PRON, conj., AUX)': lambda toks: ' '.join(t for t, p, l in toks if p not in FUNCIONAIS),
    'Só substantivos, nomes próprios e adjetivos': lambda toks: ' '.join(t for t, p, l in toks if p in {'NOUN', 'PROPN', 'ADJ'}),
    'Só substantivos e verbos': lambda toks: ' '.join(t for t, p, l in toks if p in {'NOUN', 'VERB'}),
    'Sem nomes próprios (PROPN)': lambda toks: ' '.join(t for t, p, l in toks if p != 'PROPN'),
    'Só palavras de conteúdo, lematizadas': lambda toks: ' '.join(l for t, p, l in toks if p in CONTEUDO),
    'Texto + classes gramaticais (ex.: "pedido NOUN")': lambda toks: ' '.join(f'{t} {p}' for t, p, l in toks),
}

MODELOS = {
    'Baseline TF-IDF + LogReg': lambda: make_pipeline(
        TfidfVectorizer(), LogisticRegression(class_weight='balanced', max_iter=3000, random_state=123)),
    'TF-IDF (1,2) + LinearSVC': lambda: make_pipeline(
        TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True), LinearSVC(C=0.1)),
    'TF-IDF (1,2) + ComplementNB': lambda: make_pipeline(
        TfidfVectorizer(ngram_range=(1, 2)), ComplementNB(alpha=0.5)),
    'TF-IDF (1,2) + MultinomialNB': lambda: make_pipeline(
        TfidfVectorizer(ngram_range=(1, 2)), MultinomialNB(alpha=0.1)),
}


def etiquetar(textos):
    if os.path.exists(CAMINHO_CACHE):
        with open(CAMINHO_CACHE, 'rb') as f:
            return pickle.load(f)
    nlp = spacy.load('pt_core_news_sm', disable=['parser', 'ner'])
    nlp.max_length = 2_000_000
    inicio, docs = time.time(), []
    for i, doc in enumerate(nlp.pipe(textos, batch_size=64), start=1):
        docs.append([(t.text, t.pos_, t.lemma_.lower()) for t in doc if not t.is_space])
        if i % 2000 == 0:
            print(f'  etiquetados {i}/{len(textos)} ({(time.time() - inicio) / 60:.1f} min)', flush=True)
    with open(CAMINHO_CACHE, 'wb') as f:
        pickle.dump(docs, f)
    return docs


def main():
    data = load_data('data/train.xlsx')
    print('Etiquetando classes gramaticais com spaCy...', flush=True)
    docs = etiquetar(data['resp_text'].tolist())
    idx_tr, idx_te = obter_divisao(data, 'baseline')
    y_tr, y_te = data['clarity'].iloc[idx_tr], data['clarity'].iloc[idx_te]

    linhas = []
    for variante, filtrar in VARIANTES.items():
        textos = pd.Series([filtrar(d) for d in docs])
        X_tr, X_te = textos.iloc[idx_tr], textos.iloc[idx_te]
        for nome, criar in MODELOS.items():
            pred = criar().fit(X_tr, y_tr).predict(X_te)
            linhas.append({'variante': variante, 'modelo': nome, 'f1_macro': f1_score(y_te, pred, average='macro'),
                           'accuracy': accuracy_score(y_te, pred)})
        print(f'{variante:52s} ' + ' | '.join(f"{l['modelo'].split(' + ')[-1]} {l['accuracy']:.4f}"
                                                for l in linhas[-len(MODELOS):]), flush=True)

    tabela = pd.DataFrame(linhas)
    tabela.to_csv(CAMINHO_SAIDA, index=False, encoding='utf-8-sig')
    for metrica in ('accuracy', 'f1_macro'):
        print(f'\n{metrica}:')
        print(tabela.pivot(index='variante', columns='modelo', values=metrica).reindex(list(VARIANTES)).round(4).to_string())
    print(f'\nResultados salvos em {CAMINHO_SAIDA}')


if __name__ == '__main__':
    main()
