# Teste rápido: remover as n palavras mais comuns antes de treinar modelos simples.
#
# Divisão única (sem validação cruzada): a mesma do baseline oficial (aleatória 80/20 estratificada, semente 123).
# As n palavras mais comuns são contadas SOMENTE nos textos de treino (total de ocorrências) e passadas como
# stop words ao vetorizador. Referência extra: lista de stop words do NLTK para português.
#
# Uso (a partir da raiz do projeto): python -m analise_duplicatas.remover_palavras_comuns

from collections import Counter

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.naive_bayes import ComplementNB, MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.svm import LinearSVC

from src.data_loader import load_data
from src.divisoes import obter_divisao

VALORES_N = [0, 10, 25, 50, 100, 200, 500, 1000, 2000]
CAMINHO_SAIDA = 'analise_duplicatas/remover_palavras_comuns.csv'

MODELOS = {
    'Baseline TF-IDF + LogReg': lambda sw: make_pipeline(
        TfidfVectorizer(stop_words=sw), LogisticRegression(class_weight='balanced', max_iter=3000, random_state=123)),
    'TF-IDF (1,2) + LinearSVC': lambda sw: make_pipeline(
        TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, stop_words=sw), LinearSVC(C=0.1)),
    'TF-IDF (1,2) + ComplementNB': lambda sw: make_pipeline(
        TfidfVectorizer(ngram_range=(1, 2), stop_words=sw), ComplementNB(alpha=0.5)),
    'TF-IDF (1,2) + MultinomialNB': lambda sw: make_pipeline(
        TfidfVectorizer(ngram_range=(1, 2), stop_words=sw), MultinomialNB(alpha=0.1)),
}


def main():
    data = load_data('data/train.xlsx')
    idx_tr, idx_te = obter_divisao(data, 'baseline')
    X_tr, y_tr = data['resp_text'].iloc[idx_tr], data['clarity'].iloc[idx_tr]
    X_te, y_te = data['resp_text'].iloc[idx_te], data['clarity'].iloc[idx_te]

    # Contagem de palavras só no treino, com a mesma tokenização do TfidfVectorizer
    analisador = TfidfVectorizer().build_analyzer()
    contagem = Counter(p for t in X_tr for p in analisador(t))
    ranking = [p for p, _ in contagem.most_common()]
    print('20 palavras mais comuns no treino:', ', '.join(ranking[:20]))

    variantes = [(f'n = {n}', ranking[:n] or None) for n in VALORES_N]
    try:  # referência opcional: stop words do NLTK, se o pacote estiver instalado
        import nltk
        nltk.download('stopwords', quiet=True)
        stop_nltk = sorted(set(nltk.corpus.stopwords.words('portuguese')))
        variantes.append((f'NLTK ({len(stop_nltk)} palavras)', stop_nltk))
    except ImportError:
        print('NLTK não instalado: referência com stop words do NLTK omitida.')

    linhas = []
    for rotulo, sw in variantes:
        for nome, criar in MODELOS.items():
            pred = criar(sw).fit(X_tr, y_tr).predict(X_te)
            linhas.append({'remocao': rotulo, 'modelo': nome, 'f1_macro': f1_score(y_te, pred, average='macro'),
                           'accuracy': accuracy_score(y_te, pred)})
        print(f'{rotulo:20s} ' + ' | '.join(f"{l['modelo'].split(' + ')[-1]} {l['accuracy']:.4f}" for l in linhas[-len(MODELOS):]), flush=True)

    tabela = pd.DataFrame(linhas)
    tabela.to_csv(CAMINHO_SAIDA, index=False, encoding='utf-8-sig')
    for metrica in ('accuracy', 'f1_macro'):
        print(f'\n{metrica}:')
        print(tabela.pivot(index='remocao', columns='modelo', values=metrica)
              .reindex([r for r, _ in variantes]).round(4).to_string())
    print(f'\nResultados salvos em {CAMINHO_SAIDA}')


if __name__ == '__main__':
    main()
