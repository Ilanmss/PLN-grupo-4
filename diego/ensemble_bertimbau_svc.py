# Ensemble BERTimbau + LinearSVC por média ponderada das probabilidades.
#
# Usa a MESMA divisão do holdout do finetune_bertimbau.py (GroupKFold agrupado por texto, fold 1, semente 42):
#   - o LinearSVC é treinado nos mesmos 80% de treino e gera probabilidades para os mesmos 20% de validação;
#   - as probabilidades do BERTimbau vêm do arquivo holdout_probabilidades_epoca_N.csv.
#
# O peso do BERT (w) é escolhido com validação cruzada DENTRO do conjunto de validação (2 metades agrupadas):
# escolhe w em uma metade e mede na outra, e vice-versa. Assim o F1 reportado não fica otimista
# por ter escolhido o peso olhando os mesmos dados em que é medido.
#
# Exemplo:
#   python ensemble_bertimbau_svc.py --bert modelos/bertimbau/holdout_2ep/holdout_probabilidades_epoca_2.csv

import argparse
import json
import os

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.svm import LinearSVC

from src.data_loader import load_data
from src.deduplicacao import chave_texto

CAMINHO_DADOS = 'data/train.xlsx'
CLASSES = ['c1', 'c234', 'c5']
PESOS = np.round(np.arange(0, 1.0001, 0.05), 2)  # peso do BERT; (1 - w) é o peso do LinearSVC


def ler_argumentos():
    p = argparse.ArgumentParser(description='Ensemble BERTimbau + LinearSVC no holdout')
    p.add_argument('--bert', required=True, help='CSV de probabilidades do BERTimbau no holdout')
    p.add_argument('--saida', default='modelos/ensemble_bertimbau_svc')
    p.add_argument('--folds', type=int, default=5)
    p.add_argument('--semente', type=int, default=42)
    return p.parse_args()


def criar_svc():
    # Mesma configuração do melhor TF-IDF; calibração sigmoide para obter probabilidades
    return make_pipeline(
        TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True),
        CalibratedClassifierCV(LinearSVC(C=0.1), method='sigmoid', cv=5),
    )


def metricas(y_real, y_pred):
    return {'f1_macro': f1_score(y_real, y_pred, average='macro'),
            'accuracy': accuracy_score(y_real, y_pred),
            'kappa': cohen_kappa_score(y_real, y_pred)}


def prever(probs_bert, probs_svc, w):
    return (w * probs_bert + (1 - w) * probs_svc).argmax(axis=1)


def main():
    args = ler_argumentos()
    os.makedirs(args.saida, exist_ok=True)

    data = load_data(CAMINHO_DADOS)
    grupos = data['resp_text'].apply(chave_texto)
    idx_treino, idx_val = list(GroupKFold(n_splits=args.folds, shuffle=True, random_state=args.semente)
                               .split(data, groups=grupos))[0]

    bert = pd.read_csv(args.bert).set_index('indice').loc[idx_val]
    probs_bert = bert[[f'p_{c}' for c in CLASSES]].to_numpy()
    y_val = data['clarity'].iloc[idx_val].map({c: i for i, c in enumerate(CLASSES)}).to_numpy()
    assert (bert['rotulo'].to_numpy() == data['clarity'].iloc[idx_val].to_numpy()).all(), 'divisão diferente do holdout do BERT'

    print('Treinando LinearSVC calibrado no mesmo split...')
    svc = criar_svc().fit(data['resp_text'].iloc[idx_treino], data['clarity'].iloc[idx_treino])
    assert list(svc.classes_) == CLASSES
    probs_svc = svc.predict_proba(data['resp_text'].iloc[idx_val])

    # Curva completa (peso escolhido no próprio holdout = otimista; só para visualização)
    curva = pd.DataFrame([{'peso_bert': w, **metricas(y_val, prever(probs_bert, probs_svc, w))} for w in PESOS])

    # Estimativa honesta: escolhe w em uma metade do holdout e mede na outra (metades agrupadas por texto)
    grupos_val = grupos.iloc[idx_val].to_numpy()
    pred_honesto = np.zeros_like(y_val)
    pesos_escolhidos = []
    for a, b in GroupKFold(n_splits=2, shuffle=True, random_state=args.semente).split(probs_bert, groups=grupos_val):
        f1_a = [f1_score(y_val[a], prever(probs_bert[a], probs_svc[a], w), average='macro') for w in PESOS]
        w = PESOS[int(np.argmax(f1_a))]
        pesos_escolhidos.append(float(w))
        pred_honesto[b] = prever(probs_bert[b], probs_svc[b], w)

    resultados = pd.DataFrame([
        {'modelo': 'LinearSVC (calibrado)', **metricas(y_val, probs_svc.argmax(axis=1))},
        {'modelo': 'BERTimbau', **metricas(y_val, probs_bert.argmax(axis=1))},
        {'modelo': 'Ensemble média simples (w=0.5)', **metricas(y_val, prever(probs_bert, probs_svc, 0.5))},
        {'modelo': f'Ensemble peso escolhido em metades {pesos_escolhidos}', **metricas(y_val, pred_honesto)},
    ])

    peso_final = float(np.mean(pesos_escolhidos))
    curva.to_csv(os.path.join(args.saida, 'curva_pesos.csv'), index=False)
    resultados.to_csv(os.path.join(args.saida, 'resultados.csv'), index=False)
    with open(os.path.join(args.saida, 'config.json'), 'w', encoding='utf-8') as f:
        json.dump({'arquivo_bert': args.bert, 'peso_bert_recomendado': peso_final}, f, indent=2)

    texto = ('ENSEMBLE BERTIMBAU + LINEARSVC (holdout agrupado, mesmo split do BERT)\n' + '=' * 70 + '\n\n'
             f'Probabilidades do BERT: {args.bert}\n\n'
             + resultados.round(4).to_string(index=False)
             + f'\n\nPeso do BERT recomendado para o modelo final: {peso_final:.2f}\n\n'
             'Curva de pesos (escolher w olhando esta curva é otimista):\n'
             + curva.round(4).to_string(index=False) + '\n')
    with open(os.path.join(args.saida, 'relatorio.txt'), 'w', encoding='utf-8') as f:
        f.write(texto)
    print(texto)


if __name__ == '__main__':
    main()
