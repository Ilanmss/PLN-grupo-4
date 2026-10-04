# Validação cruzada agrupada (5 folds) dos baselines e dos modelos clássicos.
#
# Usa exatamente os mesmos folds do finetune_bertimbau.py --modo cv (GroupKFold por texto, semente 42),
# para que todos os modelos sejam comparados nos mesmos dados. Para cada modelo, salva as probabilidades
# out-of-fold (cada texto previsto pelo modelo que não o viu no treino) em modelos/cv/oof_<modelo>.csv.
# Modelos já concluídos são pulados (a execução pode ser retomada).
#
# Uso: python cv_classicos.py

import os
import time

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline

from src.data_loader import load_data
from src.divisoes import folds_cv
from src.modelos_classicos import complementnb_word12, logreg_word12, svc_word12, svc_word12_char25

CAMINHO_DADOS = 'data/train.xlsx'
PASTA_SAIDA = 'modelos/cv'
CLASSES = ['c1', 'c234', 'c5']

MODELOS = {
    # Baselines (mesmas configurações de baselines.py)
    'baseline_classe_majoritaria': lambda: DummyClassifier(strategy='most_frequent'),
    'baseline_bow_logreg': lambda: make_pipeline(
        CountVectorizer(), LogisticRegression(class_weight='balanced', max_iter=3000, random_state=123)),
    'baseline_tfidf_logreg': lambda: make_pipeline(
        TfidfVectorizer(), LogisticRegression(class_weight='balanced', max_iter=3000, random_state=123)),
    # Outros métodos
    'svc_word12': svc_word12,
    'svc_word12_char25': svc_word12_char25,
    'logreg_word12': logreg_word12,
    'complementnb_word12': complementnb_word12,
    'multinomialnb_word12': lambda: make_pipeline(TfidfVectorizer(ngram_range=(1, 2)), MultinomialNB(alpha=0.1)),
}


def main():
    os.makedirs(PASTA_SAIDA, exist_ok=True)
    data = load_data(CAMINHO_DADOS)
    X, y = data['resp_text'], data['clarity']
    folds = folds_cv(data)

    for nome, criar in MODELOS.items():
        caminho = os.path.join(PASTA_SAIDA, f'oof_{nome}.csv')
        if os.path.exists(caminho):
            print(f'{nome}: já concluído, pulando', flush=True)
            continue
        inicio, partes, f1s = time.time(), [], []
        for fold, (idx_tr, idx_val) in enumerate(folds, start=1):
            modelo = criar().fit(X.iloc[idx_tr], y.iloc[idx_tr])
            probs = pd.DataFrame(modelo.predict_proba(X.iloc[idx_val]), columns=[f'p_{c}' for c in modelo.classes_])
            probs = probs.reindex(columns=[f'p_{c}' for c in CLASSES], fill_value=0.0)
            probs.insert(0, 'indice', idx_val)
            probs.insert(1, 'fold', fold)
            probs['rotulo'] = y.iloc[idx_val].to_numpy()
            probs['previsto'] = np.array(CLASSES)[probs[[f'p_{c}' for c in CLASSES]].to_numpy().argmax(axis=1)]
            partes.append(probs)
            f1s.append(f1_score(probs['rotulo'], probs['previsto'], average='macro'))
        pd.concat(partes).sort_values('indice').to_csv(caminho, index=False)
        print(f'{nome:28s} F1 por fold {np.round(f1s, 4)} | média {np.mean(f1s):.4f} ± {np.std(f1s):.4f} '
              f'| {(time.time() - inicio) / 60:.1f} min', flush=True)


if __name__ == '__main__':
    main()
