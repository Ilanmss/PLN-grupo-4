# Probabilidades dos modelos clássicos (TF-IDF) no MESMO holdout do BERTimbau.
#
# Gera um CSV por modelo em modelos/holdout_probabilidades/, no mesmo formato do finetune_bertimbau.py
# (indice, p_c1, p_c234, p_c5, rotulo, previsto), para serem combinados em ensemble_holdout.py.
# Divisões (src/divisoes.py): 'agrupada' (padrão, mesmo holdout do BERTimbau) ou 'baseline' (mesma do baselines.py).
#
# Uso: python classicos_holdout.py [--divisao agrupada|baseline]

import argparse
import os

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

from src.data_loader import load_data
from src.divisoes import DESCRICAO, obter_divisao
from src.modelos_classicos import complementnb_word12, logreg_word12, svc_word12, svc_word12_char25

CAMINHO_DADOS = 'data/train.xlsx'
PASTAS_SAIDA = {'agrupada': 'modelos/holdout_probabilidades', 'baseline': 'modelos/probabilidades_split_baseline'}
CLASSES = ['c1', 'c234', 'c5']


MODELOS = {
    'svc_word12': svc_word12,
    'svc_word12_char25': svc_word12_char25,
    'logreg_word12': logreg_word12,
    'complementnb_word12': complementnb_word12,
}


def main():
    p = argparse.ArgumentParser(description='Probabilidades dos modelos clássicos em uma divisão de avaliação')
    p.add_argument('--divisao', choices=list(PASTAS_SAIDA), default='agrupada')
    args = p.parse_args()
    PASTA_SAIDA = PASTAS_SAIDA[args.divisao]
    os.makedirs(PASTA_SAIDA, exist_ok=True)
    data = load_data(CAMINHO_DADOS)
    idx_treino, idx_val = obter_divisao(data, args.divisao)
    print(f'Divisão: {DESCRICAO[args.divisao]}')
    X_tr, y_tr = data['resp_text'].iloc[idx_treino], data['clarity'].iloc[idx_treino]
    X_val, y_val = data['resp_text'].iloc[idx_val], data['clarity'].iloc[idx_val]

    linhas = []
    for nome, criar in MODELOS.items():
        modelo = criar().fit(X_tr, y_tr)
        assert list(modelo.classes_) == CLASSES
        probs = modelo.predict_proba(X_val)
        pred = np.array(CLASSES)[probs.argmax(axis=1)]

        df = pd.DataFrame(probs, columns=[f'p_{c}' for c in CLASSES])
        df.insert(0, 'indice', idx_val)
        df['rotulo'] = y_val.to_numpy()
        df['previsto'] = pred
        df.to_csv(os.path.join(PASTA_SAIDA, f'{nome}.csv'), index=False)

        linhas.append({'modelo': nome, 'f1_macro': f1_score(y_val, pred, average='macro'),
                       'accuracy': accuracy_score(y_val, pred)})
        print(f'{nome:22s} F1={linhas[-1]["f1_macro"]:.4f} Acc={linhas[-1]["accuracy"]:.4f}', flush=True)

    pd.DataFrame(linhas).to_csv(os.path.join(PASTA_SAIDA, 'resumo_classicos.csv'), index=False)


if __name__ == '__main__':
    main()
