# Ensemble geral de modelos no holdout agrupado (mesmo split do BERTimbau e dos clássicos).
#
# Recebe N arquivos de probabilidades (formato: indice, p_c1, p_c234, p_c5, rotulo) e compara:
#   1. cada modelo sozinho;
#   2. média simples das probabilidades;
#   3. média ponderada com pesos otimizados (minimizando log-loss);
#   4. stacking: regressão logística sobre as log-probabilidades de todos os modelos;
#   5. os métodos 2-4 + ajuste de viés por classe (desloca a fronteira de decisão para maximizar o F1-macro).
#
# Avaliação honesta: tudo que é "aprendido" (pesos, stacking, viés) é ajustado em uma metade do holdout
# e medido na outra (metades agrupadas por texto), trocando os papéis, e repetindo com 5 sorteios de metades.
# Os parâmetros finais (para o conjunto de teste) são ajustados no holdout inteiro e salvos em config.json.
#
# Uso:
#   python ensemble_holdout.py --modelos bert=modelos/bertimbau/holdout_2ep/holdout_probabilidades_epoca_2.csv \
#       svc=modelos/holdout_probabilidades/svc_word12.csv --saida modelos/ensemble/bert_svc

import argparse
import itertools
import json
import os

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score, log_loss
from sklearn.model_selection import GroupKFold

from src.data_loader import load_data
from src.deduplicacao import chave_texto

CAMINHO_DADOS = 'data/train.xlsx'
CLASSES = ['c1', 'c234', 'c5']
N_REPETICOES = 5
GRADE_VIES = np.round(np.arange(-0.4, 0.4001, 0.05), 2)
EPS = 1e-6


def ler_argumentos():
    p = argparse.ArgumentParser(description='Ensemble de modelos no holdout agrupado')
    p.add_argument('--modelos', nargs='+', required=True, help='nome=caminho_do_csv de probabilidades')
    p.add_argument('--saida', required=True)
    return p.parse_args()


def carregar(modelos, data):
    """Lê os CSVs e alinha todos pelo índice do train.xlsx."""
    probs, indices = {}, None
    for item in modelos:
        nome, caminho = item.split('=', 1)
        df = pd.read_csv(caminho).set_index('indice').sort_index()
        if indices is None:
            indices = df.index.to_numpy()
        assert np.array_equal(df.index.to_numpy(), indices), f'{nome}: holdout diferente dos demais'
        p = np.clip(df[[f'p_{c}' for c in CLASSES]].to_numpy(dtype=np.float64), EPS, 1)
        probs[nome] = p / p.sum(axis=1, keepdims=True)  # renormaliza (CSVs em float32 não somam 1 exatamente)
    y = data['clarity'].iloc[indices].map({c: i for i, c in enumerate(CLASSES)}).to_numpy()
    grupos = data['resp_text'].iloc[indices].apply(chave_texto).to_numpy()
    return probs, y, grupos, indices


def metricas(y, pred):
    return {'f1_macro': f1_score(y, pred, average='macro'), 'accuracy': accuracy_score(y, pred),
            'kappa': cohen_kappa_score(y, pred)}


# ---------------------------------------------------------------------------
# Combinadores: cada um tem ajustar(P, y) -> params e aplicar(P, params) -> log-probabilidades
# P é uma lista de matrizes (n x 3), uma por modelo.
# ---------------------------------------------------------------------------

def media_aplicar(P, pesos):
    return np.log(sum(w * p for w, p in zip(pesos, P)))


def media_simples_ajustar(P, y):
    return np.full(len(P), 1 / len(P))


def media_ponderada_ajustar(P, y):
    """Pesos no simplex (via softmax) que minimizam a log-loss."""
    if len(P) == 1:
        return np.ones(1)

    def perda(theta):
        w = np.exp(theta) / np.exp(theta).sum()
        return log_loss(y, np.exp(media_aplicar(P, w)), labels=[0, 1, 2])

    theta = minimize(perda, np.zeros(len(P)), method='Nelder-Mead', options={'maxiter': 2000}).x
    return np.exp(theta) / np.exp(theta).sum()


def stacking_ajustar(P, y):
    return LogisticRegression(C=1.0, max_iter=2000).fit(np.hstack([np.log(p) for p in P]), y)


def stacking_aplicar(P, modelo):
    return modelo.predict_log_proba(np.hstack([np.log(p) for p in P]))


COMBINADORES = {
    'média simples': (media_simples_ajustar, media_aplicar),
    'média ponderada': (media_ponderada_ajustar, media_aplicar),
    'stacking (LogReg)': (stacking_ajustar, stacking_aplicar),
}


def ajustar_vies(logp, y):
    """Viés somado às log-probabilidades de c1 e c5 (c234 fica em 0) que maximiza o F1-macro."""
    melhor, melhor_f1 = np.zeros(3), -1
    for b1, b5 in itertools.product(GRADE_VIES, GRADE_VIES):
        vies = np.array([b1, 0.0, b5])
        f1 = f1_score(y, (logp + vies).argmax(axis=1), average='macro')
        if f1 > melhor_f1 + 1e-9:
            melhor, melhor_f1 = vies, f1
    return melhor


def avaliar(P, y, grupos, ajustar, aplicar, com_vies):
    """F1/acc médios em validação cruzada 2-fold agrupada dentro do holdout, repetida N vezes."""
    resultados = []
    for rep in range(N_REPETICOES):
        pred = np.zeros_like(y)
        gkf = GroupKFold(n_splits=2, shuffle=True, random_state=rep)
        for a, b in gkf.split(P[0], groups=grupos):
            params = ajustar([p[a] for p in P], y[a])
            logp_b = aplicar([p[b] for p in P], params)
            if com_vies:
                vies = ajustar_vies(aplicar([p[a] for p in P], params), y[a])
                logp_b = logp_b + vies
            pred[b] = logp_b.argmax(axis=1)
        resultados.append(metricas(y, pred))
    return pd.DataFrame(resultados).mean().to_dict(), pd.DataFrame(resultados)['f1_macro'].std(ddof=0)


def main():
    args = ler_argumentos()
    os.makedirs(args.saida, exist_ok=True)
    data = load_data(CAMINHO_DADOS)
    probs, y, grupos, _ = carregar(args.modelos, data)
    nomes = list(probs)
    P = [probs[n] for n in nomes]

    linhas = []
    for nome in nomes:
        linhas.append({'método': f'sozinho: {nome}', **metricas(y, probs[nome].argmax(axis=1)), 'f1_dp': 0.0})
        m, dp = avaliar([probs[nome]], y, grupos, media_simples_ajustar, media_aplicar, com_vies=True)
        linhas.append({'método': f'sozinho: {nome} + viés', **m, 'f1_dp': dp})

    if len(P) > 1:
        for nome_comb, (ajustar, aplicar) in COMBINADORES.items():
            for com_vies in (False, True):
                m, dp = avaliar(P, y, grupos, ajustar, aplicar, com_vies)
                linhas.append({'método': nome_comb + (' + viés' if com_vies else ''), **m, 'f1_dp': dp})

    resultados = pd.DataFrame(linhas)
    resultados.to_csv(os.path.join(args.saida, 'resultados.csv'), index=False)

    # Parâmetros finais ajustados no holdout inteiro (para aplicar nas probabilidades do teste)
    pesos = media_ponderada_ajustar(P, y)
    vies = ajustar_vies(media_aplicar(P, pesos), y)
    config = {'modelos': dict(item.split('=', 1) for item in args.modelos),
              'pesos_media_ponderada': dict(zip(nomes, np.round(pesos, 4).tolist())),
              'vies_media_ponderada': dict(zip(CLASSES, vies.tolist()))}
    with open(os.path.join(args.saida, 'config.json'), 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    texto = ('ENSEMBLE NO HOLDOUT AGRUPADO\n' + '=' * 70 + '\n'
             'Métodos com parâmetros aprendidos: medidos com 2-fold agrupado dentro do holdout, '
             f'repetido {N_REPETICOES}x (média; f1_dp = desvio entre repetições).\n'
             '"sozinho: X" (sem viés) não aprende nada: é o desempenho direto do modelo.\n\n'
             + resultados.round(4).to_string(index=False)
             + '\n\nParâmetros finais (ajustados no holdout inteiro):\n' + json.dumps(config, indent=2, ensure_ascii=False) + '\n')
    with open(os.path.join(args.saida, 'relatorio.txt'), 'w', encoding='utf-8') as f:
        f.write(texto)
    print(texto)


if __name__ == '__main__':
    main()
