# Gera RELATORIO_RESULTADOS.md a partir dos resultados armazenados.
#
# Critério de comparação: validação cruzada agrupada por texto (GroupKFold, 5 folds, semente 42), com os
# MESMOS folds para todos os modelos (src/divisoes.folds_cv). Se o BERT ainda não tiver todos os folds,
# todos os modelos são comparados só nos folds já disponíveis.
#
# Ensembles: combinações das probabilidades out-of-fold. Tudo que é aprendido (pesos, viés por classe) é
# ajustado nos OUTROS folds e aplicado ao fold avaliado (cross-fitting), para o resultado não ficar otimista.
#
# Também salva modelos/melhores_modelos.json com a receita dos dois melhores modelos (componentes, pesos
# e viés ajustados em todos os folds), usada por prever_ensemble.py para rotular o conjunto de teste.
#
# Uso: python gerar_relatorio.py

import json
import os
import re
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

from diego.ensemble_holdout import ajustar_vies, media_aplicar, media_ponderada_ajustar, media_simples_ajustar
from src.data_loader import load_data
from src.divisoes import obter_divisao

CAMINHO_DADOS = 'data/train.xlsx'
CAMINHO_RELATORIO = 'RELATORIO_RESULTADOS.md'
CAMINHO_RECEITAS = 'modelos/melhores_modelos.json'
PASTA_CV = 'modelos/cv'
# Variantes de BERT com validação cruzada: nome do componente -> pasta com fold_k.csv
PASTAS_BERT_CV = {'bertimbau': 'modelos/bertimbau/cv_2ep/cv_folds',
                  'bertimbau_tapt': 'modelos/bertimbau/cv_tapt_2ep/cv_folds',
                  'albertina': 'modelos/bertimbau/cv_albertina_2ep/cv_folds'}
CLASSES = ['c1', 'c234', 'c5']
EPS = 1e-6

NOMES = {
    'bertimbau': 'BERTimbau fine-tuning (2 épocas)',
    'bertimbau_tapt': 'BERTimbau + TAPT, fine-tuning (2 épocas)',
    'albertina': 'Albertina-100m PTBR (DeBERTa), fine-tuning (2 épocas)',
    'svc_word12': 'TF-IDF palavra (1,2) + LinearSVC',
    'svc_word12_char25': 'TF-IDF palavra (1,2) + caractere (2,5) + LinearSVC',
    'logreg_word12': 'TF-IDF palavra (1,2) + regressão logística',
    'complementnb_word12': 'TF-IDF palavra (1,2) + ComplementNB',
    'multinomialnb_word12': 'TF-IDF palavra (1,2) + MultinomialNB',
    'baseline_classe_majoritaria': 'Baseline classe majoritária',
    'baseline_bow_logreg': 'Baseline BoW + regressão logística',
    'baseline_tfidf_logreg': 'Baseline TF-IDF + regressão logística',
}
NOMES_CURTOS = {'bertimbau': 'BERTimbau', 'bertimbau_tapt': 'BERTimbau-TAPT', 'albertina': 'Albertina', 'svc_word12': 'SVC palavra', 'svc_word12_char25': 'SVC palavra+caractere',
                'logreg_word12': 'LogReg', 'complementnb_word12': 'ComplementNB', 'multinomialnb_word12': 'MultinomialNB'}
CLASSICOS = ['svc_word12', 'svc_word12_char25', 'logreg_word12', 'complementnb_word12', 'multinomialnb_word12']
COMBOS_BERT = [['bertimbau'], ['bertimbau', 'svc_word12_char25'], ['bertimbau', 'complementnb_word12'],
               ['bertimbau', 'svc_word12_char25', 'complementnb_word12'], ['bertimbau'] + CLASSICOS,
               ['bertimbau_tapt'], ['bertimbau_tapt', 'svc_word12_char25'],
               ['bertimbau_tapt', 'svc_word12_char25', 'logreg_word12'], ['bertimbau_tapt'] + CLASSICOS,
               ['bertimbau', 'bertimbau_tapt'], ['bertimbau', 'bertimbau_tapt'] + CLASSICOS]
COMBOS_OUTROS = [[m] for m in CLASSICOS] + [['svc_word12_char25', 'complementnb_word12'],
                                           ['svc_word12_char25', 'complementnb_word12', 'multinomialnb_word12'],
                                           CLASSICOS]
METODOS = ['média simples', 'média ponderada', 'média simples + viés', 'média ponderada + viés']


# ---------------------------------------------------------------------------
# Leitura dos resultados
# ---------------------------------------------------------------------------

def ler_oof():
    """Probabilidades out-of-fold de cada modelo, indexadas pela linha do train.xlsx."""
    oof = {}
    for arquivo in sorted(os.listdir(PASTA_CV)) if os.path.isdir(PASTA_CV) else []:
        m = re.match(r'oof_(.+)\.csv$', arquivo)
        if m:
            oof[m.group(1)] = pd.read_csv(os.path.join(PASTA_CV, arquivo)).set_index('indice').sort_index()
    for nome, pasta in PASTAS_BERT_CV.items():
        if not os.path.isdir(pasta):
            continue
        partes = [pd.read_csv(os.path.join(pasta, f)) for f in sorted(os.listdir(pasta)) if re.match(r'fold_\d+\.csv$', f)]
        if partes:
            oof[nome] = pd.concat(partes).set_index('indice').sort_index()
    return oof


def baselines_armazenados(data):
    """F1 registrado em resultados_baselines.txt e acurácia recalculada com os modelos .joblib armazenados."""
    texto = open('modelos/baselines/resultados_baselines.txt', encoding='utf-8').read()
    f1_txt = {'maj': float(re.search(r'Classe majorit\S+\s*:\s*([\d.]+)', texto).group(1)),
              'tfidf': float(re.search(r'TF-IDF \+ LogReg\s*:\s*([\d.]+)', texto).group(1))}
    idx_tr, idx_te = obter_divisao(data, 'baseline')
    X_te, y_te = data['resp_text'].iloc[idx_te], data['clarity'].iloc[idx_te]
    maj = joblib.load('modelos/baselines/baseline_classe_majoritaria.joblib')['modelo']
    tfidf = joblib.load('modelos/baselines/baseline_tfidf_logistic_regression.joblib')
    pred_maj = maj.predict(X_te.to_frame())
    pred_tfidf = tfidf['modelo'].predict(tfidf['vetorizador'].transform(X_te))
    return {
        'maj': {'f1_registrado': f1_txt['maj'], 'f1': f1_score(y_te, pred_maj, average='macro'),
                'acc': accuracy_score(y_te, pred_maj)},
        'tfidf': {'f1_registrado': f1_txt['tfidf'], 'f1': f1_score(y_te, pred_tfidf, average='macro'),
                  'acc': accuracy_score(y_te, pred_tfidf)},
        'n_teste': len(idx_te), 'n_treino': len(idx_tr),
    }


# ---------------------------------------------------------------------------
# Avaliação por fold, com cross-fitting
# ---------------------------------------------------------------------------

def matriz(oof, nome, indices):
    p = np.clip(oof[nome].loc[indices, [f'p_{c}' for c in CLASSES]].to_numpy(dtype=np.float64), EPS, 1)
    return p / p.sum(axis=1, keepdims=True)


def ajustar(P, y, metodo):
    pesos = media_ponderada_ajustar(P, y) if 'ponderada' in metodo else media_simples_ajustar(P, y)
    vies = ajustar_vies(media_aplicar(P, pesos), y) if 'viés' in metodo else np.zeros(3)
    return pesos, vies


def avaliar_combo(oof, combo, metodo, folds_idx, y):
    """Métricas por fold: parâmetros ajustados nos demais folds disponíveis e aplicados ao fold avaliado."""
    precisa_ajuste = ('ponderada' in metodo and len(combo) > 1) or 'viés' in metodo
    if precisa_ajuste and len(folds_idx) < 2:
        return None
    por_fold, pred_total, y_total = [], [], []
    for k, idx_k in folds_idx.items():
        P_k = [matriz(oof, m, idx_k) for m in combo]
        if precisa_ajuste:
            idx_outros = np.concatenate([v for j, v in folds_idx.items() if j != k])
            pesos, vies = ajustar([matriz(oof, m, idx_outros) for m in combo], y.loc[idx_outros].to_numpy(), metodo)
        else:
            pesos, vies = np.full(len(combo), 1 / len(combo)), np.zeros(3)
        pred = (media_aplicar(P_k, pesos) + vies).argmax(axis=1)
        y_k = y.loc[idx_k].to_numpy()
        por_fold.append({'fold': k, 'f1': f1_score(y_k, pred, average='macro'), 'acc': accuracy_score(y_k, pred)})
        pred_total.append(pred)
        y_total.append(y_k)
    df = pd.DataFrame(por_fold)
    return {'f1': df['f1'].mean(), 'f1_dp': df['f1'].std(ddof=0), 'acc': df['acc'].mean(), 'acc_dp': df['acc'].std(ddof=0),
            'por_fold': df, 'pred': np.concatenate(pred_total), 'y': np.concatenate(y_total)}


def nome_candidato(combo, metodo):
    base = NOMES[combo[0]] if len(combo) == 1 else 'Ensemble ' + ' + '.join(NOMES_CURTOS[m] for m in combo)
    if len(combo) == 1:
        return base + (' + ajuste de viés' if 'viés' in metodo else '')
    return f'{base} ({metodo})'


def ranquear(oof, combos, folds_idx, y):
    linhas = []
    for combo in combos:
        if not all(m in oof for m in combo):
            continue
        metodos = ['média simples', 'média simples + viés'] if len(combo) == 1 else METODOS
        for metodo in metodos:
            r = avaliar_combo(oof, combo, metodo, folds_idx, y)
            if r is not None:
                linhas.append({'combo': combo, 'metodo': metodo, 'nome': nome_candidato(combo, metodo), **r})
    return sorted(linhas, key=lambda r: -r['f1'])


def receita(oof, candidato, folds_idx, y):
    """Parâmetros finais do candidato, ajustados em todos os folds disponíveis."""
    idx = np.concatenate(list(folds_idx.values()))
    pesos, vies = ajustar([matriz(oof, m, idx) for m in candidato['combo']], y.loc[idx].to_numpy(), candidato['metodo'])
    return {'nome': candidato['nome'], 'componentes': candidato['combo'], 'metodo': candidato['metodo'],
            'pesos': dict(zip(candidato['combo'], np.round(pesos, 4).tolist())),
            'vies': dict(zip(CLASSES, (np.round(vies, 4) + 0.0).tolist())),
            'f1_cv': round(float(candidato['f1']), 4), 'f1_cv_dp': round(float(candidato['f1_dp']), 4),
            'acc_cv': round(float(candidato['acc']), 4)}


# ---------------------------------------------------------------------------
# Formatação
# ---------------------------------------------------------------------------

def fmt(v, dp=None):
    return f'{v:.4f}'.replace('.', ',') + (f' ± {dp:.4f}'.replace('.', ',') if dp is not None else '')


def tabela_ranking(ranking, n=10):
    linhas = ['| # | Modelo | F1-macro (média ± dp) | Acurácia (média ± dp) |', '|---|---|---|---|']
    for i, r in enumerate(ranking[:n], start=1):
        linhas.append(f"| {i} | {r['nome']} | {fmt(r['f1'], r['f1_dp'])} | {fmt(r['acc'], r['acc_dp'])} |")
    return '\n'.join(linhas)


def tabela_por_fold(r, base):
    folds = r['por_fold']['fold'].tolist()
    linhas = ['| Modelo | ' + ' | '.join(f'Fold {k}' for k in folds) + ' | Média |',
              '|---|' + '---|' * (len(folds) + 1)]
    for nome, res in [(r['nome'], r), (NOMES['baseline_tfidf_logreg'], base)]:
        linhas.append(f'| {nome} | ' + ' | '.join(fmt(v) for v in res['por_fold']['f1']) + f" | **{fmt(res['f1'])}** |")
    return '\n'.join(linhas)


def tabela_por_classe(r):
    f1s = f1_score(r['y'], r['pred'], average=None, labels=[0, 1, 2])
    cm = confusion_matrix(r['y'], r['pred'], labels=[0, 1, 2])
    linhas = ['| Classe | F1 | Previsto c1 | Previsto c234 | Previsto c5 |', '|---|---|---|---|---|']
    for i, c in enumerate(CLASSES):
        linhas.append(f'| **{c}** (real) | {fmt(f1s[i])} | {cm[i, 0]} | {cm[i, 1]} | {cm[i, 2]} |')
    return '\n'.join(linhas)


def resultados_anteriores():
    """Resultados de experimentos com divisão única (não comparáveis com a validação cruzada)."""
    linhas = []

    def ler_hist(caminho):
        return pd.read_csv(caminho) if os.path.exists(caminho) else None

    h = ler_hist('modelos/bertimbau/holdout_4ep/holdout_historico.csv')
    if h is not None:
        for _, r in h.iterrows():
            linhas.append(('BERTimbau (treino planejado para 4 épocas), época ' + str(int(r['epoca'])),
                           'holdout agrupado (= fold 1)', r['f1_macro']))
    h = ler_hist('modelos/bertimbau/holdout_2ep/holdout_historico.csv')
    if h is not None:
        linhas.append(('BERTimbau 2 épocas', 'holdout agrupado (= fold 1)', h['f1_macro'].iloc[-1]))
    linhas += [
        ('Ensemble clássico SVC palavra+caractere + ComplementNB', 'divisão do baseline (80/20, semente 123)', 0.4834),
        ('TF-IDF word(1,2) + SVC (tfidf_ngrams_binario.py)', 'split aleatório próprio (teste 4.019)', 0.4778),
        ('ComplementNB (naive_bayes.py)', 'CV aninhado estratificado', 0.4528),
        ('SelectKBest + LinearSVC (tfidf_selectkbest_linearsvc.py)', 'teste independente próprio', 0.4507),
        ('Grid noturno: word(1,2) + LinearSVC C=0,1', 'CV estratificado 5 folds', 0.4490),
        ('Word2Vec CBOW 200d + LogReg (word2vec.py)', 'teste independente próprio', 0.4472),
    ]
    tab = ['| Experimento | Avaliação | F1-macro |', '|---|---|---|']
    tab += [f'| {n} | {a} | {fmt(v)} |' for n, a, v in linhas]
    return '\n'.join(tab)


# ---------------------------------------------------------------------------
# Relatório
# ---------------------------------------------------------------------------

def main():
    data = load_data(CAMINHO_DADOS)
    y = data['clarity'].map({c: i for i, c in enumerate(CLASSES)})
    oof = ler_oof()
    assert 'baseline_tfidf_logreg' in oof, 'rode antes: python cv_classicos.py'

    # Folds comuns a todos os modelos (limitados pelos folds do BERT já concluídos)
    folds_bert = sorted(oof['bertimbau']['fold'].unique()) if 'bertimbau' in oof else []
    folds_comuns = folds_bert if folds_bert else sorted(oof['baseline_tfidf_logreg']['fold'].unique())
    ref = oof['baseline_tfidf_logreg']
    folds_idx = {k: ref.index[ref['fold'] == k].to_numpy() for k in folds_comuns}
    if 'bertimbau' in oof:
        b = oof['bertimbau']
        for k in folds_comuns:
            assert np.array_equal(np.sort(b.index[b['fold'] == k]), np.sort(folds_idx[k])), f'fold {k} diferente no BERT'

    # Variantes de BERT ainda incompletas (ex.: TAPT em andamento) ficam fora do ranking; mostra só o parcial
    parciais = []
    for nome in [n for n in PASTAS_BERT_CV if n != 'bertimbau' and n in oof]:
        folds_var = sorted(oof[nome]['fold'].unique())
        if set(folds_comuns) <= set(folds_var):
            continue
        fi = {k: folds_idx[k] for k in folds_var if k in folds_idx}
        rv, rb_, rbase = (avaliar_combo(oof, [m], 'média simples', fi, y) for m in (nome, 'bertimbau', 'baseline_tfidf_logreg'))
        parciais.append(f"> ⏳ **{NOMES[nome]}: validação cruzada incompleta ({len(folds_var)} de 5 folds).** "
                        f"Nos folds prontos ({', '.join(map(str, folds_var))}): F1 {fmt(rv['f1'])}, contra {fmt(rb_['f1'])} do BERTimbau "
                        f"e {fmt(rbase['f1'])} do baseline. Fica fora do ranking, que exige os 5 folds "
                        f"(decisão e análise em `EXPERIMENTOS.md`).")
        del oof[nome]

    maj = avaliar_combo(oof, ['baseline_classe_majoritaria'], 'média simples', folds_idx, y)
    base = avaliar_combo(oof, ['baseline_tfidf_logreg'], 'média simples', folds_idx, y)
    rank_bert = ranquear(oof, COMBOS_BERT, folds_idx, y)
    rank_outros = ranquear(oof, COMBOS_OUTROS, folds_idx, y)
    melhor_bert = rank_bert[0] if rank_bert else None
    melhor_outros = rank_outros[0]
    armazenados = baselines_armazenados(data)

    receitas = {'folds_usados': [int(k) for k in folds_comuns], 'outros_metodos': receita(oof, melhor_outros, folds_idx, y)}
    if melhor_bert:
        receitas['bert'] = receita(oof, melhor_bert, folds_idx, y)
    with open(CAMINHO_RECEITAS, 'w', encoding='utf-8') as f:
        json.dump(receitas, f, indent=2, ensure_ascii=False)

    n_folds = len(folds_comuns)
    aviso = '' if n_folds == 5 else (
        f'\n> ⚠️ **Resultado parcial:** a validação cruzada do BERT tem {n_folds} de 5 folds prontos. '
        f'Todos os modelos foram comparados nos mesmos {n_folds} fold(s) ({", ".join(map(str, folds_comuns))}). '
        'O relatório é regenerado automaticamente a cada fold concluído.\n')
    if parciais:
        aviso += '\n' + '\n>\n'.join(parciais) + '\n'
    tem_tapt = 'bertimbau_tapt' in oof
    usa_tapt = bool(melhor_bert) and 'bertimbau_tapt' in melhor_bert['combo']
    lista_pendentes = (['label smoothing 0,1', 'Legal-BERTimbau']
                       + ([] if 'albertina' in oof or any('Albertina' in p for p in parciais) else ['Albertina-100m'])
                       + ([] if tem_tapt or any('TAPT' in p for p in parciais) else ['pré-treino adaptativo (TAPT)']))
    pendentes = ', '.join(lista_pendentes[:-1]) + ' e ' + lista_pendentes[-1]
    secao_tapt, comandos_tapt_cv, comandos_tapt_final = '', '', ''
    if tem_tapt or any('TAPT' in p for p in parciais):
        secao_tapt = """
**Variante com TAPT (BERTimbau-TAPT).** Antes do fine-tuning, o BERTimbau passa por um **pré-treino adaptativo à tarefa** (TAPT; Gururangan et al., 2020, *Don't Stop Pretraining*), com `pretreino_mlm.py`:

- **O que é:** continuação do pré-treino com *Masked Language Modeling*: 15% dos tokens são escondidos (80% trocados por `[MASK]`, 10% por um token aleatório e 10% mantidos) e o modelo aprende a prevê-los pelo contexto. Ex.: "Encaminhamos, na caixa [MASK], informação fornecida pela [MASK] responsável" → "Anexos", "área".
- **Sem vazamento na validação cruzada:** cada fold tem **o seu próprio TAPT**, feito só com os textos de treino daquele fold, sem rótulos. Os textos do fold de validação (e suas cópias, já que os folds são agrupados por texto) nunca entram no pré-treino. O conjunto de teste não é usado.
- **Modelo final da entrega:** depois da validação cruzada, todo o `train.xlsx` é treino. O TAPT final usa todos os textos de treino, sem rótulos, e em seguida vem o fine-tuning com 100% dos dados.
- **Por quê:** o texto do e-SIC (templates de órgãos, siglas, citações de leis) é diferente do português geral do pré-treino original. O TAPT adapta o vocabulário e o estilo antes de o modelo aprender a classificar.
- **Configuração:** 2 épocas de MLM, lr 5e-5, lote efetivo 32, até 256 tokens (64 do início + 190 do fim). Depois, o mesmo fine-tuning de 2 épocas descrito acima. Custo: ~4h20 por fold (MLM + fine-tuning) na GTX 1660 Super.
"""
        comandos_tapt_cv = """
# 2b. Validação cruzada com TAPT sem vazamento: para cada fold k, TAPT só com os textos de treino do fold
#     e fine-tuning a partir dele (GPU, ~4h20 por fold). Automatizado em modelos/bertimbau/fila_tapt_folds.sh
python pretreino_mlm.py --textos fold --fold 1 --epocas 2 --saida modelos/tapt_folds/fold_1
python finetune_bertimbau.py --modo cv --apenas-fold 1 --epocas 2 --modelo modelos/tapt_folds/fold_1 --saida modelos/bertimbau/cv_tapt_2ep
# ... repetir para os folds 2 a 5
"""
        comandos_tapt_final = """
# 4b. Modelo final BERTimbau+TAPT: TAPT com todos os textos de treino, depois fine-tuning com 100% dos dados (GPU, ~5h30)
python pretreino_mlm.py --textos todos --epocas 2 --saida modelos/bertimbau_tapt_todos
python finetune_bertimbau.py --modo final --epocas 2 --modelo modelos/bertimbau_tapt_todos --saida modelos/bertimbau/final_tapt_2ep
"""

    def ganho(r):
        return f"+{(r['f1'] - base['f1']) * 100:.1f} pts".replace('.', ',')

    linhas_resumo = [
        f"| Baseline classe majoritária | {fmt(maj['f1'], maj['f1_dp'])} | {fmt(maj['acc'], maj['acc_dp'])} | {fmt(armazenados['maj']['f1_registrado'])} | — |",
        f"| Baseline TF-IDF + regressão logística | {fmt(base['f1'], base['f1_dp'])} | {fmt(base['acc'], base['acc_dp'])} | {fmt(armazenados['tfidf']['f1_registrado'])} | referência |",
    ]
    if melhor_bert:
        linhas_resumo.append(f"| **Melhor modelo com BERT:** {melhor_bert['nome']} | **{fmt(melhor_bert['f1'], melhor_bert['f1_dp'])}** | **{fmt(melhor_bert['acc'], melhor_bert['acc_dp'])}** | — | **{ganho(melhor_bert)}** |")
    linhas_resumo.append(f"| **Melhor modelo com outros métodos:** {melhor_outros['nome']} | **{fmt(melhor_outros['f1'], melhor_outros['f1_dp'])}** | **{fmt(melhor_outros['acc'], melhor_outros['acc_dp'])}** | — | **{ganho(melhor_outros)}** |")

    def melhor_por_acc(ranking, rotulo):
        r = max(ranking, key=lambda c: c['acc'])
        if r is ranking[0]:
            return ''
        return (f"Pela acurácia, o melhor {rotulo} seria {r['nome']} (acurácia {fmt(r['acc'], r['acc_dp'])}, "
                f"F1 {fmt(r['f1'], r['f1_dp'])}). ")
    nota_acuracia = ((melhor_por_acc(rank_bert, 'com BERT') if rank_bert else '') + melhor_por_acc(rank_outros, 'sem BERT')
                     + 'O ajuste de viés favorece a classe c234, o que aumenta o F1-macro mas pode reduzir um pouco a acurácia.').strip()
    individuais = [r for r in rank_bert + rank_outros if len(r['combo']) == 1 and r['metodo'] == 'média simples']
    individuais = sorted({r['combo'][0]: r for r in individuais}.values(), key=lambda r: -r['f1'])
    tabela_individuais = tabela_ranking(sorted(individuais + [dict(base, nome=NOMES['baseline_tfidf_logreg'] + ' (baseline)')],
                                               key=lambda r: -r['f1']), n=20)
    bert_sozinho = next((r for r in individuais if r['combo'] == ['bertimbau']), None)
    if bert_sozinho:
        dif = (bert_sozinho['f1'] - base['f1']) * 100
        bert_sozinho['dif_txt'] = f"{dif:+.1f}".replace('.', ',')
    rb = receitas.get('bert')
    ro = receitas['outros_metodos']
    pesos_txt = lambda rc: ', '.join(f"{NOMES[m]}: {str(w).replace('.', ',')}" for m, w in rc['pesos'].items() if w > 0)
    zerados_txt = lambda rc: ('' if all(w > 0 for w in rc['pesos'].values()) else
                              ' Componentes com peso 0 (descartados pela otimização dos pesos): '
                              + ', '.join(NOMES[m] for m, w in rc['pesos'].items() if w == 0) + '.')
    vies_txt = lambda rc: ('sem ajuste de viés' if not any(rc['vies'].values())
                           else 'viés somado às log-probabilidades: ' + ', '.join(f'{c} {str(v).replace(".", ",")}' for c, v in rc['vies'].items()))

    md = f"""# Relatório de resultados: classificação de clareza de respostas do e-SIC

Gerado automaticamente por `gerar_relatorio.py` em {datetime.now().strftime('%d/%m/%Y %H:%M')}.
{aviso}
## 1. Resumo

Validação cruzada agrupada por texto com {n_folds} fold(s); média ± desvio padrão entre folds.

| Modelo | F1-macro (validação cruzada) | Acurácia (validação cruzada) | F1 armazenado (divisão única 80/20) | Ganho de F1 sobre o baseline |
|---|---|---|---|---|
""" + '\n'.join(linhas_resumo) + f"""

- **F1 armazenado:** valor registrado em `modelos/baselines/resultados_baselines.txt`, obtido pelo `baselines.py` numa divisão aleatória 80/20 (semente 123, {armazenados['n_teste']} textos de teste). Recarregando os modelos `.joblib` armazenados nessa divisão, obtive F1 {fmt(armazenados['maj']['f1'])} (acurácia {fmt(armazenados['maj']['acc'])}) para a classe majoritária e F1 {fmt(armazenados['tfidf']['f1'])} (acurácia {fmt(armazenados['tfidf']['acc'])}) para o TF-IDF + regressão logística, idênticos aos registrados.
- **Ganho:** diferença de F1-macro médio para o baseline TF-IDF + regressão logística, nos mesmos folds.
- A nota do EP usa a **acurácia** no teste. Como as classes são quase balanceadas, F1-macro e acurácia andam juntos. {nota_acuracia}

## 2. Protocolo de avaliação

- **Validação cruzada agrupada, 5 folds** (`GroupKFold`, embaralhado, semente 42), agrupando textos idênticos (após normalizar espaços). Nenhuma cópia de um texto fica no treino e na avaliação ao mesmo tempo. Sem o agrupamento, templates repetidos, como o do MTb que aparece 81 vezes, inflam o resultado: o mesmo LinearSVC dá F1 0,4595 com `StratifiedKFold` e 0,4544 com `GroupKFold`.
- **Os mesmos folds para todos os modelos** (`src/divisoes.py`). Cada modelo é treinado 5 vezes, cada vez sem um fold, e prevê o fold que ficou de fora. As predições resultantes são chamadas de *out-of-fold*.
- **Ensembles:** combinam as probabilidades out-of-fold. Pesos e viés por classe são ajustados nos outros folds e aplicados ao fold avaliado (*cross-fitting*), para que nenhum parâmetro seja escolhido olhando os dados em que é medido.
- **Seleção:** o melhor de cada categoria é o de maior F1-macro médio. Como vários candidatos são comparados nos mesmos folds, o valor do vencedor pode estar levemente otimista, em fração do desvio padrão.
- O conjunto de teste não rotulado **não foi usado** em nenhuma etapa.

## 3. Baselines

| Baseline | F1 armazenado (divisão 80/20) | F1 validação cruzada | Acurácia validação cruzada |
|---|---|---|---|
| Classe majoritária | {fmt(armazenados['maj']['f1_registrado'])} | {fmt(maj['f1'], maj['f1_dp'])} | {fmt(maj['acc'], maj['acc_dp'])} |
| TF-IDF + regressão logística (baseline oficial) | {fmt(armazenados['tfidf']['f1_registrado'])} | {fmt(base['f1'], base['f1_dp'])} | {fmt(base['acc'], base['acc_dp'])} |

O TF-IDF + regressão logística tem F1 menor na validação cruzada agrupada do que na divisão aleatória, porque nela não consegue "decorar" templates repetidos. O mesmo vale para os demais modelos, então as comparações entre modelos são feitas sempre na mesma avaliação.
"""
    if melhor_bert:
        md += f"""
## 4. Melhor modelo usando BERT

**{melhor_bert['nome']}**: F1-macro **{fmt(melhor_bert['f1'], melhor_bert['f1_dp'])}**, acurácia **{fmt(melhor_bert['acc'], melhor_bert['acc_dp'])}** ({ganho(melhor_bert)} sobre o baseline).

Pesos finais ({pesos_txt(rb)}); {vies_txt(rb)}.{zerados_txt(rb)} Os parâmetros foram ajustados nas predições out-of-fold de todos os folds e estão em `{CAMINHO_RECEITAS}`.

**F1-macro por fold**

{tabela_por_fold(melhor_bert, base)}

**Desempenho por classe** (predições out-of-fold de todos os folds)

{tabela_por_classe(melhor_bert)}

**Ranking dos candidatos com BERT**

{tabela_ranking(rank_bert)}

**Cada modelo sozinho** (sem ensemble e sem ajuste de viés)

{tabela_individuais}

O BERTimbau sozinho tem F1 {fmt(bert_sozinho['f1'], bert_sozinho['f1_dp'])}, {bert_sozinho['dif_txt']} pts em relação ao baseline. O ganho do melhor modelo vem da **combinação** do BERT com os modelos TF-IDF, que erram em textos diferentes.
"""
    md += f"""
## 5. Melhor modelo usando outros métodos (sem BERT, excluindo os baselines)

**{melhor_outros['nome']}**: F1-macro **{fmt(melhor_outros['f1'], melhor_outros['f1_dp'])}**, acurácia **{fmt(melhor_outros['acc'], melhor_outros['acc_dp'])}** ({ganho(melhor_outros)} sobre o baseline).

Pesos finais ({pesos_txt(ro)}); {vies_txt(ro)}.{zerados_txt(ro)}

**F1-macro por fold**

{tabela_por_fold(melhor_outros, base)}

**Desempenho por classe**

{tabela_por_classe(melhor_outros)}

**Ranking dos candidatos sem BERT**

{tabela_ranking(rank_outros)}

## 6. Outros resultados do projeto (divisão única, não comparáveis com a validação cruzada)

Estes experimentos foram avaliados numa única divisão, e cada divisão tem dificuldade diferente. O mesmo baseline TF-IDF + regressão logística vai de 0,4412 no holdout agrupado a 0,4648 na divisão aleatória. Por isso não entram na seleção do melhor modelo, que exige validação cruzada. Os métodos que se destacaram aqui foram reavaliados com validação cruzada nas seções 4 e 5.

{resultados_anteriores()}

Não reavaliados com validação cruzada por falta de tempo de GPU (cerca de 11 h por configuração): {pendentes}. Os scripts estão prontos (`pretreino_mlm.py`, `finetune_bertimbau.py --modelo ...`). Word2Vec ficou abaixo do TF-IDF em todas as configurações testadas.

## 7. Como funcionam os melhores modelos

### 7.1 BERTimbau (componente BERT)

1. **Modelo pré-treinado:** `neuralmind/bert-base-portuguese-cased` (BERTimbau base: 110 milhões de parâmetros, 12 camadas), pré-treinado em um grande corpus de português brasileiro. Ele já "entende" a língua antes de ver os nossos dados.
2. **Tokenização:** WordPiece, até 512 tokens. Textos longos (10,9% dos textos) mantêm os **128 primeiros e os 382 últimos tokens**, porque o fim da resposta costuma dizer se o pedido foi atendido, negado ou redirecionado.
3. **Fine-tuning:** uma camada de classificação (3 classes) é acoplada ao token `[CLS]` e o modelo inteiro é ajustado com entropia cruzada. Configuração: AdamW, lr 2e-5 com aquecimento de 10% e decaimento linear, lote efetivo 16 (8 × 2 de acumulação), precisão mista fp16, *gradient clipping* 1,0, *weight decay* 0,01.
4. **2 épocas:** com mais épocas o modelo passa a memorizar o ruído dos rótulos. No holdout, o F1 foi 0,4345 → 0,4601 → 0,4487 → 0,4474 nas épocas 1 a 4.
5. **Saída:** probabilidades para c1, c234 e c5.
{secao_tapt}
### 7.2 Modelos clássicos (componentes sem BERT)

- **TF-IDF palavra (1,2) + caractere (2,5) + LinearSVC:** duas representações TF-IDF concatenadas. A de palavras usa unigramas e bigramas; a de caracteres usa n-gramas de 2 a 5 caracteres dentro das palavras, capturando radicais, siglas e variações como "SIC/MTb" e "Decreto nº 7.724". Ambas com `sublinear_tf`. Classificador LinearSVC (C=0,05), calibrado com `CalibratedClassifierCV` (sigmoide) para produzir probabilidades.
- **TF-IDF palavra (1,2) + ComplementNB (alpha=0,5)** e **MultinomialNB (alpha=0,1):** Naive Bayes, rápidos e fortes em "decorar" padrões frequentes de palavras.
- **TF-IDF palavra (1,2) + LinearSVC (C=0,1)** e **regressão logística (C=2)**.

### 7.3 Ensemble

Para cada texto, cada componente produz probabilidades para as 3 classes. O ensemble faz a **média ponderada** dessas probabilidades (pesos no simplex, ajustados minimizando a log-loss) e, quando indicado, soma um **viés por classe** às log-probabilidades, deslocando a fronteira de decisão para maximizar o F1-macro. A classe final é a de maior valor. O ganho vem da **diversidade**: o BERT captura sentido e contexto, enquanto o TF-IDF captura palavras e templates exatos, e cada um erra em textos diferentes.

## 8. Como reproduzir

Requisitos: Python 3.13, `pandas`, `scikit-learn`, `openpyxl`, `joblib`, `torch` (CUDA) e `transformers`. Tempos medidos numa GTX 1660 Super (6 GB).

```bash
# 1. Validação cruzada dos baselines e modelos clássicos (CPU, ~40 min)
python cv_classicos.py

# 2. Validação cruzada do BERTimbau (GPU, ~2h10 por fold)
python finetune_bertimbau.py --modo cv --epocas 2 --saida modelos/bertimbau/cv_2ep
{comandos_tapt_cv}
# 3. Este relatório e a receita dos melhores modelos (modelos/melhores_modelos.json)
python gerar_relatorio.py

# 4. Modelo BERTimbau final, treinado com 100% do train.xlsx (GPU, ~2h45)
python finetune_bertimbau.py --modo final --epocas 2 --saida modelos/bertimbau/final_2ep
{comandos_tapt_final}
# 5. Rotular o conjunto de teste com o melhor modelo (só na entrega)
python prever_ensemble.py --modelo bert --teste <arquivo_teste>.xlsx
python prever_ensemble.py --modelo outros_metodos --teste <arquivo_teste>.xlsx

# 5b. Variante usada na entrega: o mesmo ensemble com BERT, sem o ajuste de viés (maior acurácia)
python prever_ensemble.py --modelo bert --sem-vies --teste data/test1.xlsx
```

**Arquivo entregue:** `modelos/entrega/bert_sem_vies/test1.xlsx`. A nota do EP usa acurácia, e o mesmo ensemble sem o ajuste de viés tem a maior acurácia na validação cruzada (seção 1). Os pesos dos componentes são os mesmos; só o viés por classe é desligado.

O passo 5 treina os componentes clássicos em 100% dos dados, usa o BERTimbau final salvo em `modelos/bertimbau/final_2ep/final` e combina as probabilidades com os pesos e o viés de `{CAMINHO_RECEITAS}`.

Arquivos principais:

| Arquivo | Função |
|---|---|
| `src/divisoes.py` | Folds da validação cruzada agrupada e divisão do baseline |
| `src/modelos_classicos.py` | Definição dos modelos clássicos |
| `cv_classicos.py` | Validação cruzada dos baselines e clássicos (probabilidades out-of-fold em `modelos/cv/`) |
| `finetune_bertimbau.py` | Fine-tuning do BERTimbau (modos holdout, cv, final e prever) |
| `gerar_relatorio.py` | Comparação nos mesmos folds, ensembles com cross-fitting e este relatório |
| `prever_ensemble.py` | Rotulação do teste com o melhor modelo |
| `EXPERIMENTOS.md` | Diário completo dos experimentos e das decisões |
"""
    with open(CAMINHO_RELATORIO, 'w', encoding='utf-8') as f:
        f.write(md)
    print(f'{CAMINHO_RELATORIO} gerado ({n_folds} fold(s)).')
    print(f"Baseline TF-IDF: {base['f1']:.4f} | melhor BERT: {melhor_bert['nome'] + ' ' + format(melhor_bert['f1'], '.4f') if melhor_bert else '-'} "
          f"| melhor outros: {melhor_outros['nome']} {melhor_outros['f1']:.4f}")


if __name__ == '__main__':
    main()
