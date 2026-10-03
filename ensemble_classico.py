# Ensemble clássico: TF-IDF palavra+caractere + LinearSVC calibrado  +  TF-IDF palavra + ComplementNB
# (média simples das probabilidades). Definição em src/modelos_classicos.py.
#
# O script:
#   1. treina e avalia na MESMA divisão do baseline oficial (baselines.py: aleatória 80/20, estratificada,
#      semente 123), junto com o baseline, para comparação direta;
#   2. avalia também no holdout agrupado por texto (mesmo split do BERTimbau), mais rigoroso;
#   3. salva o modelo avaliado, recarrega do disco e confere se as predições são idênticas;
#   4. treina o modelo final com 100% do train.xlsx e salva (é esse que rotula o teste na entrega);
#   5. opcionalmente (--teste), rotula um arquivo com a coluna resp_text.
#
# Uso:
#   python ensemble_classico.py
#   python ensemble_classico.py --teste data/test.xlsx   # só na entrega final

import argparse
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, cohen_kappa_score, confusion_matrix, f1_score
from sklearn.model_selection import GroupKFold, train_test_split

from src.data_loader import load_data
from src.deduplicacao import chave_texto
from src.modelos_classicos import ensemble_svc_cnb

CAMINHO_DADOS = 'data/train.xlsx'
PASTA_SAIDA = 'modelos/ensemble_classico'
CLASSES = ['c1', 'c234', 'c5']


def metricas(y, pred):
    return {'f1_macro': f1_score(y, pred, average='macro'), 'accuracy': accuracy_score(y, pred),
            'kappa': cohen_kappa_score(y, pred)}


def baseline_oficial():
    # Idêntico ao baseline TF-IDF + LogReg de baselines.py
    from sklearn.pipeline import make_pipeline
    return make_pipeline(TfidfVectorizer(), LogisticRegression(class_weight='balanced', max_iter=3000, random_state=123))


def avaliar(nome_divisao, X_tr, y_tr, X_te, y_te):
    """Treina baseline e ensemble na mesma divisão; devolve (tabela, texto do relatório, ensemble treinado, predições)."""
    print(f'[{nome_divisao}] treinando baseline e ensemble...', flush=True)
    base = baseline_oficial().fit(X_tr, y_tr)
    ens = ensemble_svc_cnb().fit(X_tr, y_tr)

    pred_base = base.predict(X_te)
    probs_comp = [m.predict_proba(X_te) for m in ens.modelos_]
    pred_comp = [ens.classes_[p.argmax(axis=1)] for p in probs_comp]
    pred_ens = ens.predict(X_te)

    tabela = pd.DataFrame([
        {'divisão': nome_divisao, 'modelo': 'Baseline oficial (TF-IDF + LogReg)', **metricas(y_te, pred_base)},
        {'divisão': nome_divisao, 'modelo': 'SVC palavra+caractere (componente)', **metricas(y_te, pred_comp[0])},
        {'divisão': nome_divisao, 'modelo': 'ComplementNB (componente)', **metricas(y_te, pred_comp[1])},
        {'divisão': nome_divisao, 'modelo': 'Ensemble SVC + ComplementNB', **metricas(y_te, pred_ens)},
    ])
    texto = (f'--- {nome_divisao}: {len(X_tr)} treino / {len(X_te)} teste ---\n\n'
             + tabela.drop(columns='divisão').round(4).to_string(index=False) + '\n\n'
             + 'Ensemble - relatório por classe:\n'
             + classification_report(y_te, pred_ens, labels=CLASSES, digits=4)
             + 'Matriz de confusão (linhas = real, colunas = previsto), ordem ' + str(CLASSES) + ':\n'
             + str(confusion_matrix(y_te, pred_ens, labels=CLASSES)) + '\n\n')
    return tabela, texto, ens, pred_ens


def main():
    p = argparse.ArgumentParser(description='Ensemble clássico SVC + ComplementNB')
    p.add_argument('--teste', default=None, help='arquivo .xlsx com a coluna resp_text para rotular')
    args = p.parse_args()
    os.makedirs(PASTA_SAIDA, exist_ok=True)
    data = load_data(CAMINHO_DADOS)
    X, y = data['resp_text'], data['clarity']

    # 1. Divisão do baseline oficial
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=123, stratify=y)
    tab1, txt1, ens, pred = avaliar('Divisão do baseline (aleatória 80/20, semente 123)', X_tr, y_tr, X_te, y_te)

    # 3. Salva o modelo avaliado, recarrega e confere as predições
    caminho_avaliado = os.path.join(PASTA_SAIDA, 'modelo_avaliado_split_baseline.joblib')
    joblib.dump(ens, caminho_avaliado)
    recarregado = joblib.load(caminho_avaliado)
    pred_recarregado = recarregado.predict(X_te)
    assert np.array_equal(pred, pred_recarregado), 'modelo recarregado deu predições diferentes'
    f1_recarregado = f1_score(y_te, pred_recarregado, average='macro')
    pd.DataFrame({'indice': X_te.index, 'rotulo': y_te.to_numpy(), 'previsto': pred_recarregado}).to_csv(
        os.path.join(PASTA_SAIDA, 'predicoes_split_baseline.csv'), index=False)
    txt_recarga = (f'Teste de recarga: {caminho_avaliado} carregado do disco; predições idênticas às do modelo '
                   f'em memória (F1-macro {f1_recarregado:.4f}).\n\n')

    # 2. Holdout agrupado por texto (mesmo split do BERTimbau)
    grupos = X.apply(chave_texto)
    idx_tr, idx_val = list(GroupKFold(n_splits=5, shuffle=True, random_state=42).split(data, groups=grupos))[0]
    tab2, txt2, _, _ = avaliar('Holdout agrupado por texto (fold 1, semente 42)',
                               X.iloc[idx_tr], y.iloc[idx_tr], X.iloc[idx_val], y.iloc[idx_val])

    # 4. Modelo final com 100% dos dados
    print('Treinando modelo final com 100% dos dados...', flush=True)
    final = ensemble_svc_cnb().fit(X, y)
    caminho_final = os.path.join(PASTA_SAIDA, 'modelo_final.joblib')
    joblib.dump(final, caminho_final)

    pd.concat([tab1, tab2]).to_csv(os.path.join(PASTA_SAIDA, 'resultados.csv'), index=False)
    texto = ('ENSEMBLE CLÁSSICO: TF-IDF palavra+caractere + LinearSVC calibrado  +  TF-IDF palavra + ComplementNB\n'
             + '=' * 90 + '\n'
             'Combinação: média simples das probabilidades dos dois modelos.\n\n'
             + txt1 + txt_recarga + txt2
             + f'Modelo final (100% dos {len(X)} textos) salvo em {caminho_final}\n')
    with open(os.path.join(PASTA_SAIDA, 'relatorio.txt'), 'w', encoding='utf-8') as f:
        f.write(texto)
    print(texto)

    # 5. Rotulação do teste (só na entrega)
    if args.teste:
        teste = pd.read_excel(args.teste)
        rotulado = teste.copy()
        rotulado['clarity'] = final.predict(teste['resp_text'].fillna('').astype(str))
        caminho = os.path.join(PASTA_SAIDA, 'teste_rotulado.xlsx')
        rotulado.to_excel(caminho, index=False)
        print(f'Teste rotulado ({len(rotulado)} linhas) salvo em {caminho}')


if __name__ == '__main__':
    main()
