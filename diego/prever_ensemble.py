# Rotula um arquivo (ex.: o conjunto de teste) com um dos melhores modelos do relatório.
#
# Lê a receita em modelos/melhores_modelos.json (gerada por gerar_relatorio.py): componentes, pesos e viés.
#   - componentes BERT: modelos finais salvos (bertimbau -> modelos/bertimbau/final_2ep/final;
#     bertimbau_tapt -> modelos/bertimbau/final_tapt_2ep/final)
#     (treinado com 100% do train.xlsx por: python finetune_bertimbau.py --modo final --epocas 2 --saida modelos/bertimbau/final_2ep)
#   - componentes clássicos: treinados com 100% do train.xlsx e guardados em modelos/finais/<nome>.joblib
#     (reaproveitados nas execuções seguintes)
# Combina as probabilidades com a média ponderada da receita, soma o viés por classe e escolhe a maior.
#
# Uso:
#   python prever_ensemble.py --modelo bert --teste <arquivo>.xlsx
#   python prever_ensemble.py --modelo outros_metodos --teste <arquivo>.xlsx
#   python prever_ensemble.py --modelo bert --sem-vies --teste <arquivo>.xlsx   # variante de maior acurácia (usada na entrega)

import argparse
import json
import os
from argparse import Namespace

import joblib
import numpy as np
import pandas as pd

from src.data_loader import load_data
from src.modelos_classicos import FABRICAS

CAMINHO_DADOS = 'data/train.xlsx'
CAMINHO_RECEITAS = 'modelos/melhores_modelos.json'
PASTAS_BERT_FINAL = {'bertimbau': 'modelos/bertimbau/final_2ep/final',
                     'bertimbau_tapt': 'modelos/bertimbau/final_tapt_2ep/final'}
PASTA_FINAIS = 'modelos/finais'
CLASSES = ['c1', 'c234', 'c5']


def probabilidades_bert(textos, pasta):
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    from diego.finetune_bertimbau import prever_probabilidades, tokenizar

    with open(os.path.join(pasta, 'config_treino.json'), encoding='utf-8') as f:
        config = json.load(f)
    dispositivo = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = AutoTokenizer.from_pretrained(pasta)
    modelo = AutoModelForSequenceClassification.from_pretrained(pasta).to(dispositivo)
    assert [modelo.config.id2label[i] for i in range(3)] == CLASSES
    ids = tokenizar(textos, tokenizer, config['max_len'], config['tokens_inicio'])
    return prever_probabilidades(modelo, ids, Namespace(lote=config['lote']), dispositivo, tokenizer.pad_token_id)


def probabilidades_classico(nome, textos):
    caminho = os.path.join(PASTA_FINAIS, f'{nome}.joblib')
    if os.path.exists(caminho):
        modelo = joblib.load(caminho)
    else:
        print(f'Treinando {nome} com 100% do train.xlsx...', flush=True)
        data = load_data(CAMINHO_DADOS)
        modelo = FABRICAS[nome]().fit(data['resp_text'], data['clarity'])
        os.makedirs(PASTA_FINAIS, exist_ok=True)
        joblib.dump(modelo, caminho)
    assert list(modelo.classes_) == CLASSES
    return modelo.predict_proba(textos)


def main():
    p = argparse.ArgumentParser(description='Rotula um arquivo com o melhor modelo do relatório')
    p.add_argument('--modelo', choices=['bert', 'outros_metodos'], required=True)
    p.add_argument('--teste', required=True, help='arquivo .xlsx com a coluna resp_text')
    p.add_argument('--saida', default='modelos/entrega')
    p.add_argument('--sem-vies', action='store_true',
                   help='usa a mesma média ponderada, sem o ajuste de viés por classe (variante de maior acurácia na '
                        'validação cruzada; o viés favorece o F1-macro). Saída em <saida>/<modelo>_sem_vies')
    args = p.parse_args()

    with open(CAMINHO_RECEITAS, encoding='utf-8') as f:
        receita = json.load(f)[args.modelo]
    vies = np.zeros(len(CLASSES)) if args.sem_vies else np.array([receita['vies'][c] for c in CLASSES])
    print(f"Modelo: {receita['nome']} (F1 validação cruzada {receita['f1_cv']})")
    print('Viés por classe: ' + ('desativado (--sem-vies)' if args.sem_vies else str(dict(zip(CLASSES, vies.tolist())))))

    aba = pd.ExcelFile(args.teste).sheet_names[0]
    teste = pd.read_excel(args.teste, sheet_name=aba)
    textos = teste['resp_text'].fillna('').astype(str)
    pasta = os.path.join(args.saida, args.modelo + ('_sem_vies' if args.sem_vies else ''))
    os.makedirs(pasta, exist_ok=True)

    soma = np.zeros((len(textos), len(CLASSES)))
    for nome, peso in receita['pesos'].items():
        if peso == 0:
            continue  # componente descartado pela otimização dos pesos
        if nome in PASTAS_BERT_FINAL:
            probs = probabilidades_bert(textos, PASTAS_BERT_FINAL[nome])
        else:
            probs = probabilidades_classico(nome, textos)
        pd.DataFrame(probs, columns=[f'p_{c}' for c in CLASSES]).to_csv(
            os.path.join(pasta, f'probabilidades_{nome}.csv'), index_label='indice')
        soma += peso * probs
    escores = np.log(np.clip(soma, 1e-12, None)) + vies

    # Mesmo nome de arquivo, mesma aba, mesmas colunas e mesma ordem de linhas do arquivo recebido;
    # só a coluna clarity é preenchida.
    rotulado = teste.copy()
    rotulado['clarity'] = np.array(CLASSES)[escores.argmax(axis=1)]
    caminho = os.path.join(pasta, os.path.basename(args.teste))
    rotulado.to_excel(caminho, index=False, sheet_name=aba)

    # Conferência do arquivo gravado contra o original (erros de formato são penalizados na entrega)
    conferido = pd.read_excel(caminho, sheet_name=aba)
    assert pd.ExcelFile(caminho).sheet_names == pd.ExcelFile(args.teste).sheet_names, 'abas diferentes do original'
    assert list(conferido.columns) == list(teste.columns), 'colunas diferentes do original'
    assert len(conferido) == len(teste), 'número de linhas diferente do original'
    assert conferido['resp_text'].fillna('').astype(str).equals(textos), 'textos alterados ou fora de ordem'
    assert conferido['clarity'].isin(CLASSES).all(), 'rótulo ausente ou fora de {c1, c234, c5}'
    print(f'{len(conferido)} linhas rotuladas -> {caminho}')
    print(f'Conferido: aba "{aba}", colunas {list(conferido.columns)}, textos idênticos e na mesma ordem, rótulos válidos.')
    print(conferido['clarity'].value_counts(normalize=True).round(4).to_string())


if __name__ == '__main__':
    main()
