# Fine-tuning do BERTimbau (neuralmind/bert-base-portuguese-cased) para classificação {c1, c234, c5}
#
# Modos de execução:
#   holdout -> treina em 80% e valida em 20% (split agrupado por texto). Rápido; use para escolher
#              hiperparâmetros (nº de épocas, lr, max_len). Mostra as métricas a cada época.
#   cv      -> validação cruzada agrupada (GroupKFold, 5 folds). Gera probabilidades out-of-fold
#              (modelos/bertimbau/oof_bertimbau.csv) para comparar com os outros modelos e para stacking.
#              Salva cada fold ao terminar: se a execução for interrompida, é retomada do fold seguinte.
#   final   -> treina em 100% do train.xlsx e salva o modelo. Com --teste, rotula o conjunto de teste.
#   prever  -> carrega o modelo final salvo e rotula o arquivo passado em --teste.
#
# Exemplos:
#   python finetune_bertimbau.py --modo holdout
#   python finetune_bertimbau.py --modo cv --epocas 3
#   python finetune_bertimbau.py --modo final --epocas 3 --teste data/test.xlsx
#
# Decisões principais:
#   - Truncamento início+fim: textos longos mantêm os primeiros e os últimos tokens, pois o final da
#     resposta costuma dizer se o pedido foi atendido, negado ou redirecionado.
#   - Padding dinâmico com lotes de tamanhos parecidos: a mediana é ~100 palavras, então quase nenhum
#     lote precisa de 512 tokens. Isso deixa o treino bem mais rápido na GPU de 6 GB.
#   - Precisão mista (fp16) e acumulação de gradiente para caber na memória da GTX 1660 Super.
#   - Nº de épocas fixo (sem early stopping no fold de validação), para que as métricas de CV não
#     fiquem otimistas. Escolha o nº de épocas pelo modo holdout.

import argparse
import json
import math
import os
import random
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, classification_report, cohen_kappa_score, confusion_matrix, f1_score
from sklearn.model_selection import GroupKFold
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers.utils import logging as hf_logging

from src.data_loader import load_data
from src.deduplicacao import chave_texto
from src.divisoes import DESCRICAO, obter_divisao

CAMINHO_DADOS = 'data/train.xlsx'
PASTA_SAIDA = 'modelos/bertimbau'
CLASSES = ['c1', 'c234', 'c5']
ROTULO_PARA_ID = {c: i for i, c in enumerate(CLASSES)}

hf_logging.set_verbosity_error()  # esconde o aviso de textos maiores que 512 tokens (truncamos manualmente)


def ler_argumentos():
    p = argparse.ArgumentParser(description='Fine-tuning do BERTimbau')
    p.add_argument('--modo', choices=['holdout', 'cv', 'final', 'prever'], default='holdout')
    p.add_argument('--modelo', default='neuralmind/bert-base-portuguese-cased')
    p.add_argument('--max-len', type=int, default=512, help='tamanho máximo em tokens (<= 512)')
    p.add_argument('--tokens-inicio', type=int, default=128, help='tokens mantidos do início em textos longos; o restante vem do fim')
    p.add_argument('--epocas', type=int, default=3)
    p.add_argument('--lr', type=float, default=2e-5)
    p.add_argument('--lote', type=int, default=8, help='tamanho do lote na GPU')
    p.add_argument('--acumulacao', type=int, default=2, help='passos de acumulação (lote efetivo = lote x acumulacao)')
    p.add_argument('--warmup', type=float, default=0.1, help='fração dos passos com aquecimento do lr')
    p.add_argument('--label-smoothing', type=float, default=0.0, help='suavização de rótulos (útil com rótulos ruidosos, ex.: 0.1)')
    p.add_argument('--folds', type=int, default=5)
    p.add_argument('--semente', type=int, default=42, help='semente do treino (inicialização da cabeça, ordem dos lotes)')
    p.add_argument('--semente-divisao', type=int, default=42, help='semente do split treino/validação (manter fixa para comparar experimentos)')
    p.add_argument('--divisao', choices=['agrupada', 'baseline'], default='agrupada',
                   help='holdout: agrupada (GroupKFold por texto) ou baseline (mesma divisão do baselines.py)')
    p.add_argument('--apenas-fold', type=int, default=None,
                   help='modo cv: treina só este fold (ex.: quando cada fold usa um modelo TAPT próprio)')
    p.add_argument('--amostra', type=int, default=None, help='usa só os N primeiros textos (teste rápido do código)')
    p.add_argument('--teste', default=None, help='arquivo .xlsx com a coluna resp_text para rotular')
    p.add_argument('--saida', default=PASTA_SAIDA)
    return p.parse_args()


def fixar_semente(semente):
    random.seed(semente)
    np.random.seed(semente)
    torch.manual_seed(semente)
    torch.cuda.manual_seed_all(semente)


# ---------------------------------------------------------------------------
# Tokenização e lotes
# ---------------------------------------------------------------------------

def tokenizar(textos, tokenizer, max_len, tokens_inicio):
    """Tokeniza os textos; os que passam de max_len mantêm o início e o fim."""
    corpo = max_len - 2  # espaço para [CLS] e [SEP]
    tokens_fim = corpo - tokens_inicio
    textos = [chave_texto(str(t)) for t in textos]
    codificados = tokenizer(textos, add_special_tokens=False, truncation=False)['input_ids']

    ids = []
    for tokens in codificados:
        if len(tokens) > corpo:
            tokens = tokens[:tokens_inicio] + (tokens[-tokens_fim:] if tokens_fim > 0 else [])
        ids.append([tokenizer.cls_token_id] + tokens + [tokenizer.sep_token_id])
    return ids


def criar_lotes(ids, tamanho_lote, embaralhar, rng=None):
    """
    Agrupa exemplos de tamanho parecido no mesmo lote (menos padding = treino mais rápido).
    No treino, embaralha em blocos para manter a aleatoriedade; na predição, só ordena por tamanho.
    """
    indices = np.arange(len(ids))
    if embaralhar:
        rng.shuffle(indices)
        tamanho_bloco = tamanho_lote * 50
        blocos = [indices[i:i + tamanho_bloco] for i in range(0, len(indices), tamanho_bloco)]
        indices = np.concatenate([sorted(b, key=lambda i: len(ids[i])) for b in blocos])
    else:
        indices = np.array(sorted(indices, key=lambda i: len(ids[i])))

    lotes = [indices[i:i + tamanho_lote] for i in range(0, len(indices), tamanho_lote)]
    if embaralhar:
        rng.shuffle(lotes)
    return lotes


def montar_lote(ids, lote, id_padding, dispositivo):
    """Monta os tensores de um lote com padding até o maior texto do próprio lote."""
    maior = max(len(ids[i]) for i in lote)
    input_ids = torch.full((len(lote), maior), id_padding, dtype=torch.long)
    mascara = torch.zeros((len(lote), maior), dtype=torch.long)
    for j, i in enumerate(lote):
        n = len(ids[i])
        input_ids[j, :n] = torch.tensor(ids[i], dtype=torch.long)
        mascara[j, :n] = 1
    return input_ids.to(dispositivo), mascara.to(dispositivo)


# ---------------------------------------------------------------------------
# Treino e predição
# ---------------------------------------------------------------------------

def criar_otimizador(modelo, lr):
    # Sem weight decay em bias e LayerNorm (prática padrão no fine-tuning de BERT)
    sem_decay = ['bias', 'LayerNorm.weight']
    grupos = [
        {'params': [p for n, p in modelo.named_parameters() if not any(s in n for s in sem_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in modelo.named_parameters() if any(s in n for s in sem_decay)], 'weight_decay': 0.0},
    ]
    return torch.optim.AdamW(grupos, lr=lr)


def criar_agendador(otimizador, passos_totais, fracao_warmup):
    """lr sobe linearmente no aquecimento e depois decai linearmente até zero."""
    passos_warmup = int(passos_totais * fracao_warmup)

    def fator(passo):
        if passo < passos_warmup:
            return passo / max(1, passos_warmup)
        return max(0.0, (passos_totais - passo) / max(1, passos_totais - passos_warmup))

    return torch.optim.lr_scheduler.LambdaLR(otimizador, fator)


@torch.no_grad()
def prever_probabilidades(modelo, ids, args, dispositivo, id_padding):
    """Devolve a matriz de probabilidades (n_exemplos x 3) na ordem original dos textos."""
    modelo.eval()
    if modelo.config.model_type == 'deberta':
        corrigir_deberta_fp16()
    usar_amp = dispositivo.type == 'cuda'
    probs = np.zeros((len(ids), len(CLASSES)), dtype=np.float32)
    for lote in criar_lotes(ids, args.lote * 4, embaralhar=False):
        input_ids, mascara = montar_lote(ids, lote, id_padding, dispositivo)
        with torch.autocast(device_type=dispositivo.type, dtype=torch.float16, enabled=usar_amp):
            logits = modelo(input_ids=input_ids, attention_mask=mascara).logits
        probs[lote] = torch.softmax(logits.float(), dim=-1).cpu().numpy()
    return probs


def calcular_metricas(y_real, y_pred):
    return {
        'f1_macro': f1_score(y_real, y_pred, average='macro'),
        'accuracy': accuracy_score(y_real, y_pred),
        'kappa': cohen_kappa_score(y_real, y_pred),
    }


def corrigir_deberta_fp16():
    """
    Correção para DeBERTa (ex.: Albertina) com precisão mista fp16.
    No transformers, a atenção do DeBERTa preenche posições de padding com torch.finfo(query_layer.dtype).min.
    Com autocast, query_layer vira float32 (soma com um viés em float32), mas os escores de atenção ficam em
    float16, e o mínimo de float32 não cabe em float16 ("value cannot be converted to type c10::Half without overflow").
    A correção usa o tipo dos próprios escores. Só a linha do preenchimento muda; o resto do método é o original.
    """
    import inspect
    import textwrap

    from transformers.models.deberta import modeling_deberta as md

    classe = md.DisentangledSelfAttention
    if getattr(classe, '_corrigido_fp16', False):
        return
    fonte = textwrap.dedent(inspect.getsource(classe.forward))
    trecho = 'torch.finfo(query_layer.dtype).min'
    assert fonte.count(trecho) == 1, 'versão do transformers diferente da esperada; revise a correção do DeBERTa'
    namespace = {}
    exec(compile(fonte.replace(trecho, 'torch.finfo(attention_scores.dtype).min'), md.__file__, 'exec'),
         md.__dict__, namespace)
    classe.forward = namespace['forward']
    classe._corrigido_fp16 = True


def treinar(ids_treino, y_treino, args, dispositivo, id_padding, ids_val=None, y_val=None, ao_fim_da_epoca=None):
    """
    Treina um modelo do zero (a partir do BERTimbau pré-treinado) pelo nº fixo de épocas.
    Se houver validação, avalia ao fim de cada época e chama ao_fim_da_epoca(epoca, probs_val), se informado.
    Devolve (modelo, histórico, probs_val da última época).
    """
    fixar_semente(args.semente)
    rng = np.random.default_rng(args.semente)

    modelo = AutoModelForSequenceClassification.from_pretrained(
        args.modelo, num_labels=len(CLASSES),
        id2label=dict(enumerate(CLASSES)), label2id=ROTULO_PARA_ID,
    ).float().to(dispositivo)  # pesos mestres em float32 (alguns checkpoints, como o Albertina, vêm em bfloat16)
    if modelo.config.model_type == 'deberta':
        corrigir_deberta_fp16()

    otimizador = criar_otimizador(modelo, args.lr)
    lotes_por_epoca = math.ceil(len(ids_treino) / args.lote)
    passos_totais = math.ceil(lotes_por_epoca / args.acumulacao) * args.epocas
    agendador = criar_agendador(otimizador, passos_totais, args.warmup)

    usar_amp = dispositivo.type == 'cuda'
    escalador = torch.amp.GradScaler('cuda', enabled=usar_amp)
    y_treino = torch.tensor(y_treino, dtype=torch.long)

    historico, probs_val = [], None
    for epoca in range(1, args.epocas + 1):
        modelo.train()
        inicio = time.time()
        perda_acumulada = 0.0
        lotes = criar_lotes(ids_treino, args.lote, embaralhar=True, rng=rng)
        otimizador.zero_grad()

        for n_lote, lote in enumerate(lotes, start=1):
            input_ids, mascara = montar_lote(ids_treino, lote, id_padding, dispositivo)
            rotulos = y_treino[torch.as_tensor(lote)].to(dispositivo)
            with torch.autocast(device_type=dispositivo.type, dtype=torch.float16, enabled=usar_amp):
                logits = modelo(input_ids=input_ids, attention_mask=mascara).logits
            perda = F.cross_entropy(logits.float(), rotulos, label_smoothing=args.label_smoothing)
            escalador.scale(perda / args.acumulacao).backward()
            perda_acumulada += perda.item()

            if n_lote % args.acumulacao == 0 or n_lote == len(lotes):
                escalador.unscale_(otimizador)
                torch.nn.utils.clip_grad_norm_(modelo.parameters(), 1.0)
                escalador.step(otimizador)
                escalador.update()
                otimizador.zero_grad()
                agendador.step()

            if n_lote % 200 == 0:
                decorrido = time.time() - inicio
                restante = decorrido / n_lote * (len(lotes) - n_lote)
                print(f'  época {epoca} | lote {n_lote}/{len(lotes)} | perda {perda_acumulada / n_lote:.4f} '
                      f'| ~{restante / 60:.1f} min restantes', flush=True)

        registro = {'epoca': epoca, 'perda_treino': perda_acumulada / len(lotes), 'minutos': (time.time() - inicio) / 60}
        if ids_val is not None:
            probs_val = prever_probabilidades(modelo, ids_val, args, dispositivo, id_padding)
            registro.update(calcular_metricas(y_val, probs_val.argmax(axis=1)))
            if ao_fim_da_epoca is not None:
                ao_fim_da_epoca(epoca, probs_val)
        historico.append(registro)
        print('  ' + ' | '.join(f'{k}={v:.4f}' if isinstance(v, float) else f'{k}={v}' for k, v in registro.items()), flush=True)

    return modelo, historico, probs_val


def liberar_memoria(modelo):
    del modelo
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def relatorio_texto(titulo, y_real, y_pred, args, extra=''):
    m = calcular_metricas(y_real, y_pred)
    linhas = [
        titulo, '=' * 70, '',
        'Configuração:', json.dumps(vars(args), indent=2, ensure_ascii=False), '',
        f"F1-macro: {m['f1_macro']:.4f}", f"Accuracy: {m['accuracy']:.4f}", f"Cohen's kappa: {m['kappa']:.4f}", '',
        extra,
        classification_report(y_real, y_pred, target_names=CLASSES, digits=4),
        'Matriz de confusão (linhas = real, colunas = previsto):',
        f'Ordem: {CLASSES}', str(confusion_matrix(y_real, y_pred)),
    ]
    return '\n'.join(linhas)


# ---------------------------------------------------------------------------
# Modos
# ---------------------------------------------------------------------------

def dividir_folds(data, folds, semente_divisao):
    grupos = data['resp_text'].apply(chave_texto)
    gkf = GroupKFold(n_splits=folds, shuffle=True, random_state=semente_divisao)
    return list(gkf.split(data, groups=grupos))


def modo_holdout(data, ids, y, args, dispositivo, id_padding):
    if args.divisao == 'baseline':
        idx_treino, idx_val = obter_divisao(data, 'baseline')
    else:
        idx_treino, idx_val = dividir_folds(data, args.folds, args.semente_divisao)[0]
    print(f'Holdout ({DESCRICAO[args.divisao]}): {len(idx_treino)} treino / {len(idx_val)} validação')

    # Salva as probabilidades de cada época (permite comparar épocas e montar ensembles depois)
    def salvar_epoca(epoca, probs_epoca):
        salvar_probabilidades(os.path.join(args.saida, f'holdout_probabilidades_epoca_{epoca}.csv'),
                              idx_val, probs_epoca, y[idx_val])

    modelo, historico, probs = treinar(
        [ids[i] for i in idx_treino], y[idx_treino], args, dispositivo, id_padding,
        ids_val=[ids[i] for i in idx_val], y_val=y[idx_val], ao_fim_da_epoca=salvar_epoca)
    liberar_memoria(modelo)

    pd.DataFrame(historico).to_csv(os.path.join(args.saida, 'holdout_historico.csv'), index=False)
    salvar_probabilidades(os.path.join(args.saida, 'holdout_probabilidades.csv'), idx_val, probs, y[idx_val])
    texto = relatorio_texto(f'TRANSFORMER - HOLDOUT: {DESCRICAO[args.divisao]} (última época)', y[idx_val], probs.argmax(axis=1), args,
                            extra='Histórico por época:\n' + pd.DataFrame(historico).to_string(index=False) + '\n')
    escrever(os.path.join(args.saida, 'holdout_relatorio.txt'), texto)
    print('\n' + texto)


def modo_cv(data, ids, y, args, dispositivo, id_padding):
    pasta_folds = os.path.join(args.saida, 'cv_folds')
    os.makedirs(pasta_folds, exist_ok=True)

    for fold, (idx_treino, idx_val) in enumerate(dividir_folds(data, args.folds, args.semente_divisao), start=1):
        arquivo_fold = os.path.join(pasta_folds, f'fold_{fold}.csv')
        if args.apenas_fold is not None and fold != args.apenas_fold:
            continue
        if os.path.exists(arquivo_fold):
            print(f'Fold {fold} já concluído, pulando.')
            continue
        print(f'\n=== Fold {fold}/{args.folds}: {len(idx_treino)} treino / {len(idx_val)} validação ===')
        modelo, historico, probs = treinar(
            [ids[i] for i in idx_treino], y[idx_treino], args, dispositivo, id_padding,
            ids_val=[ids[i] for i in idx_val], y_val=y[idx_val])
        liberar_memoria(modelo)
        pd.DataFrame(historico).assign(fold=fold).to_csv(os.path.join(pasta_folds, f'historico_fold_{fold}.csv'), index=False)
        salvar_probabilidades(arquivo_fold, idx_val, probs, y[idx_val], fold=fold)

    # Junta os folds em um único arquivo out-of-fold, na ordem original do train.xlsx (só quando todos estão prontos)
    faltando = [f for f in range(1, args.folds + 1) if not os.path.exists(os.path.join(pasta_folds, f'fold_{f}.csv'))]
    if faltando:
        print(f'Folds ainda não concluídos: {faltando}. O relatório da validação cruzada sai quando todos terminarem.')
        return
    oof = pd.concat([pd.read_csv(os.path.join(pasta_folds, f'fold_{f}.csv')) for f in range(1, args.folds + 1)])
    oof = oof.sort_values('indice').reset_index(drop=True)
    oof.to_csv(os.path.join(args.saida, 'oof_bertimbau.csv'), index=False)

    por_fold = oof.groupby('fold').apply(
        lambda d: pd.Series(calcular_metricas(d['rotulo'], d['previsto'])), include_groups=False)
    y_oof = oof['rotulo'].map(ROTULO_PARA_ID).to_numpy()
    y_pred = oof['previsto'].map(ROTULO_PARA_ID).to_numpy()
    resumo = (f"F1-macro por fold: média {por_fold['f1_macro'].mean():.4f} ± {por_fold['f1_macro'].std(ddof=0):.4f}\n"
              f"Accuracy por fold: média {por_fold['accuracy'].mean():.4f} ± {por_fold['accuracy'].std(ddof=0):.4f}\n\n"
              + por_fold.round(4).to_string() + '\n')
    texto = relatorio_texto(f'BERTIMBAU - VALIDAÇÃO CRUZADA AGRUPADA ({args.folds} folds, out-of-fold)',
                            y_oof, y_pred, args, extra=resumo)
    escrever(os.path.join(args.saida, 'cv_relatorio.txt'), texto)
    print('\n' + texto)


def modo_final(data, ids, y, args, dispositivo, id_padding, tokenizer):
    print(f'Treino final com {len(ids)} exemplos')
    modelo, historico, _ = treinar(ids, y, args, dispositivo, id_padding)
    pasta_modelo = os.path.join(args.saida, 'final')
    modelo.save_pretrained(pasta_modelo)
    tokenizer.save_pretrained(pasta_modelo)
    with open(os.path.join(pasta_modelo, 'config_treino.json'), 'w', encoding='utf-8') as f:
        json.dump(vars(args), f, indent=2, ensure_ascii=False)
    print(f'Modelo salvo em {pasta_modelo}')

    if args.teste:
        rotular_teste(modelo, tokenizer, args, dispositivo)
    liberar_memoria(modelo)


def modo_prever(args, dispositivo):
    pasta_modelo = os.path.join(args.saida, 'final')
    with open(os.path.join(pasta_modelo, 'config_treino.json'), encoding='utf-8') as f:
        config = json.load(f)
    # Usa o mesmo truncamento do treino
    args.max_len, args.tokens_inicio = config['max_len'], config['tokens_inicio']
    tokenizer = AutoTokenizer.from_pretrained(pasta_modelo)
    modelo = AutoModelForSequenceClassification.from_pretrained(pasta_modelo).to(dispositivo)
    rotular_teste(modelo, tokenizer, args, dispositivo)


def rotular_teste(modelo, tokenizer, args, dispositivo):
    teste = pd.read_excel(args.teste)
    ids = tokenizar(teste['resp_text'].fillna(''), tokenizer, args.max_len, args.tokens_inicio)
    probs = prever_probabilidades(modelo, ids, args, dispositivo, tokenizer.pad_token_id)

    # Probabilidades separadas (para ensemble/stacking com outros modelos)
    pd.DataFrame(probs, columns=[f'p_{c}' for c in CLASSES]).to_csv(
        os.path.join(args.saida, 'teste_probabilidades.csv'), index_label='indice')

    # Planilha rotulada: mesmas linhas, mesma ordem, coluna clarity preenchida
    rotulado = teste.copy()
    rotulado['clarity'] = [CLASSES[i] for i in probs.argmax(axis=1)]
    caminho = os.path.join(args.saida, 'teste_rotulado.xlsx')
    rotulado.to_excel(caminho, index=False)
    print(f'Teste rotulado ({len(rotulado)} linhas) salvo em {caminho}')
    print(rotulado['clarity'].value_counts(normalize=True).round(4).to_string())


# ---------------------------------------------------------------------------
# Utilitários
# ---------------------------------------------------------------------------

def salvar_probabilidades(caminho, indices, probs, y_real, fold=None):
    df = pd.DataFrame(probs, columns=[f'p_{c}' for c in CLASSES])
    df.insert(0, 'indice', indices)
    if fold is not None:
        df.insert(1, 'fold', fold)
    df['rotulo'] = [CLASSES[i] for i in y_real]
    df['previsto'] = [CLASSES[i] for i in probs.argmax(axis=1)]
    df.to_csv(caminho, index=False)


def escrever(caminho, texto):
    with open(caminho, 'w', encoding='utf-8') as f:
        f.write(texto)


def main():
    args = ler_argumentos()
    os.makedirs(args.saida, exist_ok=True)
    fixar_semente(args.semente)

    dispositivo = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Dispositivo: {torch.cuda.get_device_name(0) if dispositivo.type == "cuda" else "CPU (vai ser muito lento!)"}')

    if args.modo == 'prever':
        if not args.teste:
            raise SystemExit('Informe o arquivo a rotular com --teste')
        modo_prever(args, dispositivo)
        return

    data = load_data(CAMINHO_DADOS)
    if args.amostra:
        data = data.iloc[:args.amostra].reset_index(drop=True)
    tokenizer = AutoTokenizer.from_pretrained(args.modelo)
    print('Tokenizando...')
    ids = tokenizar(data['resp_text'], tokenizer, args.max_len, args.tokens_inicio)
    y = data['clarity'].map(ROTULO_PARA_ID).to_numpy()
    tamanhos = np.array([len(i) for i in ids])
    print(f'Tokens por texto: mediana {np.median(tamanhos):.0f} | '
          f'{(tamanhos >= args.max_len).mean():.1%} truncados em {args.max_len}')

    if args.modo == 'holdout':
        modo_holdout(data, ids, y, args, dispositivo, tokenizer.pad_token_id)
    elif args.modo == 'cv':
        modo_cv(data, ids, y, args, dispositivo, tokenizer.pad_token_id)
    else:
        modo_final(data, ids, y, args, dispositivo, tokenizer.pad_token_id, tokenizer)


if __name__ == '__main__':
    main()
