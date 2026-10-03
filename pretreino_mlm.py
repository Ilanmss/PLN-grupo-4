# Pré-treino adaptativo à tarefa (TAPT - Task-Adaptive Pre-Training, Gururangan et al., 2020)
#
# Continua o pré-treino do modelo (ex.: BERTimbau) com Masked Language Modeling (MLM) nos textos do e-SIC,
# SEM rótulos, antes do fine-tuning. O modelo se adapta ao vocabulário e estilo das respostas
# (jargão jurídico-administrativo, templates de órgãos, "Decreto nº 7.724/2012" etc.).
#
# Quais textos entram no MLM (--textos):
#   holdout -> só os textos do treino do holdout (80%, mesmo split agrupado do finetune_bertimbau.py).
#   fold    -> só os textos de treino do fold --fold da validação cruzada. Usado na validação cruzada: para cada fold,
#              um TAPT próprio que nunca vê os textos do fold de validação (sem vazamento).
#   todos   -> todos os textos do train.xlsx, sem rótulos. Usado SÓ para o modelo final da entrega, depois da
#              validação cruzada (nesse momento, todo o train.xlsx é treino).
# A perda MLM é acompanhada em textos fora do MLM (validação), exceto em 'todos'. O conjunto de teste NUNCA é usado.
#
# Saída: um modelo base (sem cabeça) que pode ser usado no fine-tuning:
#   python pretreino_mlm.py --textos fold --fold 1 --saida modelos/tapt_folds/fold_1
#   python finetune_bertimbau.py --modo cv --apenas-fold 1 --epocas 2 --modelo modelos/tapt_folds/fold_1 --saida modelos/bertimbau/cv_tapt_2ep

import argparse
import json
import math
import os
import time

import numpy as np
import torch
from transformers import AutoModel, AutoModelForMaskedLM, AutoTokenizer

from finetune_bertimbau import (criar_agendador, criar_lotes, criar_otimizador, dividir_folds, fixar_semente,
                                montar_lote, tokenizar)
from src.data_loader import load_data

CAMINHO_DADOS = 'data/train.xlsx'


def ler_argumentos():
    p = argparse.ArgumentParser(description='Pré-treino MLM adaptativo (TAPT)')
    p.add_argument('--modelo', default='neuralmind/bert-base-portuguese-cased')
    p.add_argument('--saida', default='modelos/bertimbau_tapt')
    p.add_argument('--max-len', type=int, default=256)
    p.add_argument('--tokens-inicio', type=int, default=64)
    p.add_argument('--epocas', type=int, default=3)
    p.add_argument('--lr', type=float, default=5e-5)
    p.add_argument('--lote', type=int, default=16)
    p.add_argument('--acumulacao', type=int, default=2)
    p.add_argument('--warmup', type=float, default=0.06)
    p.add_argument('--prob-mascara', type=float, default=0.15)
    p.add_argument('--folds', type=int, default=5)
    p.add_argument('--semente', type=int, default=42)
    p.add_argument('--semente-divisao', type=int, default=42)
    p.add_argument('--textos', choices=['holdout', 'fold', 'todos'], default='holdout',
                   help='holdout: só o treino do holdout; fold: só o treino do fold --fold da validação cruzada; '
                        'todos: todos os textos do train.xlsx (sem rótulos; só para o modelo final)')
    p.add_argument('--fold', type=int, default=None, help='fold da validação cruzada (1 a --folds), com --textos fold')
    p.add_argument('--amostra', type=int, default=None, help='usa só N textos (teste rápido do código)')
    return p.parse_args()


def mascarar(input_ids, mascara_atencao, tokenizer, prob, gerador):
    """Mascaramento padrão do BERT: 15% dos tokens; destes, 80% -> [MASK], 10% -> aleatório, 10% mantidos."""
    input_ids = input_ids.clone()
    rotulos = input_ids.clone()
    especiais = (mascara_atencao == 0) | (input_ids == tokenizer.cls_token_id) | (input_ids == tokenizer.sep_token_id)
    probs = torch.full(input_ids.shape, prob)
    probs[especiais] = 0.0
    selecionados = torch.bernoulli(probs, generator=gerador).bool()
    rotulos[~selecionados] = -100  # a perda só considera os tokens selecionados

    trocar_por_mask = torch.bernoulli(torch.full(input_ids.shape, 0.8), generator=gerador).bool() & selecionados
    input_ids[trocar_por_mask] = tokenizer.mask_token_id
    aleatorios = torch.bernoulli(torch.full(input_ids.shape, 0.5), generator=gerador).bool() & selecionados & ~trocar_por_mask
    input_ids[aleatorios] = torch.randint(len(tokenizer), input_ids.shape, generator=gerador)[aleatorios]
    return input_ids, rotulos


def perda_mlm(modelo, ids, args, tokenizer, dispositivo, semente=0):
    """Perda MLM média (máscara fixa pela semente, para ser comparável entre épocas)."""
    modelo.eval()
    gerador = torch.Generator().manual_seed(semente)
    usar_amp = dispositivo.type == 'cuda'
    total, n = 0.0, 0
    with torch.no_grad():
        for lote in criar_lotes(ids, args.lote * 2, embaralhar=False):
            input_ids, mascara = montar_lote(ids, lote, tokenizer.pad_token_id, torch.device('cpu'))
            entrada, rotulos = mascarar(input_ids, mascara, tokenizer, args.prob_mascara, gerador)
            with torch.autocast(device_type=dispositivo.type, dtype=torch.float16, enabled=usar_amp):
                saida = modelo(input_ids=entrada.to(dispositivo), attention_mask=mascara.to(dispositivo),
                               labels=rotulos.to(dispositivo))
            total += saida.loss.item() * len(lote)
            n += len(lote)
    return total / n


def main():
    args = ler_argumentos()
    fixar_semente(args.semente)
    dispositivo = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Dispositivo: {torch.cuda.get_device_name(0) if dispositivo.type == "cuda" else "CPU"}')

    data = load_data(CAMINHO_DADOS)
    if args.textos == 'todos':
        # Todos os textos no MLM; a perda é acompanhada em 1000 deles (também vistos no treino)
        idx_treino = np.arange(len(data))
        idx_val = np.random.default_rng(args.semente).permutation(len(data))
        descricao = 'todos os textos do train.xlsx (sem rótulos); perda acompanhada em textos também usados no MLM'
    elif args.textos == 'fold':
        # Só os textos de treino do fold: os textos do fold de validação não entram no MLM (sem vazamento)
        assert args.fold is not None and 1 <= args.fold <= args.folds, 'informe --fold entre 1 e --folds'
        idx_treino, idx_val = dividir_folds(data, args.folds, args.semente_divisao)[args.fold - 1]
        descricao = f'textos de treino do fold {args.fold}; perda acompanhada em textos da validação do fold (fora do MLM)'
    else:
        idx_treino, idx_val = dividir_folds(data, args.folds, args.semente_divisao)[0]
        descricao = 'textos do treino do holdout; perda acompanhada em textos da validação (fora do MLM)'
    if args.amostra:
        idx_treino, idx_val = idx_treino[:args.amostra], idx_val[:max(8, args.amostra // 4)]

    tokenizer = AutoTokenizer.from_pretrained(args.modelo)
    ids_treino = tokenizar(data['resp_text'].iloc[idx_treino], tokenizer, args.max_len, args.tokens_inicio)
    ids_val = tokenizar(data['resp_text'].iloc[idx_val[:1000]], tokenizer, args.max_len, args.tokens_inicio)
    print(f'MLM em {len(ids_treino)} textos: {descricao} ({len(ids_val)} textos)')

    modelo = AutoModelForMaskedLM.from_pretrained(args.modelo).to(dispositivo)
    otimizador = criar_otimizador(modelo, args.lr)
    passos_totais = math.ceil(math.ceil(len(ids_treino) / args.lote) / args.acumulacao) * args.epocas
    agendador = criar_agendador(otimizador, passos_totais, args.warmup)
    usar_amp = dispositivo.type == 'cuda'
    escalador = torch.amp.GradScaler('cuda', enabled=usar_amp)
    rng = np.random.default_rng(args.semente)
    gerador = torch.Generator().manual_seed(args.semente)

    historico = [{'epoca': 0, 'perda_mlm_val': perda_mlm(modelo, ids_val, args, tokenizer, dispositivo)}]
    print(f"  época 0 (modelo original) | perda MLM validação {historico[0]['perda_mlm_val']:.4f}", flush=True)

    for epoca in range(1, args.epocas + 1):
        modelo.train()
        inicio, perda_acumulada = time.time(), 0.0
        lotes = criar_lotes(ids_treino, args.lote, embaralhar=True, rng=rng)
        otimizador.zero_grad()
        for n_lote, lote in enumerate(lotes, start=1):
            input_ids, mascara = montar_lote(ids_treino, lote, tokenizer.pad_token_id, torch.device('cpu'))
            entrada, rotulos = mascarar(input_ids, mascara, tokenizer, args.prob_mascara, gerador)
            with torch.autocast(device_type=dispositivo.type, dtype=torch.float16, enabled=usar_amp):
                perda = modelo(input_ids=entrada.to(dispositivo), attention_mask=mascara.to(dispositivo),
                               labels=rotulos.to(dispositivo)).loss
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
                restante = (time.time() - inicio) / n_lote * (len(lotes) - n_lote)
                print(f'  época {epoca} | lote {n_lote}/{len(lotes)} | perda {perda_acumulada / n_lote:.4f} '
                      f'| ~{restante / 60:.1f} min restantes', flush=True)

        registro = {'epoca': epoca, 'perda_mlm_treino': perda_acumulada / len(lotes),
                    'perda_mlm_val': perda_mlm(modelo, ids_val, args, tokenizer, dispositivo),
                    'minutos': (time.time() - inicio) / 60}
        historico.append(registro)
        print('  ' + ' | '.join(f'{k}={v:.4f}' if isinstance(v, float) else f'{k}={v}' for k, v in registro.items()), flush=True)

    # Salva só o encoder adaptado, reaproveitando o pooler original do checkpoint (o MLM não treina o pooler)
    base = AutoModel.from_pretrained(args.modelo)
    faltando, _ = base.load_state_dict(modelo.base_model.state_dict(), strict=False)
    assert all('pooler' in chave for chave in faltando), f'pesos do encoder não carregados: {faltando}'
    os.makedirs(args.saida, exist_ok=True)
    base.save_pretrained(args.saida)
    tokenizer.save_pretrained(args.saida)
    with open(os.path.join(args.saida, 'historico_mlm.json'), 'w', encoding='utf-8') as f:
        json.dump({'config': vars(args), 'historico': historico}, f, indent=2, ensure_ascii=False)
    print(f'Modelo adaptado salvo em {args.saida}')


if __name__ == '__main__':
    main()
