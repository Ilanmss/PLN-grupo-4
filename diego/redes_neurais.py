# Teste rápido com redes neurais simples, treinadas do zero (PyTorch).
#
# Arquiteturas:
#   mlp_tfidf -> perceptron multicamadas sobre TF-IDF (palavras 1-2, 30 mil atributos)
#   fasttext  -> média dos embeddings das palavras + camada linear (estilo fastText)
#   textcnn   -> CNN de texto (Kim, 2014): convoluções de larguras 2 a 5 sobre embeddings + max-pooling
#   bigru     -> GRU bidirecional sobre embeddings + pooling com atenção
#
# Avaliação: divisão única do baseline oficial (aleatória 80/20 estratificada, semente 123), sem validação cruzada.
# 10% do treino é separado como validação interna para escolher a época de parada; o teste só é usado no final.
#
# Uso: python redes_neurais.py [--modelos mlp_tfidf fasttext textcnn bigru] [--epocas 12]

import argparse
import os
import re
import time
from collections import Counter

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

from src.data_loader import load_data
from src.divisoes import obter_divisao

PASTA_SAIDA = 'modelos/redes_neurais'
CLASSES = ['c1', 'c234', 'c5']
TAM_VOCAB, MAX_TOKENS, DIM_EMB = 30000, 400, 200
PAD, UNK = 0, 1


def fixar_semente(semente):
    np.random.seed(semente)
    torch.manual_seed(semente)
    torch.cuda.manual_seed_all(semente)


# ---------------------------------------------------------------------------
# Modelos
# ---------------------------------------------------------------------------

class MLP(nn.Module):
    def __init__(self, n_atributos):
        super().__init__()
        self.rede = nn.Sequential(nn.Dropout(0.5), nn.Linear(n_atributos, 256), nn.ReLU(),
                                  nn.Dropout(0.5), nn.Linear(256, len(CLASSES)))

    def forward(self, x):
        return self.rede(x)


class FastText(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(TAM_VOCAB, DIM_EMB, padding_idx=PAD)
        self.saida = nn.Sequential(nn.Dropout(0.3), nn.Linear(DIM_EMB, len(CLASSES)))

    def forward(self, x):
        mascara = (x != PAD).unsqueeze(-1).float()
        media = (self.emb(x) * mascara).sum(1) / mascara.sum(1).clamp(min=1)
        return self.saida(media)


class TextCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(TAM_VOCAB, DIM_EMB, padding_idx=PAD)
        self.convs = nn.ModuleList([nn.Conv1d(DIM_EMB, 128, k, padding=k // 2) for k in (2, 3, 4, 5)])
        self.saida = nn.Sequential(nn.Dropout(0.5), nn.Linear(128 * 4, len(CLASSES)))

    def forward(self, x):
        e = F.dropout(self.emb(x), 0.2, self.training).transpose(1, 2)
        return self.saida(torch.cat([F.relu(c(e)).max(dim=2).values for c in self.convs], dim=1))


class BiGRU(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(TAM_VOCAB, DIM_EMB, padding_idx=PAD)
        self.gru = nn.GRU(DIM_EMB, 128, batch_first=True, bidirectional=True)
        self.atencao = nn.Linear(256, 1)
        self.saida = nn.Sequential(nn.Dropout(0.5), nn.Linear(256, len(CLASSES)))

    def forward(self, x):
        h, _ = self.gru(F.dropout(self.emb(x), 0.2, self.training))
        pesos = self.atencao(h).squeeze(-1).masked_fill(x == PAD, -1e9).softmax(dim=1)
        return self.saida((h * pesos.unsqueeze(-1)).sum(1))


MODELOS = {'mlp_tfidf': MLP, 'fasttext': FastText, 'textcnn': TextCNN, 'bigru': BiGRU}
NOMES = {'mlp_tfidf': 'MLP sobre TF-IDF', 'fasttext': 'fastText (média de embeddings)',
         'textcnn': 'TextCNN', 'bigru': 'BiGRU com atenção'}


# ---------------------------------------------------------------------------
# Dados
# ---------------------------------------------------------------------------

def tokens(texto):
    return re.findall(r'\w+', texto.lower())


def sequencias(textos, vocab):
    seqs = np.full((len(textos), MAX_TOKENS), PAD, dtype=np.int64)
    for i, t in enumerate(textos):
        toks = tokens(t)
        if len(toks) > MAX_TOKENS:  # mantém início e fim, como no BERTimbau
            toks = toks[:MAX_TOKENS // 4] + toks[-(MAX_TOKENS - MAX_TOKENS // 4):]
        ids = [vocab.get(p, UNK) for p in toks] or [UNK]
        seqs[i, :len(ids)] = ids
    return seqs


def prever(modelo, X, dispositivo, denso):
    modelo.eval()
    probs = []
    with torch.no_grad():
        for i in range(0, X.shape[0], 512):
            lote = X[i:i + 512]
            lote = torch.from_numpy(lote.toarray() if denso else lote).to(dispositivo)
            probs.append(F.softmax(modelo(lote.float() if denso else lote), dim=1).cpu().numpy())
    return np.vstack(probs)


def treinar(nome, X_tr, y_tr, X_val, y_val, X_te, args, dispositivo):
    """Treina por args.epocas e devolve as probabilidades de teste da época com melhor acurácia na validação interna."""
    fixar_semente(args.semente)
    denso = nome == 'mlp_tfidf'
    modelo = (MLP(X_tr.shape[1]) if denso else MODELOS[nome]()).to(dispositivo)
    otimizador = torch.optim.AdamW(modelo.parameters(), lr=5e-4 if denso else 1e-3, weight_decay=1e-2)
    y_t = torch.from_numpy(y_tr)
    rng = np.random.default_rng(args.semente)
    melhor = {'acc_val': -1}
    for epoca in range(1, args.epocas + 1):
        modelo.train()
        ordem, perda_total = rng.permutation(X_tr.shape[0]), 0.0
        for i in range(0, len(ordem), args.lote):
            idx = np.sort(ordem[i:i + args.lote])
            lote = torch.from_numpy(X_tr[idx].toarray() if denso else X_tr[idx]).to(dispositivo)
            perda = F.cross_entropy(modelo(lote.float() if denso else lote), y_t[idx].to(dispositivo))
            otimizador.zero_grad()
            perda.backward()
            otimizador.step()
            perda_total += perda.item() * len(idx)
        acc_val = accuracy_score(y_val, prever(modelo, X_val, dispositivo, denso).argmax(1))
        if acc_val > melhor['acc_val']:
            melhor = {'acc_val': acc_val, 'epoca': epoca, 'probs': prever(modelo, X_te, dispositivo, denso)}
        print(f'  {nome:10s} época {epoca:2d} | perda treino {perda_total / len(ordem):.4f} | acc validação interna {acc_val:.4f}', flush=True)
    return melhor


def main():
    p = argparse.ArgumentParser(description='Redes neurais simples treinadas do zero')
    p.add_argument('--modelos', nargs='+', default=list(MODELOS), choices=list(MODELOS))
    p.add_argument('--epocas', type=int, default=12)
    p.add_argument('--lote', type=int, default=64)
    p.add_argument('--semente', type=int, default=42)
    args = p.parse_args()
    os.makedirs(PASTA_SAIDA, exist_ok=True)
    dispositivo = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Dispositivo: {torch.cuda.get_device_name(0) if dispositivo.type == "cuda" else "CPU"}')

    data = load_data('data/train.xlsx')
    y = data['clarity'].map({c: i for i, c in enumerate(CLASSES)}).to_numpy()
    idx_tr, idx_te = obter_divisao(data, 'baseline')
    idx_tr, idx_val = train_test_split(idx_tr, test_size=0.1, random_state=args.semente, stratify=y[idx_tr])
    textos = data['resp_text']

    # Representações ajustadas só no treino
    contagem = Counter(p for t in textos.iloc[idx_tr] for p in tokens(t))
    vocab = {p: i + 2 for i, (p, n) in enumerate(contagem.most_common(TAM_VOCAB - 2)) if n >= 2}
    seq = {k: sequencias(textos.iloc[i].tolist(), vocab) for k, i in [('tr', idx_tr), ('val', idx_val), ('te', idx_te)]}
    vet = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, max_features=30000, dtype=np.float32)
    tfidf = {'tr': vet.fit_transform(textos.iloc[idx_tr]), 'val': vet.transform(textos.iloc[idx_val]),
             'te': vet.transform(textos.iloc[idx_te])}
    print(f'Treino {len(idx_tr)} | validação interna {len(idx_val)} | teste {len(idx_te)} | vocabulário {len(vocab) + 2}')

    linhas = []
    for nome in args.modelos:
        dados = tfidf if nome == 'mlp_tfidf' else seq
        inicio = time.time()
        r = treinar(nome, dados['tr'], y[idx_tr], dados['val'], y[idx_val], dados['te'], args, dispositivo)
        pred = r['probs'].argmax(1)
        linhas.append({'modelo': NOMES[nome], 'accuracy': accuracy_score(y[idx_te], pred),
                       'f1_macro': f1_score(y[idx_te], pred, average='macro'), 'melhor_epoca': r['epoca'],
                       'acc_validacao_interna': r['acc_val'], 'minutos': (time.time() - inicio) / 60})
        df = pd.DataFrame(r['probs'], columns=[f'p_{c}' for c in CLASSES])
        df.insert(0, 'indice', idx_te)
        df['rotulo'] = [CLASSES[i] for i in y[idx_te]]
        df.to_csv(os.path.join(PASTA_SAIDA, f'probabilidades_{nome}.csv'), index=False)
        print(f"=> {NOMES[nome]}: acurácia de teste {linhas[-1]['accuracy']:.4f} (época {r['epoca']}, {linhas[-1]['minutos']:.1f} min)\n", flush=True)

    tabela = pd.DataFrame(linhas)
    tabela.to_csv(os.path.join(PASTA_SAIDA, 'resultados.csv'), index=False, encoding='utf-8-sig')
    print(tabela.round(4).to_string(index=False))


if __name__ == '__main__':
    main()
