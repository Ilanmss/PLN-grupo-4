"""
Testes de vetorizacao com Word2Vec + regressao logistica, com varias
configuracoes.

Cada texto vira um vetor denso a partir dos vetores de suas palavras, com tres
formas de agregacao (POOLINGS):
- media      : media simples dos vetores das palavras
- media_idf  : media ponderada pelo IDF (palavras raras pesam mais)
- media_max  : concatenacao da media com o maximo elemento a elemento
               (dobra a dimensao)

Fontes dos vetores de palavras:
- w2v_<nome>  : Word2Vec treinado nos textos de DESENVOLVIMENTO, uma fonte para
                cada configuracao de W2V_CONFIGS (dimensao, janela, skip-gram
                vs CBOW, min_count, epocas). O teste nao participa do treino
                dos vetores nem do IDF.
- nilc_<nome> : vetores pre-treinados do NILC no Hugging Face
                (PRETRAINED_HF_REPOS). Repositorios que nao existirem ou que
                falharem no download sao pulados com um aviso.
- w2v_prev    : arquivo local .txt/.bin no formato word2vec (PRETRAINED_PATH).

Para cada variante (fonte x pooling): escolhe C da regressao logistica por CV
(lista curta) e avalia no teste independente (mesmo split dos scripts
anteriores) em tres decodificacoes: multiclasse, binarios e binarios_soma.

Os resultados parciais sao salvos apos cada fonte, entao uma interrupcao nao
perde o que ja foi concluido.

Requisitos:
    pip install gensim
    (para vetores do NILC: pip install huggingface_hub safetensors)
"""

import gc
import os
import re
from itertools import product

import numpy as np
import pandas as pd

from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_val_score,
    cross_val_predict,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    cohen_kappa_score,
    f1_score,
)

# ============================================================
# 1. CONFIGURACOES
# ============================================================

DATA_PATH = "data/train.xlsx"
TEXT_COL = "resp_text"
TARGET_COL = "clarity"

OUTPUT_DIR = "modelos/resultados_word2vec"
os.makedirs(OUTPUT_DIR, exist_ok=True)

RANDOM_STATE = 123
TEST_SIZE = 0.20
N_SPLITS = 5
N_JOBS = 5  # paralelismo nos folds da CV

# --- Word2Vec treinado no proprio corpus: uma linha = uma configuracao ---
# sg=1: skip-gram | sg=0: CBOW
W2V_CONFIGS = {
    "sg_d100_w5":        dict(vector_size=100, window=5,  sg=1, min_count=2, epochs=30),
    "sg_d200_w5":        dict(vector_size=200, window=5,  sg=1, min_count=2, epochs=30),
    "sg_d300_w5":        dict(vector_size=300, window=5,  sg=1, min_count=2, epochs=30),
    "sg_d200_w10":       dict(vector_size=200, window=10, sg=1, min_count=2, epochs=30),
    "cbow_d200_w5":      dict(vector_size=200, window=5,  sg=0, min_count=2, epochs=30),
    "sg_d200_w5_mc5":    dict(vector_size=200, window=5,  sg=1, min_count=5, epochs=30),
    "sg_d200_w5_ep60":   dict(vector_size=200, window=5,  sg=1, min_count=2, epochs=60),
}
W2V_WORKERS = 1  # 1 = reprodutivel; aumente (ex.: 4) para treinar mais rapido

# --- Vetores pre-treinados do NILC no Hugging Face (opcional) ---
# Repositorios ja confirmados: word2vec-cbow-300d (~1,1 GB), word2vec-cbow-600d.
# Confira outros nomes/dimensoes em huggingface.co/nilc-nlp
PRETRAINED_HF_REPOS = [
    "nilc-nlp/word2vec-cbow-300d",
    "nilc-nlp/word2vec-cbow-600d",
    "nilc-nlp/fasttext-cbow-300d",
]

# --- Arquivo local no formato word2vec (opcional) ---
PRETRAINED_PATH = None
PRETRAINED_BINARY = False  # True se o arquivo for .bin

POOLINGS = ["media", "media_idf", "media_max"]

# Lista curta de C. Deixe [1.0] para nao ter nenhuma busca.
VALORES_C = [0.01, 0.1, 1.0]

BASELINE_F1_TESTE = 0.477827  # LinearSVC word_12 C=0.1 (melhor TF-IDF)


# ============================================================
# 2. TOKENIZACAO, VETORES E VETORIZACAO
# ============================================================

def tokenizar(texto):
    return re.findall(r"\w+", texto.lower())


def treinar_w2v(docs, params):
    from gensim.models import Word2Vec

    modelo = Word2Vec(
        sentences=docs,
        workers=W2V_WORKERS,
        seed=RANDOM_STATE,
        **params,
    )
    return modelo.wv


class VetoresEstaticos:
    """Interface minima (key_to_index, [], vector_size) sobre uma matriz."""

    def __init__(self, vocab, vetores):
        self.key_to_index = {}
        for i, palavra in enumerate(vocab):
            self.key_to_index.setdefault(palavra, i)
        self.vetores = vetores
        self.vector_size = vetores.shape[1]

    def __getitem__(self, palavra):
        return self.vetores[self.key_to_index[palavra]]


def montar_vetores_safetensors(caminho_vetores, caminho_vocab):
    from safetensors.numpy import load_file

    vetores = load_file(caminho_vetores)["embeddings"]
    with open(caminho_vocab, encoding="utf-8") as f:
        vocab = [linha.strip() for linha in f]

    assert len(vocab) == vetores.shape[0], (
        f"Vocabulario ({len(vocab)}) e matriz ({vetores.shape[0]}) "
        "com tamanhos diferentes"
    )
    return VetoresEstaticos(vocab, vetores)


def carregar_pretreinado_hf(repo_id):
    from huggingface_hub import hf_hub_download

    caminho_vetores = hf_hub_download(
        repo_id=repo_id, filename="embeddings.safetensors"
    )
    caminho_vocab = hf_hub_download(repo_id=repo_id, filename="vocab.txt")
    return montar_vetores_safetensors(caminho_vetores, caminho_vocab)


def carregar_pretreinado(caminho, binario):
    from gensim.models import KeyedVectors

    return KeyedVectors.load_word2vec_format(caminho, binary=binario)


def vetorizar(docs, wv, idf, idf_max, pooling):
    dim = wv.vector_size
    largura = dim * 2 if pooling == "media_max" else dim
    saida = np.zeros((len(docs), largura), dtype=np.float32)
    total_tokens, tokens_cobertos, docs_vazios = 0, 0, 0

    for i, tokens in enumerate(docs):
        vetores, pesos = [], []

        for w in tokens:
            total_tokens += 1
            if w in wv.key_to_index:
                tokens_cobertos += 1
                vetores.append(wv[w])
                if pooling == "media_idf":
                    pesos.append(idf.get(w, idf_max))

        if not vetores:
            docs_vazios += 1
            continue

        matriz = np.asarray(vetores)

        if pooling == "media":
            saida[i] = matriz.mean(axis=0)
        elif pooling == "media_idf":
            saida[i] = np.average(matriz, axis=0, weights=pesos)
        else:  # media_max
            saida[i] = np.concatenate([matriz.mean(axis=0), matriz.max(axis=0)])

    cobertura = tokens_cobertos / max(total_tokens, 1)
    return saida, cobertura, docs_vazios


# ============================================================
# 3. DECODIFICADORES ORDINAIS E TUNING DE LIMIARES
# ============================================================

def f1_macro_rapido(y, p, k=3):
    cm = np.bincount(y * k + p, minlength=k * k).reshape(k, k)
    tp = np.diag(cm)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp
    denom = 2 * tp + fp + fn
    f1 = np.where(denom > 0, 2 * tp / np.maximum(denom, 1), 0.0)
    return f1.mean()


def prever_escore(s, t1, t2):
    return np.where(s < t1, 0, np.where(s < t2, 1, 2))


def prever_binarios(s1, s2, t1, t2):
    """s1: 'nao e c1' (alto = nao e c1); s2: 'e c5' (alto = e c5)."""
    e_c1 = s1 < t1
    e_c5 = s2 >= t2

    pred = np.ones(len(s1), dtype=int)
    pred[e_c1] = 0
    pred[e_c5] = 2

    conflito = e_c1 & e_c5
    if conflito.any():
        margem_c1 = t1 - s1[conflito]
        margem_c5 = s2[conflito] - t2
        pred[conflito] = np.where(margem_c1 > margem_c5, 0, 2)

    return pred


def tunar_escore(s, y, n=80):
    candidatos = np.unique(np.quantile(s, np.linspace(0.02, 0.98, n)))
    melhor = (-1.0, None, None)

    for i, t1 in enumerate(candidatos):
        for t2 in candidatos[i + 1:]:
            f1 = f1_macro_rapido(y, prever_escore(s, t1, t2))
            if f1 > melhor[0]:
                melhor = (f1, t1, t2)

    return melhor


def tunar_binarios(s1, s2, y, n=60):
    c1 = np.unique(np.quantile(s1, np.linspace(0.05, 0.95, n)))
    c2 = np.unique(np.quantile(s2, np.linspace(0.05, 0.95, n)))
    melhor = (-1.0, None, None)

    for t1, t2 in product(c1, c2):
        f1 = f1_macro_rapido(y, prever_binarios(s1, s2, t1, t2))
        if f1 > melhor[0]:
            melhor = (f1, t1, t2)

    return melhor


# ============================================================
# 4. DADOS (mesmo split dos scripts anteriores)
# ============================================================

for p in POOLINGS:
    assert p in {"media", "media_idf", "media_max"}, f"Pooling invalido: {p}"

df = pd.read_excel(DATA_PATH)
df = df[[TEXT_COL, TARGET_COL]].copy()
df = df.dropna(subset=[TARGET_COL])
df[TEXT_COL] = df[TEXT_COL].fillna("").astype(str)
df[TARGET_COL] = df[TARGET_COL].astype(str)

ORDEM = {"c1": 0, "c234": 1, "c5": 2}
NOMES = ["c1", "c234", "c5"]

assert set(df[TARGET_COL].unique()) == set(ORDEM), (
    f"Classes inesperadas: {sorted(df[TARGET_COL].unique())}"
)

textos = df[TEXT_COL].tolist()
y_num = df[TARGET_COL].map(ORDEM).to_numpy()

# Split por indices: mesma particao de train_test_split(X, y, ...)
indices = np.arange(len(df))
idx_dev, idx_test = train_test_split(
    indices,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y_num,
)

y_dev = y_num[idx_dev]
y_test = y_num[idx_test]

textos_dev = [textos[i] for i in idx_dev]
textos_test = [textos[i] for i in idx_test]

docs_dev = [tokenizar(t) for t in textos_dev]
docs_test = [tokenizar(t) for t in textos_test]

cv = StratifiedKFold(
    n_splits=N_SPLITS,
    shuffle=True,
    random_state=RANDOM_STATE,
)

print(f"Dev: {len(idx_dev)} | Teste: {len(idx_test)}")

# IDF calculado so no desenvolvimento
tfidf_idf = TfidfVectorizer(
    tokenizer=tokenizar, lowercase=False, token_pattern=None
).fit(textos_dev)
idf = dict(zip(tfidf_idf.get_feature_names_out(), tfidf_idf.idf_))
idf_max = float(tfidf_idf.idf_.max())


# ============================================================
# 5. AVALIACAO
# ============================================================

linhas_tabela = []
detalhes = []


def criar_modelo(C):
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(C=C, max_iter=3000, random_state=RANDOM_STATE),
    )


def avaliar(modelo, decodificacao, pred, extra=""):
    f1 = f1_score(y_test, pred, average="macro")
    f1_cls = f1_score(y_test, pred, average=None, labels=[0, 1, 2])

    linhas_tabela.append({
        "modelo": modelo,
        "decodificacao": decodificacao,
        "f1_macro": f1,
        "accuracy": accuracy_score(y_test, pred),
        "kappa": cohen_kappa_score(y_test, pred),
        "kappa_quadratico": cohen_kappa_score(
            y_test, pred, weights="quadratic"
        ),
        "f1_c1": f1_cls[0],
        "f1_c234": f1_cls[1],
        "f1_c5": f1_cls[2],
        "delta_f1_vs_tfidf": f1 - BASELINE_F1_TESTE,
    })

    detalhes.append(
        "-" * 70 + "\n"
        f"{modelo} | {decodificacao} {extra}\n\n"
        + classification_report(
            y_test, pred, target_names=NOMES, zero_division=0
        )
        + "\nMatriz de confusao (linhas = real, colunas = previsto):\n"
        + f"Ordem: {NOMES}\n"
        + str(confusion_matrix(y_test, pred, labels=[0, 1, 2]))
        + "\n"
    )


def rodar_avaliacao(rotulo, X_dev, X_test):
    # --- Escolha de C por CV (lista curta) ---
    melhor_C, melhor_f1_cv = None, -1.0
    for C in VALORES_C:
        f1_cv = cross_val_score(
            criar_modelo(C), X_dev, y_dev,
            cv=cv, scoring="f1_macro", n_jobs=N_JOBS,
        ).mean()
        print(f"  C={C}: F1-macro CV = {f1_cv:.4f}")
        if f1_cv > melhor_f1_cv:
            melhor_C, melhor_f1_cv = C, f1_cv

    print(f"  -> C escolhido: {melhor_C} (F1 CV = {melhor_f1_cv:.4f})")
    rotulo_c = f"{rotulo} (C={melhor_C})"

    # --- 1. Multiclasse ---
    pred = criar_modelo(melhor_C).fit(X_dev, y_dev).predict(X_test)
    avaliar(rotulo_c, "multiclasse", pred)
    print(
        "  F1 teste multiclasse: "
        f"{f1_score(y_test, pred, average='macro'):.4f}"
    )

    # --- 2 e 3. Ordinal ---
    y1_dev = (y_dev >= 1).astype(int)  # nao e c1
    y2_dev = (y_dev >= 2).astype(int)  # e c5

    s1_oof = cross_val_predict(
        criar_modelo(melhor_C), X_dev, y1_dev,
        cv=cv, method="decision_function", n_jobs=N_JOBS,
    )
    s2_oof = cross_val_predict(
        criar_modelo(melhor_C), X_dev, y2_dev,
        cv=cv, method="decision_function", n_jobs=N_JOBS,
    )

    m1 = criar_modelo(melhor_C).fit(X_dev, y1_dev)
    m2 = criar_modelo(melhor_C).fit(X_dev, y2_dev)
    s1_test = m1.decision_function(X_test)
    s2_test = m2.decision_function(X_test)

    f1_oof, t1, t2 = tunar_binarios(s1_oof, s2_oof, y_dev)
    pred = prever_binarios(s1_test, s2_test, t1, t2)
    avaliar(
        rotulo_c, "binarios", pred,
        extra=f"(F1 OOF={f1_oof:.4f}, t1={t1:.4f}, t2={t2:.4f})",
    )

    f1_oof, t1, t2 = tunar_escore(s1_oof + s2_oof, y_dev)
    pred = prever_escore(s1_test + s2_test, t1, t2)
    avaliar(
        rotulo_c, "binarios_soma", pred,
        extra=f"(F1 OOF={f1_oof:.4f}, t1={t1:.4f}, t2={t2:.4f})",
    )


def salvar_parcial():
    pd.DataFrame(linhas_tabela).to_csv(
        os.path.join(OUTPUT_DIR, "comparacao_word2vec.csv"),
        index=False,
        encoding="utf-8-sig",
    )


# ============================================================
# 6. EXECUCAO
# ============================================================

def construir_fontes():
    fontes = []

    for nome, params in W2V_CONFIGS.items():
        fontes.append((
            f"w2v_{nome}",
            lambda p=params: treinar_w2v(docs_dev, p),
        ))

    for repo in PRETRAINED_HF_REPOS:
        fontes.append((
            f"nilc_{repo.split('/')[-1]}",
            lambda r=repo: carregar_pretreinado_hf(r),
        ))

    if PRETRAINED_PATH:
        fontes.append((
            "w2v_prev",
            lambda: carregar_pretreinado(PRETRAINED_PATH, PRETRAINED_BINARY),
        ))

    return fontes


fontes = construir_fontes()
print(f"\nFontes de vetores: {len(fontes)} | Poolings: {POOLINGS}")

for nome_fonte, carregar in fontes:
    print("\n" + "#" * 70)
    print(f"# FONTE: {nome_fonte}")
    print("#" * 70)

    try:
        wv = carregar()
    except Exception as erro:
        print(f"AVISO: nao foi possivel carregar '{nome_fonte}': {erro}")
        print("Fonte pulada.")
        continue

    print(f"Vocabulario: {len(wv.key_to_index)} palavras | dim={wv.vector_size}")

    for pooling in POOLINGS:
        rotulo = f"{nome_fonte} | {pooling}"

        print("\n" + "=" * 70)
        print(f">>> {rotulo}")

        X_dev, cob_dev, vazios_dev = vetorizar(
            docs_dev, wv, idf, idf_max, pooling
        )
        X_test, cob_test, vazios_test = vetorizar(
            docs_test, wv, idf, idf_max, pooling
        )

        if pooling == POOLINGS[0]:
            print(
                f"  Cobertura de tokens: dev={cob_dev:.1%}, "
                f"teste={cob_test:.1%} | textos sem nenhuma palavra no "
                f"vocabulario: dev={vazios_dev}, teste={vazios_test}"
            )

        rodar_avaliacao(rotulo, X_dev, X_test)

    del wv
    gc.collect()
    salvar_parcial()


# ============================================================
# 7. RESULTADOS
# ============================================================

if not linhas_tabela:
    raise RuntimeError("Nenhuma fonte foi avaliada com sucesso.")

df_tab = pd.DataFrame(linhas_tabela)
salvar_parcial()

resumo = df_tab.round(4).to_string(index=False)
ranking = (
    df_tab.sort_values("f1_macro", ascending=False)
    .head(15)
    .round(4)
    .to_string(index=False)
)

print("\n" + "=" * 70)
print(f"TOP 15 POR F1-MACRO NO TESTE (melhor TF-IDF: {BASELINE_F1_TESTE:.4f})")
print("=" * 70)
print(ranking)

with open(
    os.path.join(OUTPUT_DIR, "resultado_word2vec.txt"),
    "w",
    encoding="utf-8",
) as f:
    f.write("WORD2VEC + REGRESSAO LOGISTICA NO TESTE INDEPENDENTE\n")
    f.write("=" * 70 + "\n\n")
    f.write(f"TOP 15 POR F1-MACRO (melhor TF-IDF: {BASELINE_F1_TESTE:.4f})\n\n")
    f.write(ranking + "\n\n")
    f.write("TODOS OS RESULTADOS\n\n")
    f.write(resumo + "\n\n")
    f.write("\n".join(detalhes))