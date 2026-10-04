"""
Comparacao de modelos fixos, SEM grid search de hiperparametros:

- LR_word12 : TF-IDF word (1,2), min_df=2 + LogisticRegression C=0.1
- LR_word1  : TF-IDF word (1,1), min_df=1 + LogisticRegression C=0.1
              (configuracao do ID648; SelectKBest k="all" nao remove nada, foi omitido)
- SVC_word12: TF-IDF word (1,2), min_df=2 + LinearSVC C=0.1 (ID45, so como referencia)

Cada modelo e avaliado em tres decodificacoes, sempre no mesmo teste independente:

1. multiclasse : predicao padrao de 3 classes
2. binarios    : c1 vs resto e c5 vs resto, com limiares tunados (c234 = intervalo)
3. binarios_soma: soma dos dois escores binarios como escore ordinal unico,
                  com dois limiares tunados

Os limiares sao escolhidos em predicoes out-of-fold sobre o conjunto de
desenvolvimento; o teste so e usado uma vez, no final.
"""

import os
from itertools import product

import numpy as np
import pandas as pd

from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_val_predict,
)
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
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

OUTPUT_DIR = "modelos/tf_idf_binario"
os.makedirs(OUTPUT_DIR, exist_ok=True)

RANDOM_STATE = 123
TEST_SIZE = 0.20
N_SPLITS = 5

ORDEM = {"c1": 0, "c234": 1, "c5": 2}
NOMES = ["c1", "c234", "c5"]

TFIDF_WORD12 = dict(
    analyzer="word", ngram_range=(1, 2), min_df=2, max_df=1.0,
    max_features=None, sublinear_tf=True, use_idf=True, norm="l2",
)

TFIDF_WORD1 = dict(
    analyzer="word", ngram_range=(1, 1), min_df=1, max_df=1.0,
    max_features=None, sublinear_tf=True, use_idf=True, norm="l2",
)

MODELOS = {
    "LR_word12": {"tfidf": TFIDF_WORD12, "clf": "LogisticRegression", "C": 0.1},
    "LR_word1": {"tfidf": TFIDF_WORD1, "clf": "LogisticRegression", "C": 0.1},
    # Referencia (melhor modelo do grid anterior). Remova se nao quiser.
    "SVC_word12": {"tfidf": TFIDF_WORD12, "clf": "LinearSVC", "C": 0.1},
}

BASELINE = "SVC_word12"

# Numeros ja obtidos por voce, so para conferir se o split/modelo batem
# (LR_word12 nao tem referencia previa)
REFERENCIAS_F1_TESTE = {"SVC_word12": 0.477827, "LR_word1": 0.4507}


def criar_clf(nome, C):
    if nome == "LinearSVC":
        return LinearSVC(
            C=C, class_weight=None,
            random_state=RANDOM_STATE, max_iter=10000,
        )

    if nome == "LogisticRegression":
        # Versoes recentes do sklearn nao aceitam liblinear em multiclasse.
        # OneVsRest mantem o mesmo esquema (um-contra-resto) em qualquer versao
        # e, no caso binario, se comporta como uma regressao logistica simples.
        return OneVsRestClassifier(
            LogisticRegression(
                C=C, solver="liblinear",
                max_iter=3000, random_state=RANDOM_STATE,
            )
        )

    raise ValueError(f"Classificador desconhecido: {nome}")


def construir(cfg):
    return Pipeline([
        ("tfidf", TfidfVectorizer(**cfg["tfidf"])),
        ("clf", criar_clf(cfg["clf"], cfg["C"])),
    ])


# ============================================================
# 2. DECODIFICADORES E TUNING DE LIMIARES
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
# 3. DADOS (mesmo split dos scripts anteriores)
# ============================================================

df = pd.read_excel(DATA_PATH)
df = df[[TEXT_COL, TARGET_COL]].copy()
df = df.dropna(subset=[TARGET_COL])
df[TEXT_COL] = df[TEXT_COL].fillna("").astype(str)
df[TARGET_COL] = df[TARGET_COL].astype(str)

assert set(df[TARGET_COL].unique()) == set(ORDEM), (
    f"Classes inesperadas: {sorted(df[TARGET_COL].unique())}"
)

X = df[TEXT_COL]
y = df[TARGET_COL]

X_dev, X_test, y_dev, y_test = train_test_split(
    X, y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y,
)

y_dev_num = y_dev.map(ORDEM).to_numpy()
y_test_num = y_test.map(ORDEM).to_numpy()

cv = StratifiedKFold(
    n_splits=N_SPLITS,
    shuffle=True,
    random_state=RANDOM_STATE,
)

print(f"Dev: {len(X_dev)} | Teste: {len(X_test)}")


# ============================================================
# 4. AVALIACAO
# ============================================================

linhas_tabela = []
detalhes = []


def avaliar(modelo, decodificacao, pred, extra=""):
    f1 = f1_score(y_test_num, pred, average="macro")
    f1_cls = f1_score(y_test_num, pred, average=None, labels=[0, 1, 2])

    linhas_tabela.append({
        "modelo": modelo,
        "decodificacao": decodificacao,
        "f1_macro": f1,
        "accuracy": accuracy_score(y_test_num, pred),
        "kappa": cohen_kappa_score(y_test_num, pred),
        "kappa_quadratico": cohen_kappa_score(
            y_test_num, pred, weights="quadratic"
        ),
        "f1_c1": f1_cls[0],
        "f1_c234": f1_cls[1],
        "f1_c5": f1_cls[2],
    })

    detalhes.append(
        "-" * 70 + "\n"
        f"{modelo} | {decodificacao} {extra}\n\n"
        + classification_report(
            y_test_num, pred, target_names=NOMES, zero_division=0
        )
        + "\nMatriz de confusao (linhas = real, colunas = previsto):\n"
        + f"Ordem: {NOMES}\n"
        + str(confusion_matrix(y_test_num, pred, labels=[0, 1, 2]))
        + "\n"
    )


for nome, cfg in MODELOS.items():
    print(f"\n>>> {nome}: {cfg['clf']} C={cfg['C']}")

    # --- 1. Multiclasse padrao ---
    modelo = construir(cfg).fit(X_dev, y_dev_num)
    pred = modelo.predict(X_test)
    avaliar(nome, "multiclasse", pred)

    f1_atual = f1_score(y_test_num, pred, average="macro")
    ref = REFERENCIAS_F1_TESTE.get(nome)
    ref_txt = f"{ref:.4f}" if ref is not None else "sem referencia"
    print(f"F1 teste multiclasse: {f1_atual:.4f} (referencia sua: {ref_txt})")

    # --- 2 e 3. Ordinal: escores OOF para tunar limiares ---
    y1_dev = (y_dev_num >= 1).astype(int)  # nao e c1
    y2_dev = (y_dev_num >= 2).astype(int)  # e c5

    s1_oof = cross_val_predict(
        construir(cfg), X_dev, y1_dev, cv=cv, method="decision_function"
    )
    s2_oof = cross_val_predict(
        construir(cfg), X_dev, y2_dev, cv=cv, method="decision_function"
    )

    m1 = construir(cfg).fit(X_dev, y1_dev)
    m2 = construir(cfg).fit(X_dev, y2_dev)
    s1_test = m1.decision_function(X_test)
    s2_test = m2.decision_function(X_test)

    f1_oof, t1, t2 = tunar_binarios(s1_oof, s2_oof, y_dev_num)
    pred = prever_binarios(s1_test, s2_test, t1, t2)
    avaliar(
        nome, "binarios", pred,
        extra=f"(F1 OOF={f1_oof:.4f}, t1={t1:.4f}, t2={t2:.4f})",
    )

    f1_oof, t1, t2 = tunar_escore(s1_oof + s2_oof, y_dev_num)
    pred = prever_escore(s1_test + s2_test, t1, t2)
    avaliar(
        nome, "binarios_soma", pred,
        extra=f"(F1 OOF={f1_oof:.4f}, t1={t1:.4f}, t2={t2:.4f})",
    )


# ============================================================
# 5. RESULTADOS
# ============================================================

df_tab = pd.DataFrame(linhas_tabela)

baseline = df_tab[
    (df_tab["modelo"] == BASELINE) & (df_tab["decodificacao"] == "multiclasse")
]["f1_macro"].iloc[0]
df_tab["delta_f1_vs_baseline"] = df_tab["f1_macro"] - baseline

df_tab.to_csv(
    os.path.join(OUTPUT_DIR, "comparacao_modelos.csv"),
    index=False,
    encoding="utf-8-sig",
)

resumo = df_tab.round(4).to_string(index=False)

print("\n" + "=" * 70)
print("RESUMO NO TESTE INDEPENDENTE")
print("=" * 70)
print(resumo)

with open(
    os.path.join(OUTPUT_DIR, "resultado_comparacao.txt"),
    "w",
    encoding="utf-8",
) as f:
    f.write("COMPARACAO DE MODELOS NO TESTE INDEPENDENTE\n")
    f.write("=" * 70 + "\n\n")
    f.write(resumo + "\n\n")
    f.write("\n".join(detalhes))