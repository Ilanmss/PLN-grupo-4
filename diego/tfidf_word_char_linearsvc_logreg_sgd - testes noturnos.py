"""
Experimentos de classificação de texto com diferentes representações TF-IDF
e classificadores lineares.

O script compara diferentes estratégias de representação textual — unigramas
de palavras, unigramas e bigramas, n-gramas de caracteres, n-gramas de
caracteres delimitados por palavras (char_wb) e a combinação de palavras
com caracteres — para investigar quais capturam melhor os padrões relevantes
à classificação das respostas.

Também avalia diferentes classificadores lineares:
- LinearSVC;
- LogisticRegression;
- SGDClassifier com função de perda hinge;
- SGDClassifier com função de perda log_loss.

Para cada classificador, são testados diferentes valores de regularização
(C ou alpha) e o uso ou não de ponderação balanceada das classes
(class_weight="balanced"). Esses testes permitem analisar o impacto da
regularização e do tratamento das classes sobre o desempenho preditivo.

As configurações são avaliadas por validação cruzada estratificada de 5 folds,
utilizando F1-macro como métrica principal, por atribuir o mesmo peso ao
desempenho de cada classe. A média e o desvio padrão dos resultados entre
os folds são registrados, permitindo comparar tanto o desempenho quanto
sua variação entre diferentes partições dos dados.

O conjunto de teste independente é separado previamente e não participa
da seleção das configurações, evitando seu uso durante o ajuste dos modelos.
Após a validação cruzada, os resultados são organizados em arquivos CSV
e TXT, incluindo o desempenho individual de cada experimento, o ranking
das configurações e um diário com a ordem e o andamento da execução.

O script também utiliza checkpoints para permitir a retomada dos experimentos
interrompidos, sem precisar executar novamente as configurações já concluídas.
"""


import os
import json
import time
import hashlib
import traceback
from datetime import datetime
from itertools import product

import numpy as np
import pandas as pd
import joblib

from sklearn.base import clone
from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_validate
)
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion

from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression, SGDClassifier

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    cohen_kappa_score,
    f1_score
)


# ============================================================
# 1. CONFIGURACOES
# ============================================================

DATA_PATH = "data/train.xlsx"

TEXT_COL = "resp_text"
TARGET_COL = "clarity"

OUTPUT_DIR = "resultados_experimentos_noturnos"
INDIVIDUAL_DIR = os.path.join(
    OUTPUT_DIR,
    "resultados_individuais"
)

RANDOM_STATE = 123
TEST_SIZE = 0.20
N_SPLITS = 5

RESUME = True

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(INDIVIDUAL_DIR, exist_ok=True)

LOG_PATH = os.path.join(
    OUTPUT_DIR,
    "00_diario_execucao.txt"
)

CSV_PATH = os.path.join(
    OUTPUT_DIR,
    "resultados_grid.csv"
)

CHECKPOINT_PATH = os.path.join(
    OUTPUT_DIR,
    "checkpoint.json"
)


# ============================================================
# 2. FUNCOES DE LOG E CHECKPOINT
# ============================================================

def agora():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def registrar_log(mensagem):
    linha = f"[{agora()}] {mensagem}"

    print(linha, flush=True)

    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(linha + "\n")


def salvar_json_atomico(caminho, dados):
    temporario = caminho + ".tmp"

    with open(temporario, "w", encoding="utf-8") as f:
        json.dump(
            dados,
            f,
            ensure_ascii=False,
            indent=4,
            default=str
        )

    os.replace(temporario, caminho)


def salvar_txt_config(config_id, conteudo):
    caminho = os.path.join(
        INDIVIDUAL_DIR,
        f"teste_{config_id:04d}.txt"
    )

    with open(caminho, "w", encoding="utf-8") as f:
        f.write(conteudo)


def carregar_checkpoint():
    if not RESUME or not os.path.exists(CHECKPOINT_PATH):
        return {}

    with open(CHECKPOINT_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def salvar_checkpoint(resultados, assinatura):
    dados = {
        "assinatura": assinatura,
        "atualizado_em": agora(),
        "resultados": resultados
    }

    salvar_json_atomico(CHECKPOINT_PATH, dados)


def assinatura_experimento(configuracoes):
    texto = json.dumps(
        configuracoes,
        sort_keys=True,
        ensure_ascii=False,
        default=str
    )

    return hashlib.md5(texto.encode("utf-8")).hexdigest()


# ============================================================
# 3. REPRESENTACOES DE TEXTO
# ============================================================

def criar_representacao(nome):
    if nome == "word_1":
        return TfidfVectorizer(
            analyzer="word",
            ngram_range=(1, 1),
            min_df=2,
            sublinear_tf=True,
            use_idf=True,
            norm="l2",
            max_features=None
        )

    if nome == "word_12":
        return TfidfVectorizer(
            analyzer="word",
            ngram_range=(1, 2),
            min_df=2,
            sublinear_tf=True,
            use_idf=True,
            norm="l2",
            max_features=None
        )

    if nome == "char_35":
        return TfidfVectorizer(
            analyzer="char",
            ngram_range=(3, 5),
            min_df=2,
            sublinear_tf=True,
            use_idf=True,
            norm="l2",
            max_features=None
        )

    if nome == "charwb_35":
        return TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(3, 5),
            min_df=2,
            sublinear_tf=True,
            use_idf=True,
            norm="l2",
            max_features=None
        )

    if nome == "word_charwb":
        return FeatureUnion([
            (
                "word",
                TfidfVectorizer(
                    analyzer="word",
                    ngram_range=(1, 2),
                    min_df=2,
                    sublinear_tf=True,
                    use_idf=True,
                    norm="l2",
                    max_features=None
                )
            ),
            (
                "charwb",
                TfidfVectorizer(
                    analyzer="char_wb",
                    ngram_range=(3, 5),
                    min_df=2,
                    sublinear_tf=True,
                    use_idf=True,
                    norm="l2",
                    max_features=None
                )
            )
        ])

    raise ValueError(f"Representacao desconhecida: {nome}")


# ============================================================
# 4. DEFINICAO DOS CLASSIFICADORES
# ============================================================

def criar_classificador(nome, parametros):
    if nome == "LinearSVC":
        return LinearSVC(
            C=parametros["C"],
            class_weight=parametros["class_weight"],
            random_state=RANDOM_STATE,
            max_iter=10000
        )

    if nome == "LogisticRegression":
        return LogisticRegression(
            C=parametros["C"],
            class_weight=parametros["class_weight"],
            max_iter=3000,
            solver="liblinear",
            random_state=RANDOM_STATE
        )

    if nome == "SGD_hinge":
        return SGDClassifier(
            loss="hinge",
            alpha=parametros["alpha"],
            class_weight=parametros["class_weight"],
            max_iter=2000,
            tol=1e-3,
            random_state=RANDOM_STATE
        )

    if nome == "SGD_log_loss":
        return SGDClassifier(
            loss="log_loss",
            alpha=parametros["alpha"],
            class_weight=parametros["class_weight"],
            max_iter=2000,
            tol=1e-3,
            random_state=RANDOM_STATE
        )

    raise ValueError(f"Classificador desconhecido: {nome}")


# ============================================================
# 5. GERAR TODAS AS CONFIGURACOES
# ============================================================

REPRESENTACOES = [
    "word_1",
    "word_12",
    "char_35",
    "charwb_35",
    "word_charwb"
]

PESOS_CLASSES = [
    None,
    "balanced"
]

VALORES_C = [
    0.01,
    0.03,
    0.1,
    0.3,
    1,
    3,
    10
]

VALORES_ALPHA = [
    0.00001,
    0.0001,
    0.001
]

configuracoes = []

for representacao in REPRESENTACOES:

    for classificador in [
        "LinearSVC",
        "LogisticRegression"
    ]:
        for C, peso in product(
            VALORES_C,
            PESOS_CLASSES
        ):
            configuracoes.append({
                "representacao": representacao,
                "classificador": classificador,
                "parametros": {
                    "C": C,
                    "class_weight": peso
                }
            })

    for classificador in [
        "SGD_hinge",
        "SGD_log_loss"
    ]:
        for alpha, peso in product(
            VALORES_ALPHA,
            PESOS_CLASSES
        ):
            configuracoes.append({
                "representacao": representacao,
                "classificador": classificador,
                "parametros": {
                    "alpha": alpha,
                    "class_weight": peso
                }
            })


# IDs fixos para permitir retomada
for i, config in enumerate(configuracoes, start=1):
    config["id"] = i


assinatura = assinatura_experimento({
    "data_path": DATA_PATH,
    "text_col": TEXT_COL,
    "target_col": TARGET_COL,
    "random_state": RANDOM_STATE,
    "test_size": TEST_SIZE,
    "n_splits": N_SPLITS,
    "configuracoes": configuracoes
})


# ============================================================
# 6. CARREGAR DADOS
# ============================================================

registrar_log("=" * 70)
registrar_log("INICIO DOS EXPERIMENTOS")
registrar_log(f"Dataset: {DATA_PATH}")
registrar_log(f"Total de configuracoes: {len(configuracoes)}")
registrar_log(f"Validacao cruzada: {N_SPLITS} folds")
registrar_log("=" * 70)

df = pd.read_excel(DATA_PATH)

df = df[[TEXT_COL, TARGET_COL]].copy()

df = df.dropna(subset=[TARGET_COL])

df[TEXT_COL] = df[TEXT_COL].fillna("").astype(str)
df[TARGET_COL] = df[TARGET_COL].astype(str)

X = df[TEXT_COL]
y = df[TARGET_COL]

X_dev, X_test, y_dev, y_test = train_test_split(
    X,
    y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y
)

registrar_log(f"Total de exemplos: {len(df)}")
registrar_log(f"Desenvolvimento: {len(X_dev)}")
registrar_log(f"Teste independente: {len(X_test)}")
registrar_log(f"Classes: {sorted(y.unique())}")


# ============================================================
# 7. PREPARAR RETOMADA
# ============================================================

checkpoint = carregar_checkpoint()

if checkpoint:
    if checkpoint.get("assinatura") == assinatura:
        resultados = checkpoint.get("resultados", {})
        registrar_log(
            f"Checkpoint encontrado. "
            f"{len(resultados)} configuracoes ja concluidas."
        )
    else:
        raise ValueError(
            "O checkpoint pertence a outra configuracao. "
            "Use uma nova OUTPUT_DIR para este experimento."
        )
else:
    resultados = {}

cv = StratifiedKFold(
    n_splits=N_SPLITS,
    shuffle=True,
    random_state=RANDOM_STATE
)


# ============================================================
# 8. EXECUTAR EXPERIMENTOS
# ============================================================

total = len(configuracoes)
inicio_total = time.time()

for posicao, config in enumerate(configuracoes, start=1):

    config_id = config["id"]
    chave = str(config_id)

    if chave in resultados:
        continue

    representacao = config["representacao"]
    nome_modelo = config["classificador"]
    parametros = config["parametros"]

    registrar_log(
        f"TESTE {posicao}/{total} | "
        f"ID={config_id} | "
        f"Representacao={representacao} | "
        f"Modelo={nome_modelo} | "
        f"Parametros={parametros}"
    )

    inicio = time.time()

    texto_config = (
        f"TESTE {config_id}/{total}\n"
        f"Inicio: {agora()}\n"
        f"Representacao: {representacao}\n"
        f"Classificador: {nome_modelo}\n"
        f"Parametros: {json.dumps(parametros, ensure_ascii=False)}\n"
        f"Folds: {N_SPLITS}\n"
        f"Status: EM EXECUCAO\n"
    )

    salvar_txt_config(config_id, texto_config)

    try:
        vetor = criar_representacao(representacao)
        classificador = criar_classificador(
            nome_modelo,
            parametros
        )

        pipeline = Pipeline([
            ("tfidf", vetor),
            ("classificador", classificador)
        ])

        scores = cross_validate(
            pipeline,
            X_dev,
            y_dev,
            cv=cv,
            scoring={
                "f1_macro": "f1_macro",
                "accuracy": "accuracy"
            },
            n_jobs=1,
            return_train_score=False,
            error_score="raise"
        )

        f1_folds = scores["test_f1_macro"].tolist()
        acc_folds = scores["test_accuracy"].tolist()

        media_f1 = float(np.mean(f1_folds))
        desvio_f1 = float(np.std(f1_folds, ddof=1))
        media_acc = float(np.mean(acc_folds))

        duracao = time.time() - inicio

        resultado = {
            "id": config_id,
            "representacao": representacao,
            "classificador": nome_modelo,
            "parametros": json.dumps(
                parametros,
                ensure_ascii=False
            ),
            "f1_macro_mean": media_f1,
            "f1_macro_std": desvio_f1,
            "accuracy_mean": media_acc,
            "fold_scores": json.dumps(f1_folds),
            "accuracy_folds": json.dumps(acc_folds),
            "duracao_segundos": duracao,
            "status": "concluido"
        }

        resultados[chave] = resultado

        texto_resultado = (
            f"TESTE {config_id}/{total}\n"
            f"Inicio: {agora()}\n"
            f"Representacao: {representacao}\n"
            f"Classificador: {nome_modelo}\n"
            f"Parametros: {json.dumps(parametros, ensure_ascii=False)}\n\n"
            f"RESULTADOS DA VALIDACAO CRUZADA\n"
            f"F1-macro por fold: {f1_folds}\n"
            f"F1-macro medio: {media_f1:.6f}\n"
            f"Desvio padrao: {desvio_f1:.6f}\n"
            f"Accuracy media: {media_acc:.6f}\n"
            f"Duracao: {duracao / 60:.2f} minutos\n"
            f"Status: CONCLUIDO\n"
        )

        salvar_txt_config(config_id, texto_resultado)

        # Salva CSV e checkpoint apos cada configuracao
        df_resultados = pd.DataFrame(resultados.values())
        df_resultados = df_resultados.sort_values(
            "f1_macro_mean",
            ascending=False
        )
        df_resultados.to_csv(
            CSV_PATH,
            index=False,
            encoding="utf-8-sig"
        )

        salvar_checkpoint(resultados, assinatura)

        registrar_log(
            f"CONCLUIDO ID={config_id} | "
            f"F1-macro={media_f1:.5f} | "
            f"DP={desvio_f1:.5f} | "
            f"Duracao={duracao / 60:.2f} min"
        )

    except Exception as erro:
        duracao = time.time() - inicio

        erro_texto = traceback.format_exc()

        resultado = {
            "id": config_id,
            "representacao": representacao,
            "classificador": nome_modelo,
            "parametros": json.dumps(
                parametros,
                ensure_ascii=False
            ),
            "f1_macro_mean": np.nan,
            "f1_macro_std": np.nan,
            "accuracy_mean": np.nan,
            "fold_scores": "",
            "accuracy_folds": "",
            "duracao_segundos": duracao,
            "status": "erro",
            "erro": str(erro)
        }

        resultados[chave] = resultado

        salvar_txt_config(
            config_id,
            f"TESTE {config_id}/{total}\n"
            f"Status: ERRO\n"
            f"Erro: {erro}\n\n"
            f"TRACEBACK:\n{erro_texto}"
        )

        pd.DataFrame(resultados.values()).to_csv(
            CSV_PATH,
            index=False,
            encoding="utf-8-sig"
        )

        salvar_checkpoint(resultados, assinatura)

        registrar_log(
            f"ERRO ID={config_id}: {erro}"
        )


# ============================================================
# 9. RANKING FINAL
# ============================================================

df_resultados = pd.DataFrame(resultados.values())

df_validos = df_resultados[
    (df_resultados["status"] == "concluido")
].copy()

df_validos = df_validos.sort_values(
    "f1_macro_mean",
    ascending=False
)

ranking_path = os.path.join(
    OUTPUT_DIR,
    "melhores_configuracoes.csv"
)

df_validos.to_csv(
    ranking_path,
    index=False,
    encoding="utf-8-sig"
)

ranking_txt_path = os.path.join(
    OUTPUT_DIR,
    "01_ranking_final.txt"
)

with open(ranking_txt_path, "w", encoding="utf-8") as f:
    f.write("RANKING FINAL DOS EXPERIMENTOS\n")
    f.write("=" * 70 + "\n\n")

    f.write(f"Configuracoes executadas: {len(resultados)}\n")
    f.write(f"Configuracoes validas: {len(df_validos)}\n\n")

    for rank, (_, row) in enumerate(
        df_validos.head(30).iterrows(),
        start=1
    ):
        f.write(
            f"{rank}. ID={row['id']} | "
            f"F1={row['f1_macro_mean']:.6f} | "
            f"DP={row['f1_macro_std']:.6f}\n"
        )
        f.write(
            f"   Representacao: {row['representacao']}\n"
        )
        f.write(
            f"   Modelo: {row['classificador']}\n"
        )
        f.write(
            f"   Parametros: {row['parametros']}\n\n"
        )

if len(df_validos) == 0:
    registrar_log("Nenhuma configuracao foi concluida com sucesso.")
    raise RuntimeError("Nao ha resultados validos para avaliar.")


# ============================================================
# 10. TREINAR MELHOR CONFIGURACAO E TESTAR UMA VEZ
# ============================================================

melhor = df_validos.iloc[0]

melhor_id = int(melhor["id"])
melhor_config = configuracoes[melhor_id - 1]

registrar_log(
    f"Melhor configuracao: ID={melhor_id}, "
    f"F1-CV={melhor['f1_macro_mean']:.6f}"
)

melhor_pipeline = Pipeline([
    (
        "tfidf",
        criar_representacao(
            melhor_config["representacao"]
        )
    ),
    (
        "classificador",
        criar_classificador(
            melhor_config["classificador"],
            melhor_config["parametros"]
        )
    )
])

melhor_pipeline.fit(X_dev, y_dev)

y_pred = melhor_pipeline.predict(X_test)

f1_teste = f1_score(
    y_test,
    y_pred,
    average="macro"
)

acc_teste = accuracy_score(y_test, y_pred)

kappa_teste = cohen_kappa_score(
    y_test,
    y_pred
)

relatorio = classification_report(
    y_test,
    y_pred,
    zero_division=0
)

matriz = confusion_matrix(
    y_test,
    y_pred,
    labels=sorted(y.unique())
)

modelo_path = os.path.join(
    OUTPUT_DIR,
    "melhor_modelo.joblib"
)

joblib.dump(melhor_pipeline, modelo_path)

predicoes_path = os.path.join(
    OUTPUT_DIR,
    "predicoes_teste.csv"
)

pd.DataFrame({
    "texto": X_test,
    "classe_real": y_test,
    "classe_prevista": y_pred
}).to_csv(
    predicoes_path,
    index=False,
    encoding="utf-8-sig"
)

teste_txt_path = os.path.join(
    OUTPUT_DIR,
    "resultado_teste_final.txt"
)

with open(teste_txt_path, "w", encoding="utf-8") as f:
    f.write("RESULTADO FINAL NO TESTE INDEPENDENTE\n")
    f.write("=" * 70 + "\n\n")

    f.write(f"Data/hora: {agora()}\n")
    f.write(f"ID da configuracao: {melhor_id}\n")
    f.write(
        f"Representacao: {melhor_config['representacao']}\n"
    )
    f.write(
        f"Classificador: {melhor_config['classificador']}\n"
    )
    f.write(
        f"Parametros: {melhor_config['parametros']}\n\n"
    )

    f.write("VALIDACAO CRUZADA\n")
    f.write(
        f"F1-macro medio: {melhor['f1_macro_mean']:.6f}\n"
    )
    f.write(
        f"Desvio padrao: {melhor['f1_macro_std']:.6f}\n\n"
    )

    f.write("TESTE INDEPENDENTE\n")
    f.write(f"F1-macro: {f1_teste:.6f}\n")
    f.write(f"Accuracy: {acc_teste:.6f}\n")
    f.write(f"Cohen's kappa: {kappa_teste:.6f}\n\n")

    f.write("RELATORIO DE CLASSIFICACAO\n")
    f.write(relatorio)
    f.write("\n\nMATRIZ DE CONFUSAO\n")
    f.write(f"Ordem das classes: {sorted(y.unique())}\n")
    f.write(str(matriz))
    f.write("\n")

registrar_log("=" * 70)
registrar_log("EXPERIMENTOS FINALIZADOS")
registrar_log(f"Melhor F1 CV: {melhor['f1_macro_mean']:.6f}")
registrar_log(f"F1 teste: {f1_teste:.6f}")
registrar_log(f"Accuracy teste: {acc_teste:.6f}")
registrar_log(f"Kappa teste: {kappa_teste:.6f}")
registrar_log(f"Resultados salvos em: {OUTPUT_DIR}")
registrar_log("=" * 70)