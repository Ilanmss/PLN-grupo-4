# ============================================================
# GRID SEARCH SEM CROSS-VALIDATION + TOP 5 + CROSS-VALIDATION
#
# Modelo:
#   TF-IDF + SelectKBest(chi2) + LinearSVC
#
# Etapa 1:
#   Grid Search em uma divisão fixa de treino/validação
#
# Etapa 2:
#   Seleção dos 5 melhores conjuntos de hiperparâmetros
#
# Etapa 3:
#   Validação cruzada dos 5 finalistas
#
# Etapa 4:
#   Treinamento final e avaliação no teste independente
#
# Divisão dos dados:
#   60% treino
#   20% validação
#   20% teste independente
#
# Métrica principal:
#   F1-macro
#
# Recursos:
#   - Checkpoint periódico do Grid Search
#   - Retomada automática após interrupção
#   - Checkpoint após cada modelo da validação cruzada
#   - Mensagens de acompanhamento no console
#   - Salvamento de métricas, previsões e modelo final
# ============================================================


# ============================================================
# 1. IMPORTAÇÕES
# ============================================================

import os
import json
import time
import hashlib
import joblib
import numpy as np
import pandas as pd

from datetime import datetime

from sklearn.base import clone

from sklearn.model_selection import (
    train_test_split,
    ParameterGrid,
    StratifiedKFold,
    cross_val_score
)

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.svm import LinearSVC

from sklearn.metrics import (
    f1_score,
    accuracy_score,
    cohen_kappa_score,
    classification_report,
    confusion_matrix
)


# ============================================================
# 2. CONFIGURAÇÕES GERAIS
# ============================================================

DATA_PATH = "data/train.xlsx"

TEXT_COLUMN = "resp_text"
TARGET_COLUMN = "clarity"

OUTPUT_DIR = "modelos/tfidf_selectkbest_linearsvc"

RANDOM_STATE = 42

# Divisão dos dados
TEST_SIZE = 0.20
VALIDATION_SIZE = 0.25

# 25% dos 80% restantes = 20% do dataset total.
# Resultado final:
# 60% treino, 20% validação, 20% teste.

N_TOP_MODELS = 5
N_CV_SPLITS = 5

# Salvamento periódico do Grid Search
SAVE_EVERY = 25

# Frequência das mensagens de progresso
LOG_EVERY = 10

# True: retoma resultados anteriores
# False: ignora checkpoints anteriores
RESUME = True

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# 3. FUNÇÕES AUXILIARES
# ============================================================

def formatar_tempo(segundos):
    """
    Converte segundos para HH:MM:SS.
    """
    segundos = int(max(0, segundos))

    horas, resto = divmod(segundos, 3600)
    minutos, segundos = divmod(resto, 60)

    return f"{horas:02d}:{minutos:02d}:{segundos:02d}"


def imprimir_progresso(
    etapa,
    atual,
    total,
    inicio,
    detalhe=""
):
    """
    Exibe progresso, tempo decorrido e estimativa restante.
    """
    decorrido = time.time() - inicio

    percentual = (atual / total * 100) if total > 0 else 100

    if atual > 0:
        tempo_medio = decorrido / atual
        restante = tempo_medio * (total - atual)
    else:
        restante = 0

    mensagem = (
        f"[{etapa}] "
        f"{atual}/{total} ({percentual:.1f}%) | "
        f"Decorrido: {formatar_tempo(decorrido)} | "
        f"Estimativa restante: {formatar_tempo(restante)}"
    )

    if detalhe:
        mensagem += f" | {detalhe}"

    print(mensagem, flush=True)


def salvar_json_atomico(caminho, dados):
    """
    Salva JSON em arquivo temporário e depois substitui
    o arquivo final, reduzindo o risco de corrupção.
    """
    caminho_temp = caminho + ".tmp"

    with open(caminho_temp, "w", encoding="utf-8") as arquivo:
        json.dump(
            dados,
            arquivo,
            ensure_ascii=False,
            indent=2,
            default=str
        )

    os.replace(caminho_temp, caminho)


def carregar_json(caminho, valor_padrao):
    """
    Carrega JSON se existir e RESUME estiver ativado.
    """
    if RESUME and os.path.exists(caminho):
        with open(caminho, "r", encoding="utf-8") as arquivo:
            return json.load(arquivo)

    return valor_padrao


def gerar_assinatura_experimento(configuracao):
    """
    Gera uma assinatura para identificar a configuração
    do experimento e evitar reutilizar checkpoints incompatíveis.
    """
    conteudo = json.dumps(
        configuracao,
        sort_keys=True,
        default=str
    )

    return hashlib.sha256(conteudo.encode("utf-8")).hexdigest()


def carregar_checkpoint_validado(caminho, assinatura):
    """
    Carrega checkpoint somente se a assinatura corresponder
    à configuração atual do experimento.
    """
    if not RESUME or not os.path.exists(caminho):
        return {}

    with open(caminho, "r", encoding="utf-8") as arquivo:
        checkpoint = json.load(arquivo)

    assinatura_salva = checkpoint.get("assinatura")

    if assinatura_salva != assinatura:
        raise ValueError(
            f"Checkpoint incompatível: {caminho}\n"
            "A configuração atual é diferente da execução anterior.\n"
            "Altere OUTPUT_DIR ou remova os checkpoints antigos "
            "para iniciar um novo experimento."
        )

    return checkpoint.get("resultados", {})


def salvar_checkpoint_validado(caminho, assinatura, resultados):
    """
    Salva resultados junto com a assinatura do experimento.
    """
    checkpoint = {
        "assinatura": assinatura,
        "data_atualizacao": datetime.now().isoformat(),
        "resultados": resultados
    }

    salvar_json_atomico(caminho, checkpoint)


# ============================================================
# 4. CARREGAMENTO DOS DADOS
# ============================================================

print("\n" + "=" * 70)
print("CARREGAMENTO DOS DADOS")
print("=" * 70)

df = pd.read_excel(DATA_PATH)

df = df[
    [TEXT_COLUMN, TARGET_COLUMN]
].dropna(subset=[TARGET_COLUMN]).copy()

# Garante que a coluna textual contenha strings
df[TEXT_COLUMN] = df[TEXT_COLUMN].fillna("").astype(str)

X = df[TEXT_COLUMN]
y = df[TARGET_COLUMN]

print(f"Dataset: {DATA_PATH}")
print(f"Quantidade de registros: {len(df)}")
print(f"Quantidade de classes: {y.nunique()}")

print("\nDistribuição das classes:")
print(y.value_counts())

print("\nExemplos de texto:")
print(X.head(3).to_string(index=False))


# ============================================================
# 5. DIVISÃO TREINO / VALIDAÇÃO / TESTE
# ============================================================

print("\n" + "=" * 70)
print("DIVISÃO DOS DADOS")
print("=" * 70)

# Reserva o teste independente antes de qualquer seleção
X_dev, X_test, y_dev, y_test = train_test_split(
    X,
    y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y
)

# Divide os dados de desenvolvimento em treino e validação
X_train, X_val, y_train, y_val = train_test_split(
    X_dev,
    y_dev,
    test_size=VALIDATION_SIZE,
    random_state=RANDOM_STATE,
    stratify=y_dev
)

print(f"Treino:     {len(X_train)} registros")
print(f"Validação:  {len(X_val)} registros")
print(f"Teste:      {len(X_test)} registros")
print(f"Desenvolvimento total: {len(X_dev)} registros")

print("\nDistribuição das classes no treino:")
print(y_train.value_counts())

print("\nDistribuição das classes na validação:")
print(y_val.value_counts())

print("\nDistribuição das classes no teste:")
print(y_test.value_counts())


# ============================================================
# 6. PIPELINE
# ============================================================

pipeline = Pipeline([
    (
        "tfidf",
        TfidfVectorizer()
    ),
    (
        "selecao",
        SelectKBest(score_func=chi2)
    ),
    (
        "classificador",
        LinearSVC(
            random_state=RANDOM_STATE,
            max_iter=10000
        )
    )
])


# ============================================================
# 7. GRID DE HIPERPARÂMETROS
# ============================================================

param_grid = {
    "tfidf__ngram_range": [
        (1, 1),
        (1, 2),
        (1, 3)
    ],

    "tfidf__min_df": [
        1,
        2
    ],

    "tfidf__max_df": [
        0.9,
        1.0
    ],

    "tfidf__max_features": [
        10000,
        None
    ],

    "tfidf__sublinear_tf": [
        True,
        False
    ],

    "tfidf__use_idf": [
        True
    ],

    "tfidf__norm": [
        "l2"
    ],

    "selecao__k": [
        500,
        1000,
        2000,
        5000,
        "all"
    ],

    "classificador__C": [
        0.1,
        1,
        2
    ]
}

configs = list(ParameterGrid(param_grid))
total_configs = len(configs)

print("\n" + "=" * 70)
print("CONFIGURAÇÃO DO GRID SEARCH")
print("=" * 70)

print(f"Total de combinações: {total_configs}")
print(f"Salvamento periódico: a cada {SAVE_EVERY} configurações")
print(f"Mensagens de progresso: a cada {LOG_EVERY} configurações")


# ============================================================
# 8. ASSINATURA DO EXPERIMENTO
# ============================================================

assinatura_config = {
    "data_path": DATA_PATH,
    "text_column": TEXT_COLUMN,
    "target_column": TARGET_COLUMN,
    "random_state": RANDOM_STATE,
    "test_size": TEST_SIZE,
    "validation_size": VALIDATION_SIZE,
    "param_grid": param_grid,
    "pipeline": "TfidfVectorizer + SelectKBest(chi2) + LinearSVC",
    "n_top_models": N_TOP_MODELS,
    "n_cv_splits": N_CV_SPLITS
}

assinatura = gerar_assinatura_experimento(assinatura_config)


# ============================================================
# 9. GRID SEARCH SEM VALIDAÇÃO CRUZADA
# ============================================================

print("\n" + "=" * 70)
print("INICIANDO GRID SEARCH SEM CROSS-VALIDATION")
print("=" * 70)

checkpoint_grid_path = os.path.join(
    OUTPUT_DIR,
    "checkpoint_grid_search.json"
)

resultados_salvos = carregar_checkpoint_validado(
    checkpoint_grid_path,
    assinatura
)

# JSON guarda chaves como strings
resultados_salvos = {
    int(k): v for k, v in resultados_salvos.items()
}

concluidas = set(resultados_salvos.keys())

print(f"Configurações já concluídas: {len(concluidas)}")
print(f"Configurações restantes: {total_configs - len(concluidas)}")

inicio_grid = time.time()
novas_desde_salvamento = 0

for indice, params in enumerate(configs):

    config_id = indice + 1

    # Ignora configurações que já foram concluídas
    if config_id in concluidas:
        continue

    inicio_config = time.time()

    try:
        modelo = clone(pipeline)
        modelo.set_params(**params)

        # Treinamento nos dados de treino
        modelo.fit(X_train, y_train)

        # Avaliação na validação fixa
        y_pred_val = modelo.predict(X_val)

        f1_macro = f1_score(
            y_val,
            y_pred_val,
            average="macro",
            zero_division=0
        )

        accuracy = accuracy_score(
            y_val,
            y_pred_val
        )

        resultados_salvos[config_id] = {
            "config_id": config_id,
            "f1_macro_validacao": float(f1_macro),
            "accuracy_validacao": float(accuracy),
            "params": params
        }

        concluidas.add(config_id)
        novas_desde_salvamento += 1

        tempo_config = time.time() - inicio_config

        if (
            len(concluidas) % LOG_EVERY == 0
            or len(concluidas) == total_configs
        ):
            imprimir_progresso(
                etapa="GRID SEARCH",
                atual=len(concluidas),
                total=total_configs,
                inicio=inicio_grid,
                detalhe=(
                    f"Último F1-macro={f1_macro:.4f}; "
                    f"última configuração={formatar_tempo(tempo_config)}"
                )
            )

        # Checkpoint periódico
        if novas_desde_salvamento >= SAVE_EVERY:

            salvar_checkpoint_validado(
                checkpoint_grid_path,
                assinatura,
                resultados_salvos
            )

            print(
                f"[CHECKPOINT] Progresso salvo: "
                f"{len(concluidas)}/{total_configs} configurações.",
                flush=True
            )

            novas_desde_salvamento = 0

    except Exception as e:

        # Salva o progresso antes de interromper
        salvar_checkpoint_validado(
            checkpoint_grid_path,
            assinatura,
            resultados_salvos
        )

        print("\n[ERRO NO GRID SEARCH]", flush=True)
        print(f"Configuração: {config_id}/{total_configs}", flush=True)
        print(f"Parâmetros: {params}", flush=True)
        print(f"Erro: {repr(e)}", flush=True)
        print("Checkpoint salvo. Execute novamente após corrigir o erro.", flush=True)

        raise

# Salva o último lote, mesmo que tenha menos de SAVE_EVERY resultados
salvar_checkpoint_validado(
    checkpoint_grid_path,
    assinatura,
    resultados_salvos
)

print("\n[GRID SEARCH] Todas as configurações foram concluídas.")

# Ordenação pelo F1-macro da validação
resultados_grid = sorted(
    resultados_salvos.values(),
    key=lambda x: x["f1_macro_validacao"],
    reverse=True
)

resultados_grid_df = pd.DataFrame([
    {
        "config_id": r["config_id"],
        "f1_macro_validacao": r["f1_macro_validacao"],
        "accuracy_validacao": r["accuracy_validacao"],
        "params": json.dumps(r["params"], default=str)
    }
    for r in resultados_grid
])

resultados_grid_df.to_csv(
    os.path.join(OUTPUT_DIR, "resultados_grid_search.csv"),
    index=False
)

print("\nTop 10 configurações do Grid Search:")
print(
    resultados_grid_df.head(10)[
        [
            "config_id",
            "f1_macro_validacao",
            "accuracy_validacao"
        ]
    ].to_string(index=False)
)


# ============================================================
# 10. SELEÇÃO DOS 5 MELHORES MODELOS
# ============================================================

top5 = resultados_grid[:N_TOP_MODELS]

print("\n" + "=" * 70)
print("TOP 5 CONFIGURAÇÕES SELECIONADAS")
print("=" * 70)

for pos, resultado in enumerate(top5, start=1):

    print(f"\nModelo {pos}")
    print(f"Configuração: {resultado['config_id']}")
    print(
        f"F1-macro validação: "
        f"{resultado['f1_macro_validacao']:.4f}"
    )
    print(
        f"Accuracy validação: "
        f"{resultado['accuracy_validacao']:.4f}"
    )
    print("Parâmetros:")
    print(resultado["params"])

top5_df = pd.DataFrame([
    {
        "posicao": pos,
        "config_id": r["config_id"],
        "f1_macro_validacao": r["f1_macro_validacao"],
        "accuracy_validacao": r["accuracy_validacao"],
        "params": json.dumps(r["params"], default=str)
    }
    for pos, r in enumerate(top5, start=1)
])

top5_df.to_csv(
    os.path.join(OUTPUT_DIR, "top5_modelos.csv"),
    index=False
)


# ============================================================
# 11. VALIDAÇÃO CRUZADA DOS 5 FINALISTAS
# ============================================================

print("\n" + "=" * 70)
print("INICIANDO VALIDAÇÃO CRUZADA DOS 5 FINALISTAS")
print("=" * 70)

checkpoint_cv_path = os.path.join(
    OUTPUT_DIR,
    "checkpoint_cv_top5.json"
)

resultados_cv_salvos = carregar_checkpoint_validado(
    checkpoint_cv_path,
    assinatura
)

resultados_cv_salvos = {
    int(k): v for k, v in resultados_cv_salvos.items()
}

cv = StratifiedKFold(
    n_splits=N_CV_SPLITS,
    shuffle=True,
    random_state=RANDOM_STATE
)

inicio_cv = time.time()

for pos, resultado in enumerate(top5, start=1):

    config_id = resultado["config_id"]

    # Pula modelo que já concluiu a validação cruzada
    if config_id in resultados_cv_salvos:

        print(
            f"[CV] Modelo {pos}/5 já concluído. Pulando.",
            flush=True
        )

        continue

    print(
        f"\n[CV] Avaliando modelo {pos}/5 "
        f"(configuração {config_id})...",
        flush=True
    )

    params = resultado["params"]

    modelo = clone(pipeline)
    modelo.set_params(**params)

    inicio_modelo = time.time()

    try:
        scores = cross_val_score(
            modelo,
            X_train,
            y_train,
            cv=cv,
            scoring="f1_macro",
            n_jobs=-1
        )

        resultados_cv_salvos[config_id] = {
            "config_id": config_id,
            "posicao_top5": pos,
            "f1_macro_validacao_inicial": float(
                resultado["f1_macro_validacao"]
            ),
            "f1_macro_cv_media": float(scores.mean()),
            "f1_macro_cv_std": float(scores.std()),
            "scores_folds": scores.tolist(),
            "params": params
        }

        # Salva imediatamente após cada finalista
        salvar_checkpoint_validado(
            checkpoint_cv_path,
            assinatura,
            resultados_cv_salvos
        )

        print(
            f"[CV] Modelo {pos}/5 concluído | "
            f"F1-macro={scores.mean():.4f} ± {scores.std():.4f} | "
            f"Folds={np.round(scores, 4)} | "
            f"Tempo={formatar_tempo(time.time() - inicio_modelo)}",
            flush=True
        )

    except Exception as e:

        salvar_checkpoint_validado(
            checkpoint_cv_path,
            assinatura,
            resultados_cv_salvos
        )

        print(
            f"\n[ERRO CV] Falha no modelo {pos}/5: {repr(e)}",
            flush=True
        )

        print("Checkpoint da CV salvo.", flush=True)

        raise

# Ordena finalistas pelo F1-macro médio da CV
resultados_cv = sorted(
    resultados_cv_salvos.values(),
    key=lambda x: x["f1_macro_cv_media"],
    reverse=True
)

resultados_cv_df = pd.DataFrame([
    {
        "posicao_top5": r["posicao_top5"],
        "config_id": r["config_id"],
        "f1_macro_validacao_inicial": r["f1_macro_validacao_inicial"],
        "f1_macro_cv_media": r["f1_macro_cv_media"],
        "f1_macro_cv_std": r["f1_macro_cv_std"],
        "scores_folds": json.dumps(r["scores_folds"]),
        "params": json.dumps(r["params"], default=str)
    }
    for r in resultados_cv
])

resultados_cv_df.to_csv(
    os.path.join(OUTPUT_DIR, "resultados_cv_top5.csv"),
    index=False
)

print("\nRanking dos finalistas pela validação cruzada:")
print(
    resultados_cv_df[
        [
            "config_id",
            "f1_macro_cv_media",
            "f1_macro_cv_std"
        ]
    ].to_string(index=False)
)


# ============================================================
# 12. ESCOLHA DO MELHOR MODELO PELA VALIDAÇÃO CRUZADA
# ============================================================

melhor_resultado = resultados_cv[0]
melhores_params = melhor_resultado["params"]

print("\n" + "=" * 70)
print("MELHOR MODELO SELECIONADO PELA VALIDAÇÃO CRUZADA")
print("=" * 70)

print(
    f"Configuração: {melhor_resultado['config_id']}"
)
print(
    f"F1-macro médio CV: "
    f"{melhor_resultado['f1_macro_cv_media']:.4f}"
)
print(
    f"Desvio padrão CV: "
    f"{melhor_resultado['f1_macro_cv_std']:.4f}"
)
print("Parâmetros:")
print(melhores_params)


# ============================================================
# 13. TREINAMENTO FINAL NOS 80% DE DESENVOLVIMENTO
# ============================================================

print("\n" + "=" * 70)
print("TREINAMENTO FINAL")
print("=" * 70)

modelo_final = clone(pipeline)
modelo_final.set_params(**melhores_params)

inicio_treinamento_final = time.time()

modelo_final.fit(X_dev, y_dev)

print(
    "Treinamento final concluído em "
    f"{formatar_tempo(time.time() - inicio_treinamento_final)}"
)


# ============================================================
# 14. AVALIAÇÃO NO TESTE INDEPENDENTE
# ============================================================

print("\n" + "=" * 70)
print("AVALIAÇÃO NO TESTE INDEPENDENTE")
print("=" * 70)

y_pred_test = modelo_final.predict(X_test)

f1_test = f1_score(
    y_test,
    y_pred_test,
    average="macro",
    zero_division=0
)

accuracy_test = accuracy_score(
    y_test,
    y_pred_test
)

kappa_test = cohen_kappa_score(
    y_test,
    y_pred_test
)

relatorio = classification_report(
    y_test,
    y_pred_test,
    zero_division=0
)

matriz_confusao = confusion_matrix(
    y_test,
    y_pred_test
)

print(f"F1-macro: {f1_test:.4f}")
print(f"Accuracy: {accuracy_test:.4f}")
print(f"Cohen's kappa: {kappa_test:.4f}")

print("\nRelatório de classificação:")
print(relatorio)

print("\nMatriz de confusão:")
print(matriz_confusao)


# ============================================================
# 15. SALVAMENTO DO MODELO FINAL
# ============================================================

modelo_path = os.path.join(
    OUTPUT_DIR,
    "modelo_final.joblib"
)

joblib.dump(
    modelo_final,
    modelo_path
)

print(f"\nModelo salvo em: {modelo_path}")


# ============================================================
# 16. SALVAMENTO DAS PREVISÕES DO TESTE
# ============================================================

predicoes_df = pd.DataFrame({
    "texto": X_test.reset_index(drop=True),
    "y_real": y_test.reset_index(drop=True),
    "y_predito": y_pred_test
})

predicoes_path = os.path.join(
    OUTPUT_DIR,
    "predicoes_teste_independente.csv"
)

predicoes_df.to_csv(
    predicoes_path,
    index=False
)

print(f"Previsões salvas em: {predicoes_path}")


# ============================================================
# 17. SALVAMENTO DO RESUMO EM TXT
# ============================================================

resumo_path = os.path.join(
    OUTPUT_DIR,
    "resumo_resultados.txt"
)

with open(resumo_path, "w", encoding="utf-8") as f:

    f.write("RESULTADOS DO EXPERIMENTO\n")
    f.write("=" * 70 + "\n\n")

    f.write("Configuração do experimento:\n")
    f.write(f"Dataset: {DATA_PATH}\n")
    f.write(f"Coluna textual: {TEXT_COLUMN}\n")
    f.write(f"Target: {TARGET_COLUMN}\n")
    f.write(f"Total de combinações: {total_configs}\n")
    f.write(f"Divisão treino: {len(X_train)}\n")
    f.write(f"Divisão validação: {len(X_val)}\n")
    f.write(f"Divisão teste: {len(X_test)}\n\n")

    f.write("Melhor configuração selecionada pela CV:\n")
    f.write(
        json.dumps(
            melhores_params,
            indent=4,
            ensure_ascii=False,
            default=str
        )
    )
    f.write("\n\n")

    f.write("Resultados da validação cruzada:\n")
    f.write(
        f"F1-macro médio: "
        f"{melhor_resultado['f1_macro_cv_media']:.4f}\n"
    )
    f.write(
        f"Desvio padrão: "
        f"{melhor_resultado['f1_macro_cv_std']:.4f}\n"
    )
    f.write(
        f"Scores dos folds: "
        f"{melhor_resultado['scores_folds']}\n\n"
    )

    f.write("Resultados no teste independente:\n")
    f.write(f"F1-macro: {f1_test:.4f}\n")
    f.write(f"Accuracy: {accuracy_test:.4f}\n")
    f.write(f"Cohen's kappa: {kappa_test:.4f}\n\n")

    f.write("Relatório de classificação:\n")
    f.write(relatorio)

    f.write("\nMatriz de confusão:\n")
    f.write(str(matriz_confusao))


print(f"Resumo salvo em: {resumo_path}")


# ============================================================
# 18. FINALIZAÇÃO
# ============================================================

print("\n" + "=" * 70)
print("EXPERIMENTO CONCLUÍDO")
print("=" * 70)

print(f"Diretório de saída: {OUTPUT_DIR}")
print(f"F1-macro no teste independente: {f1_test:.4f}")
print(f"Accuracy no teste independente: {accuracy_test:.4f}")
print(f"Cohen's kappa no teste independente: {kappa_test:.4f}")

print("\nArquivos gerados:")
print("- checkpoint_grid_search.json")
print("- checkpoint_cv_top5.json")
print("- resultados_grid_search.csv")
print("- top5_modelos.csv")
print("- resultados_cv_top5.csv")
print("- modelo_final.joblib")
print("- predicoes_teste_independente.csv")
print("- resumo_resultados.txt")