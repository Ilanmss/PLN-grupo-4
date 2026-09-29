
# ============================================================
# HOLDOUT + NESTED CROSS-VALIDATION SEM DATA LEAKAGE
#
# Modelos:
#   1. Multinomial Naive Bayes
#   2. Complement Naive Bayes
#
# Separação inicial:
#   80% Treino
#   20% Teste independente
#
# Validação externa:
#   5 folds x 10 repetições (somente treino)
#
# Validação interna:
#   5 folds para GridSearchCV
#
# Métrica principal:
#   F1-macro
#
# IMPORTANTE:
#   O conjunto de teste não participa de nenhuma etapa
#   de treinamento, seleção de hiperparâmetros ou modelo.
# ============================================================


# ============================================================
# 1. IMPORTAÇÕES
# ============================================================

import os
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB, ComplementNB

from sklearn.model_selection import (
    train_test_split,
    RepeatedStratifiedKFold,
    StratifiedKFold,
    GridSearchCV
)

from sklearn.metrics import (
    f1_score,
    accuracy_score,
    cohen_kappa_score,
    classification_report,
    confusion_matrix
)


# ============================================================
# 2. CONFIGURAÇÕES
# ============================================================

CAMINHO_ARQUIVO = "data/train.xlsx"

COLUNA_TEXTO = "resp_text"
COLUNA_ALVO = "clarity"

PASTA_SAIDA = "modelos/naive_bayes"

# Holdout
TEST_SIZE = 0.20

# Nested Cross-Validation
N_SPLITS_OUTER = 5
N_REPEATS_OUTER = 10
N_SPLITS_INNER = 5

RANDOM_STATE = 42

os.makedirs(PASTA_SAIDA, exist_ok=True)


# ============================================================
# 3. CARREGAMENTO E PREPARAÇÃO DOS DADOS
# ============================================================

df = pd.read_excel(CAMINHO_ARQUIVO)

# Remove registros sem rótulo.
df = df.dropna(subset=[COLUNA_ALVO]).copy()

# Trata valores ausentes no texto.
X = df[COLUNA_TEXTO].fillna("").astype(str)

# Alvo
y = df[COLUNA_ALVO]

print("=" * 70)
print("INFORMAÇÕES DO DATASET")
print("=" * 70)

print(f"Total de registros: {len(df)}")
print(f"Quantidade de classes: {y.nunique()}")

print("\nDistribuição das classes:")
print(y.value_counts())


# ============================================================
# 4. SEPARAÇÃO INICIAL: TREINO E TESTE
# ============================================================

# O teste é separado antes de qualquer treinamento.
# A estratificação preserva aproximadamente a proporção
# das classes em ambos os conjuntos.

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=TEST_SIZE,
    stratify=y,
    random_state=RANDOM_STATE
)

print("\n" + "=" * 70)
print("SEPARAÇÃO HOLDOUT")
print("=" * 70)

print(f"Registros de treino: {len(X_train)}")
print(f"Registros de teste: {len(X_test)}")

print("\nDistribuição das classes - Treino:")
print(y_train.value_counts())

print("\nDistribuição das classes - Teste:")
print(y_test.value_counts())


# Salva a distribuição dos conjuntos para documentação.
distribuicao = pd.DataFrame({
    "treino": y_train.value_counts(),
    "teste": y_test.value_counts()
}).fillna(0).astype(int)

distribuicao.to_csv(
    os.path.join(PASTA_SAIDA, "distribuicao_holdout.csv"),
    encoding="utf-8-sig"
)


# ============================================================
# 5. VERIFICAÇÃO DAS CLASSES
# ============================================================

# Verifica se existem exemplos suficientes para a validação
# estratificada externa e interna.

menor_classe_treino = y_train.value_counts().min()

if menor_classe_treino < N_SPLITS_OUTER:
    raise ValueError(
        f"A menor classe no treino possui apenas "
        f"{menor_classe_treino} exemplos. "
        f"Não é possível utilizar {N_SPLITS_OUTER} folds externos."
    )

# Verifica se o conjunto de teste possui todas as classes.
classes_treino = set(y_train.unique())
classes_teste = set(y_test.unique())

if classes_treino != classes_teste:
    raise ValueError(
        "O conjunto de teste não contém todas as classes presentes "
        "no conjunto de treino. Revise a distribuição dos dados "
        "ou a estratégia de divisão."
    )


# ============================================================
# 6. MODELOS E HIPERPARÂMETROS
# ============================================================

# O TF-IDF permanece dentro do Pipeline.
#
# Dessa forma, em cada fold:
# - O vocabulário é aprendido somente no treinamento.
# - Os pesos IDF são calculados somente no treinamento.
# - A transformação dos dados de validação utiliza o
#   vetorizador ajustado no respectivo treinamento.
#
# Isso evita vazamento de informações do texto entre folds.

modelos = {

    "MultinomialNB": {
        "pipeline": Pipeline([
            ("tfidf", TfidfVectorizer()),
            ("modelo", MultinomialNB())
        ]),

        "param_grid": {
            "tfidf__ngram_range": [
                (1, 1),
                (1, 2)
            ],

            "tfidf__min_df": [
                1,
                2
            ],

            "modelo__alpha": [
                0.1,
                0.5,
                1.0,
                2.0
            ]
        }
    },

    "ComplementNB": {
        "pipeline": Pipeline([
            ("tfidf", TfidfVectorizer()),
            ("modelo", ComplementNB())
        ]),

        "param_grid": {
            "tfidf__ngram_range": [
                (1, 1),
                (1, 2)
            ],

            "tfidf__min_df": [
                1,
                2
            ],

            "modelo__alpha": [
                0.1,
                0.5,
                1.0,
                2.0
            ]
        }
    }
}


# ============================================================
# 7. NESTED CROSS-VALIDATION NO CONJUNTO DE TREINO
# ============================================================

# A validação externa utiliza exclusivamente X_train e y_train.
#
# O conjunto X_test e y_test permanece intocado nesta etapa.
#
# Para cada fold externo:
#   1. Separa treino e validação externa.
#   2. Executa GridSearchCV somente no treino externo.
#   3. Ajusta o melhor pipeline no treino externo completo.
#   4. Avalia na validação externa.
#
# Total: 5 folds x 10 repetições = 50 avaliações por modelo.

cv_externa = RepeatedStratifiedKFold(
    n_splits=N_SPLITS_OUTER,
    n_repeats=N_REPEATS_OUTER,
    random_state=RANDOM_STATE
)

resultados_nested = []

for nome_modelo, configuracao in modelos.items():

    print("\n" + "=" * 70)
    print(f"NESTED CROSS-VALIDATION: {nome_modelo}")
    print("=" * 70)

    pipeline = configuracao["pipeline"]
    param_grid = configuracao["param_grid"]

    for numero_fold, (indices_treino, indices_validacao) in enumerate(
        cv_externa.split(X_train, y_train),
        start=1
    ):

        # Separação do fold externo.
        X_treino_fold = X_train.iloc[indices_treino]
        X_validacao_fold = X_train.iloc[indices_validacao]

        y_treino_fold = y_train.iloc[indices_treino]
        y_validacao_fold = y_train.iloc[indices_validacao]

        # Validação interna.
        cv_interna = StratifiedKFold(
            n_splits=N_SPLITS_INNER,
            shuffle=True,
            random_state=RANDOM_STATE
        )

        # Grid Search recebe exclusivamente o treinamento externo.
        busca = GridSearchCV(
            estimator=pipeline,
            param_grid=param_grid,
            scoring="f1_macro",
            cv=cv_interna,
            n_jobs=-1,
            refit=True,
            error_score="raise"
        )

        busca.fit(
            X_treino_fold,
            y_treino_fold
        )

        # O melhor pipeline já foi reajustado em todo o
        # treinamento externo pelo refit=True.
        melhor_modelo = busca.best_estimator_

        # Avaliação exclusivamente na validação externa.
        y_pred = melhor_modelo.predict(X_validacao_fold)

        f1 = f1_score(
            y_validacao_fold,
            y_pred,
            average="macro",
            zero_division=0
        )

        acc = accuracy_score(
            y_validacao_fold,
            y_pred
        )

        kappa = cohen_kappa_score(
            y_validacao_fold,
            y_pred
        )

        resultados_nested.append({
            "modelo": nome_modelo,
            "fold": numero_fold,
            "repeticao": (numero_fold - 1) // N_SPLITS_OUTER + 1,
            "f1_macro": f1,
            "accuracy": acc,
            "cohen_kappa": kappa,
            "melhores_parametros": json.dumps(
                busca.best_params_,
                ensure_ascii=False,
                default=str
            )
        })

        print(
            f"Fold {numero_fold:02d}/"
            f"{N_SPLITS_OUTER * N_REPEATS_OUTER} | "
            f"F1-macro: {f1:.4f} | "
            f"Accuracy: {acc:.4f} | "
            f"Kappa: {kappa:.4f}"
        )


# ============================================================
# 8. SALVAR RESULTADOS DA NESTED CV
# ============================================================

df_nested = pd.DataFrame(resultados_nested)

caminho_nested = os.path.join(
    PASTA_SAIDA,
    "resultados_nested_cv.csv"
)

df_nested.to_csv(
    caminho_nested,
    index=False,
    encoding="utf-8-sig"
)

print(f"\nResultados da Nested CV salvos em: {caminho_nested}")


# ============================================================
# 9. RESUMO DA NESTED CROSS-VALIDATION
# ============================================================

resumo_nested = (
    df_nested
    .groupby("modelo")
    .agg(
        f1_macro_media=("f1_macro", "mean"),
        f1_macro_desvio=("f1_macro", "std"),

        accuracy_media=("accuracy", "mean"),
        accuracy_desvio=("accuracy", "std"),

        kappa_medio=("cohen_kappa", "mean"),
        kappa_desvio=("cohen_kappa", "std")
    )
    .reset_index()
)

caminho_resumo_nested = os.path.join(
    PASTA_SAIDA,
    "resumo_nested_cv.csv"
)

resumo_nested.to_csv(
    caminho_resumo_nested,
    index=False,
    encoding="utf-8-sig"
)

print("\n" + "=" * 70)
print("RESUMO DA NESTED CROSS-VALIDATION")
print("=" * 70)

print(resumo_nested.to_string(index=False))


# ============================================================
# 10. TREINAMENTO FINAL NO CONJUNTO DE TREINO
# ============================================================

# Agora cada modelo é otimizado utilizando todo o conjunto
# de treinamento (80%).
#
# O conjunto de teste continua completamente separado.
#
# O best_score_ desta busca é uma métrica de validação interna,
# não uma estimativa independente do desempenho final.

cv_final = StratifiedKFold(
    n_splits=N_SPLITS_INNER,
    shuffle=True,
    random_state=RANDOM_STATE
)

resultados_teste = []
relatorios = {}

for nome_modelo, configuracao in modelos.items():

    print("\n" + "=" * 70)
    print(f"TREINAMENTO FINAL: {nome_modelo}")
    print("=" * 70)

    busca_final = GridSearchCV(
        estimator=configuracao["pipeline"],
        param_grid=configuracao["param_grid"],
        scoring="f1_macro",
        cv=cv_final,
        n_jobs=-1,
        refit=True,
        error_score="raise"
    )

    # Treina somente nos dados de treino.
    busca_final.fit(
        X_train,
        y_train
    )

    modelo_final = busca_final.best_estimator_

    print(f"Melhores parâmetros: {busca_final.best_params_}")
    print(f"F1-macro médio na CV interna: {busca_final.best_score_:.4f}")

    # ========================================================
    # 11. AVALIAÇÃO FINAL NO TESTE INDEPENDENTE
    # ========================================================

    # O teste é utilizado exclusivamente para previsão e avaliação.
    y_pred_teste = modelo_final.predict(X_test)

    f1_teste = f1_score(
        y_test,
        y_pred_teste,
        average="macro",
        zero_division=0
    )

    accuracy_teste = accuracy_score(
        y_test,
        y_pred_teste
    )

    kappa_teste = cohen_kappa_score(
        y_test,
        y_pred_teste
    )

    print("\nRESULTADOS NO TESTE INDEPENDENTE")
    print(f"F1-macro: {f1_teste:.4f}")
    print(f"Accuracy: {accuracy_teste:.4f}")
    print(f"Cohen's Kappa: {kappa_teste:.4f}")

    # Relatório por classe.
    relatorio = classification_report(
        y_test,
        y_pred_teste,
        zero_division=0,
        output_dict=True
    )

    relatorios[nome_modelo] = relatorio

    print("\nRelatório de classificação:")
    print(
        classification_report(
            y_test,
            y_pred_teste,
            zero_division=0
        )
    )

    # Matriz de confusão.
    classes = sorted(y_train.unique())

    matriz = confusion_matrix(
        y_test,
        y_pred_teste,
        labels=classes
    )

    df_matriz = pd.DataFrame(
        matriz,
        index=classes,
        columns=classes
    )

    caminho_matriz = os.path.join(
        PASTA_SAIDA,
        f"matriz_confusao_{nome_modelo}.csv"
    )

    df_matriz.to_csv(
        caminho_matriz,
        encoding="utf-8-sig"
    )

    # Salva as previsões individuais do teste.
    df_predicoes = pd.DataFrame({
        "y_real": y_test.to_numpy(),
        "y_predito": y_pred_teste
    })

    caminho_predicoes = os.path.join(
        PASTA_SAIDA,
        f"predicoes_teste_{nome_modelo}.csv"
    )

    df_predicoes.to_csv(
        caminho_predicoes,
        index=False,
        encoding="utf-8-sig"
    )

    # Armazena métricas finais.
    resultados_teste.append({
        "modelo": nome_modelo,
        "f1_macro_teste": f1_teste,
        "accuracy_teste": accuracy_teste,
        "cohen_kappa_teste": kappa_teste,
        "best_score_cv_treino": busca_final.best_score_,
        "melhores_parametros": json.dumps(
            busca_final.best_params_,
            ensure_ascii=False,
            default=str
        )
    })

    # ========================================================
    # 12. SALVAR MODELO FINAL
    # ========================================================

    caminho_modelo = os.path.join(
        PASTA_SAIDA,
        f"{nome_modelo}_final.joblib"
    )

    joblib.dump(
        {
            "pipeline": modelo_final,
            "vetorizador": modelo_final.named_steps["tfidf"],
            "modelo": modelo_final.named_steps["modelo"],
            "melhores_parametros": busca_final.best_params_,
            "best_score_cv": busca_final.best_score_,
            "coluna_texto": COLUNA_TEXTO,
            "coluna_alvo": COLUNA_ALVO,
            "random_state": RANDOM_STATE,
            "test_size": TEST_SIZE
        },
        caminho_modelo
    )

    print(f"\nModelo salvo em: {caminho_modelo}")


# ============================================================
# 13. SALVAR RESULTADOS DO TESTE
# ============================================================

df_teste = pd.DataFrame(resultados_teste)

caminho_teste = os.path.join(
    PASTA_SAIDA,
    "resultados_teste_independente.csv"
)

df_teste.to_csv(
    caminho_teste,
    index=False,
    encoding="utf-8-sig"
)

print("\n" + "=" * 70)
print("RESULTADOS FINAIS NO TESTE INDEPENDENTE")
print("=" * 70)

print(df_teste.to_string(index=False))


# ============================================================
# 14. COMPARAÇÃO NESTED CV X TESTE
# ============================================================

comparacao = resumo_nested.merge(
    df_teste,
    on="modelo",
    how="inner"
)

caminho_comparacao = os.path.join(
    PASTA_SAIDA,
    "comparacao_nested_cv_teste.csv"
)

comparacao.to_csv(
    caminho_comparacao,
    index=False,
    encoding="utf-8-sig"
)

print("\n" + "=" * 70)
print("COMPARAÇÃO: NESTED CV E TESTE INDEPENDENTE")
print("=" * 70)

print(comparacao.to_string(index=False))


# ============================================================
# 15. SALVAR RELATÓRIO TXT
# ============================================================

caminho_txt = os.path.join(
    PASTA_SAIDA,
    "relatorio_experimento.txt"
)

with open(caminho_txt, "w", encoding="utf-8") as arquivo:

    arquivo.write("EXPERIMENTO: HOLDOUT + NESTED CROSS-VALIDATION\n")
    arquivo.write("=" * 70 + "\n\n")

    arquivo.write(f"Dataset: {CAMINHO_ARQUIVO}\n")
    arquivo.write(f"Coluna de texto: {COLUNA_TEXTO}\n")
    arquivo.write(f"Coluna alvo: {COLUNA_ALVO}\n\n")

    arquivo.write(f"Total de registros: {len(df)}\n")
    arquivo.write(f"Treino: {len(X_train)} ({1 - TEST_SIZE:.0%})\n")
    arquivo.write(f"Teste: {len(X_test)} ({TEST_SIZE:.0%})\n\n")

    arquivo.write(
        f"Validação externa: {N_SPLITS_OUTER} folds x "
        f"{N_REPEATS_OUTER} repetições\n"
    )

    arquivo.write(
        f"Validação interna: {N_SPLITS_INNER} folds\n"
    )

    arquivo.write("Métrica principal: F1-macro\n")
    arquivo.write("Separação: estratificada\n")
    arquivo.write("Random State: 42\n\n")

    arquivo.write("RESULTADOS DA NESTED CROSS-VALIDATION\n")
    arquivo.write("-" * 70 + "\n")

    for _, linha in resumo_nested.iterrows():

        arquivo.write(f"\nModelo: {linha['modelo']}\n")

        arquivo.write(
            f"F1-macro: {linha['f1_macro_media']:.4f} "
            f"+/- {linha['f1_macro_desvio']:.4f}\n"
        )

        arquivo.write(
            f"Accuracy: {linha['accuracy_media']:.4f} "
            f"+/- {linha['accuracy_desvio']:.4f}\n"
        )

        arquivo.write(
            f"Cohen's Kappa: {linha['kappa_medio']:.4f} "
            f"+/- {linha['kappa_desvio']:.4f}\n"
        )

    arquivo.write("\n\nRESULTADOS NO TESTE INDEPENDENTE\n")
    arquivo.write("-" * 70 + "\n")

    for _, linha in df_teste.iterrows():

        arquivo.write(f"\nModelo: {linha['modelo']}\n")

        arquivo.write(
            f"F1-macro: {linha['f1_macro_teste']:.4f}\n"
        )

        arquivo.write(
            f"Accuracy: {linha['accuracy_teste']:.4f}\n"
        )

        arquivo.write(
            f"Cohen's Kappa: {linha['cohen_kappa_teste']:.4f}\n"
        )

        arquivo.write(
            f"F1-macro CV interna: "
            f"{linha['best_score_cv_treino']:.4f}\n"
        )

        arquivo.write(
            f"Melhores parâmetros: {linha['melhores_parametros']}\n"
        )

    arquivo.write("\n\nOBSERVAÇÃO METODOLÓGICA\n")
    arquivo.write("-" * 70 + "\n")

    arquivo.write(
        "O conjunto de teste foi separado antes do treinamento e "
        "não participou da seleção de hiperparâmetros, validação "
        "cruzada ou ajuste dos modelos.\n"
    )

    arquivo.write(
        "A Nested Cross-Validation foi realizada exclusivamente "
        "no conjunto de treinamento.\n"
    )

    arquivo.write(
        "Os resultados do teste representam a avaliação final "
        "dos modelos treinados com o conjunto de treinamento.\n"
    )


# ============================================================
# 16. FINALIZAÇÃO
# ============================================================

print("\n" + "=" * 70)
print("EXPERIMENTO CONCLUÍDO")
print("=" * 70)

print(f"Todos os resultados foram salvos em: {PASTA_SAIDA}")

print("\nArquivos principais:")
print("- resultados_nested_cv.csv")
print("- resumo_nested_cv.csv")
print("- resultados_teste_independente.csv")
print("- comparacao_nested_cv_teste.csv")
print("- relatorio_experimento.txt")
print("- MultinomialNB_final.joblib")
print("- ComplementNB_final.joblib")