# Funções de avaliação (Acurácia, F1-Score, ROC-AUC)
# Fiz levando em conta que a classificação é ternaria


import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    log_loss,
    roc_auc_score,
)
from sklearn.model_selection import learning_curve


def calcular_todas_metricas(modelo, X_test, y_test):
    """Calcula e exibe no terminal as 5 principais métricas agregadas para o problema:

    - F1-Score (Macro)
    - Acurácia
    - Cohen's Kappa
    - ROC-AUC (One-vs-Rest)
    - Log Loss
    """
    # Predições de classe simples
    preds = modelo.predict(X_test)

    # Coleta de probabilidades (necessário para o ROC-AUC e Log Loss)
    try:
        probas = modelo.predict_proba(X_test)
    except AttributeError:
        # Caso o modelo use um estimador sem predict_proba (ex: LinearSVC puro)
        probas = modelo.decision_function(X_test)
        # Normalização simples em softmax para aproximação de probabilidade se necessário
        exp_p = np.exp(probas - np.max(probas, axis=1, keepdims=True))
        probas = exp_p / np.sum(exp_p, axis=1, keepdims=True)


    f1_macro = f1_score(y_test, preds, average="macro")
    acc = accuracy_score(y_test, preds)
    kappa = cohen_kappa_score(y_test, preds)
    auc_score = roc_auc_score(y_test, probas, multi_class="ovr")
    loss_val = log_loss(y_test, probas)

    print("=" * 45)
    print("      RELATÓRIO DE MÉTRICAS DO MODELO       ")
    print("=" * 45)
    print(f"1. F1-Score (Macro):  {f1_macro:.4f}")
    print(f"2. Acurácia:          {acc:.4f}")
    print(f"3. Cohen's Kappa:     {kappa:.4f}")
    print(f"4. ROC-AUC (OvR):     {auc_score:.4f}")
    print(f"5. Log Loss:          {loss_val:.4f}")
    print("=" * 45)

    return {
        "f1_macro": f1_macro,
        "acuracia": acc,
        "kappa": kappa,
        "roc_auc": auc_score,
        "log_loss": loss_val,
    }


def plotar_matriz_confusao(
    modelo, X_test, y_test, labels=['c1', 'c234','c5'], title="Matriz de Confusão"
):
    """Gera um gráfico visual estilizado da Matriz de Confusão.
    """
    preds = modelo.predict(X_test)
    cm = confusion_matrix(y_test, preds)

    plt.figure(figsize=(7, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels if labels else "auto",
        yticklabels=labels if labels else "auto",
    )
    plt.title(title, fontsize=12, fontweight="bold")
    plt.xlabel("Classe Preditada", fontsize=10)
    plt.ylabel("Classe Real", fontsize=10)
    plt.tight_layout()
    plt.show()


def plotar_curva_aprendizado(
    estimator, X, y, cv=5, scoring="f1_macro", n_jobs=-1
):
    """Calcula e plota a Curva de Aprendizado para diagnosticar Overfitting vs
    Underfitting ao longo do tamanho do dataset de treino.
    """
    print("Calculando Curva de Aprendizado (pode levar alguns segundos)...")

    train_sizes, train_scores, val_scores = learning_curve(
        estimator=estimator,
        X=X,
        y=y,
        cv=cv,
        scoring=scoring,
        n_jobs=n_jobs,
        train_sizes=np.linspace(0.1, 1.0, 5),
        random_state=42,
    )

    # Médias e desvios padrões
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    val_mean = np.mean(val_scores, axis=1)
    val_std = np.std(val_scores, axis=1)

    plt.figure(figsize=(8, 5))

    # Curva do Treino
    plt.plot(
        train_sizes,
        train_mean,
        "o-",
        color="#1f77b4",
        label="Score de Treino",
    )
    plt.fill_between(
        train_sizes,
        train_mean - train_std,
        train_mean + train_std,
        alpha=0.15,
        color="#1f77b4",
    )

    # Curva da Validação
    plt.plot(
        train_sizes,
        val_mean,
        "o-",
        color="#ff7f0e",
        label="Score de Validação (CV)",
    )
    plt.fill_between(
        train_sizes,
        val_mean - val_std,
        val_mean + val_std,
        alpha=0.15,
        color="#ff7f0e",
    )

    plt.title(
        f"Curva de Aprendizado ({scoring.upper()})",
        fontsize=12,
        fontweight="bold",
    )
    plt.xlabel("Quantidade de Amostras de Treino", fontsize=10)
    plt.ylabel(f"Métrica: {scoring}", fontsize=10)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.show()