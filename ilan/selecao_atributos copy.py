# Seleção de Atributos 
# Coloquei aqui os principais métodos de Seleção de Atributos.

import numpy as np
from sklearn.feature_selection import (
    SelectKBest,
    SelectPercentile,
    RFE,
    chi2
)
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel


def aplicar_select_k_best(X, y, k=2000, score_func=chi2):
    """
    Seleção Univariada (SelectKBest).
    Mede a dependência estatística entre cada feature e a classe alvo.
    Ideal para matrizes TF-IDF de texto.
    
    :param k: Número absoluto de melhores atributos a manter.
    :param score_func: Função de pontuação estatística (padrão: chi2 para texto).
    """
    seletor = SelectKBest(score_func=score_func, k=k)
    X_reduzido = seletor.fit_transform(X, y)
    print(f"[SelectKBest] Atributos reduzidos de {X.shape[1]} para {X_reduzido.shape[1]}")
    return seletor, X_reduzido


def aplicar_select_percentile(X, y, percentile=10, score_func=chi2):
    """
    Seleção por Percentual (SelectPercentile).
    Mantém apenas os 'N%' melhores atributos estatísticos.
    Útil quando você altera o tamanho do vocabulário e prefere uma proporção fixa.
    
    :param percentile: Porcentagem de features a manter (ex: 10 = Top 10%).
    """
    seletor = SelectPercentile(score_func=score_func, percentile=percentile)
    X_reduzido = seletor.fit_transform(X, y)
    print(f"[SelectPercentile] Atributos reduzidos de {X.shape[1]} para {X_reduzido.shape[1]}")
    return seletor, X_reduzido


def aplicar_rfe(X, y, estimator, n_features_to_select=2000, step=0.1):
    """
    Eliminação Recursiva de Atributos (RFE).
    Treina o estimador repetidamente e remove recursivamente os atributos com menores pesos.
    
    :param estimator: Modelo base com atributo coef_ ou feature_importances_ (ex: LogisticRegression, LinearSVC).
    :param n_features_to_select: Quantidade final de atributos desejada.
    :param step: Quantidade ou fração de features a remover por iteração (ex: 0.1 remove 10% por rodada).
    """
    seletor = RFE(estimator=estimator, n_features_to_select=n_features_to_select, step=step)
    X_reduzido = seletor.fit_transform(X, y)
    print(f"[RFE] Atributos reduzidos de {X.shape[1]} para {X_reduzido.shape[1]}")
    return seletor, X_reduzido


def aplicar_truncated_svd(X, n_components=300, random_state=42):
    """
    Análise de Redução Dimensional (TruncatedSVD / 'LSA').
    Versão do PCA adaptada para matrizes esparsas de texto (onde o PCA tradicional falharia por memória).
    Não remove colunas existentes, mas cria novas dimensões combinadas (conceitos semânticos).
    
    :param n_components: Número de componentes/dimensões finais desejado.
    """
    seletor = TruncatedSVD(n_components=n_components, random_state=random_state)
    X_reduzido = seletor.fit_transform(X)
    print(f"[TruncatedSVD] Dimensionalidade reduzida de {X.shape[1]} para {X_reduzido.shape[1]}")
    return seletor, X_reduzido


def aplicar_importancia_arvores(X, y, n_estimators=100, threshold="mean", random_state=42):
    """
    Seleção por Importância de Árvores (ExtraTreesClassifier).
    Usa um conjunto de árvores de decisão para calcular a importância de cada atributo e filtra.
    
    :param threshold: Limite para corte. 'mean' (acima da média), 'median' ou valor numérico.
    """
    modelo_arvore = ExtraTreesClassifier(n_estimators=n_estimators, random_state=random_state, n_jobs=-1)
    modelo_arvore.fit(X, y)
    
    seletor = SelectFromModel(estimator=modelo_arvore, threshold=threshold, prefit=True)
    X_reduzido = seletor.transform(X)
    print(f"[ExtraTrees] Atributos reduzidos de {X.shape[1]} para {X_reduzido.shape[1]}")
    return seletor, X_reduzido