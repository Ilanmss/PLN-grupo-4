# Vetorização
# Coloquei aqui os principais métodos de Vetorização que a gente viu até agora: tipo Bag of Words e TF-IDF

from typing import List, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import joblib


def get_bag_of_words_vectorizer(
    max_features: int = 5000,
    ngram_range: Tuple[int, int] = (1, 1),
    binary: bool = False
) -> CountVectorizer:
    """
    Retorna uma Bag of Words.
    - binary=False: conta a frequência bruta (ex: 1, 2, 3...).
    - binary=True: apenas indica presença ou ausência (0 ou 1).
    """
    return CountVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        binary=binary
    )


def get_tfidf_vectorizer(
    max_features: int = 5000,
    ngram_range: Tuple[int, int] = (1, 2),
    min_df: Union[int, float] = 2,
    max_df: Union[int, float] = 0.95,
    sublinear_tf: bool = True
) -> TfidfVectorizer:
    """
    Retorna um vetorizador TF-IDF (Frequência do Termo x Frequência Inversa nos Documentos)
    - min_df=2: ignora palavras que aparecem em menos de 2 documentos (remove erros de digitação/ruído)
        - Passar min_df=1 para não remover os ruídos
    - max_df=0.95: ignora palavras que aparecem em mais de 95% dos documentos (tipo stopwords implícitas)
        - passar max_df = 1 para não desconsiderar as stopword implícitas
    - sublinear_tf=True: aplica escala logarítmica (1 + log(tf)) para suavizar contagens altas (não sei oque siginifca direito mas tava na documentação do método)
    """
    return TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        min_df=min_df,
        max_df=max_df,
        sublinear_tf=sublinear_tf
    )



def save_vectorizer(vectorizer, filepath: str) -> None:
    """Salva o vetorizador ajustado (fit) em um arquivo .joblib."""
    joblib.dump(vectorizer, filepath)
    print(f"Vetorizador salvo em: {filepath}")


def load_vectorizer(filepath: str):
    """Carrega um vetorizador treinado a partir de um arquivo .joblib."""
    return joblib.load(filepath)


