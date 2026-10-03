# Modelos clássicos (TF-IDF) usados nos experimentos e no ensemble clássico.
# Ficam em src/ para que modelos salvos com joblib possam ser carregados de qualquer script.

import numpy as np
from scipy.sparse import hstack
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin, clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import ComplementNB, MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.svm import LinearSVC


class PalavraCaractere(BaseEstimator, TransformerMixin):
    """Concatena TF-IDF de palavras (1-2) e de caracteres (2-5, dentro das palavras)."""

    def fit(self, X, y=None):
        self.palavras_ = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True).fit(X)
        self.caracteres_ = TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 5), sublinear_tf=True,
                                           max_features=300000).fit(X)
        return self

    def transform(self, X):
        return hstack([self.palavras_.transform(X), self.caracteres_.transform(X)]).tocsr()


def svc_word12():
    # Melhor configuração do grid noturno (word 1-2 + LinearSVC C=0.1), calibrada para dar probabilidades
    return make_pipeline(TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True),
                         CalibratedClassifierCV(LinearSVC(C=0.1), method='sigmoid', cv=5))


def svc_word12_char25():
    return make_pipeline(PalavraCaractere(), CalibratedClassifierCV(LinearSVC(C=0.05), method='sigmoid', cv=5))


def logreg_word12():
    return make_pipeline(TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True), LogisticRegression(C=2.0, max_iter=3000))


def complementnb_word12():
    return make_pipeline(TfidfVectorizer(ngram_range=(1, 2)), ComplementNB(alpha=0.5))


def multinomialnb_word12():
    return make_pipeline(TfidfVectorizer(ngram_range=(1, 2)), MultinomialNB(alpha=0.1))


FABRICAS = {
    'svc_word12': svc_word12,
    'svc_word12_char25': svc_word12_char25,
    'logreg_word12': logreg_word12,
    'complementnb_word12': complementnb_word12,
    'multinomialnb_word12': multinomialnb_word12,
}


class EnsembleMedia(BaseEstimator, ClassifierMixin):
    """Média (opcionalmente ponderada) das probabilidades de vários classificadores."""

    def __init__(self, modelos, pesos=None):
        self.modelos = modelos
        self.pesos = pesos

    def fit(self, X, y):
        self.modelos_ = [clone(m).fit(X, y) for m in self.modelos]
        self.classes_ = self.modelos_[0].classes_
        for m in self.modelos_:
            assert list(m.classes_) == list(self.classes_), 'classes em ordem diferente entre os modelos'
        return self

    def predict_proba(self, X):
        pesos = self.pesos if self.pesos is not None else [1 / len(self.modelos_)] * len(self.modelos_)
        return sum(w * m.predict_proba(X) for w, m in zip(pesos, self.modelos_))

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]


def ensemble_svc_cnb():
    """Ensemble clássico: SVC palavra+caractere + ComplementNB, média simples."""
    return EnsembleMedia([svc_word12_char25(), complementnb_word12()])
