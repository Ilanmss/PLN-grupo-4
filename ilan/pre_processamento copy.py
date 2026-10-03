# Pré-processamento
# Coloquei aqui os principais métodos de pré-processamento (limpeza de texto) (lowercase, stopwords, regex e etc)
# Lembrando que o professor recomendou não cortar dados na maioria das vezes.

import re
import unicodedata
import nltk
from nltk.corpus import stopwords
from nltk.stem.rslp import RSLPStemmer
import spacy

# Baixa as stopwords do NLTK se ainda não estiverem baixadas
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

# Tenta carregar o modelo em português do spaCy para lematização
try:
    nlp = spacy.load("pt_core_news_sm", disable=["parser", "ner"])
except OSError:
    nlp = None

# Carrega stopwords em português
STOPWORDS_PT = set(stopwords.words('portuguese'))



def to_lowercase(text: str) -> str:
    """Converte todo o texto para minúsculas."""
    return text.lower()


def remove_accents(text: str) -> str:
    """Remove acentos e caracteres diacríticos (ex: 'á' -> 'a', 'ç' -> 'c')."""
    nfkd_form = unicodedata.normalize('NFKD', text)
    return "".join([c for c in nfkd_form if not unicodedata.combining(c)])



def remove_numbers(text: str) -> str:
    """Remove dígitos e números do texto."""
    return re.sub(r'\d+', '', text)


def remove_punctuation(text: str) -> str:
    """Remove pontuações e símbolos especiais, mantendo apenas letras e espaços."""
    return re.sub(r'[^\w\s]', '', text)


def remove_extra_spaces(text: str) -> str:
    """Remove espaços duplos, quebras de linha e espaços no início/fim."""
    return re.sub(r'\s+', ' ', text).strip()


def remove_urls_and_emails(text: str) -> str:
    """Remove links (http/https/www) e endereços de e-mail do texto."""
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'\S+@\S+', '', text)
    return text


def remove_stopwords(text: str, custom_stopwords: set = None) -> str:
    """
    Remove palavras frequentes que não agregam valor semântico (de, para, com, etc.).
    Permite passar uma lista customizada de stopwords específicas.
    """
    words_to_remove = STOPWORDS_PT.union(custom_stopwords) if custom_stopwords else STOPWORDS_PT
    words = text.split()
    filtered_words = [word for word in words if word.lower() not in words_to_remove]
    return " ".join(filtered_words)


def apply_stemming(text: str) -> str:
    """
    Aplica o algoritmo RSLP (Deltas do Português) para reduzir as palavras ao seu radical.
    Exemplo: 'explicou', 'explicação', 'explicando' -> 'explic'
    """
    stemmer = RSLPStemmer()
    words = text.split()
    stemmed_words = [stemmer.stem(word) for word in words]
    return " ".join(stemmed_words)


def apply_lemmatization(text: str) -> str:
    """
    Reduz as palavras à sua forma canônica/dicionário usando spaCy.
    Exemplo: 'fizeram' -> 'fazer', 'melhores' -> 'bom'
    Nota: É mais lento que o stemming, mas preserva a gramática correta.
    """
    if nlp is None:
        raise RuntimeError("O modelo 'pt_core_news_sm' do spaCy não está instalado.")
    
    doc = nlp(text)
    lemmas = [token.lemma_ for token in doc]
    return " ".join(lemmas)



def preprocess_text(
    text: str,
    lowercase: bool = False,
    clean_urls: bool = False,
    clean_accents: bool = False,
    clean_numbers: bool = False,
    clean_punctuation: bool = False,
    clean_stopwords: bool = False,
    use_stemming: bool = False,
    use_lemmatization: bool = False,
    remove_extra_spaces: bool = False,
    custom_stopwords: set = None
) -> str:
    """
    Executa o pipeline completo de pré-processamento de acordo com os parâmetros configurados.
    """
    if not isinstance(text, str):
        text = str(text)

    if clean_urls:
        text = remove_urls_and_emails(text)
    if lowercase:
        text = to_lowercase(text)
    if clean_accents:
        text = remove_accents(text)
    if clean_numbers:
        text = remove_numbers(text)
    if clean_punctuation:
        text = remove_punctuation(text)
    if clean_stopwords:
        text = remove_stopwords(text, custom_stopwords=custom_stopwords)
    if use_stemming:
        text = apply_stemming(text)
    elif use_lemmatization:
        text = apply_lemmatization(text)
    elif remove_extra_spaces:
      text = remove_extra_spaces(text)

    return text