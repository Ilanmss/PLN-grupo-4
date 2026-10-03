import os
import re
import pandas as pd
import numpy as np

from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, accuracy_score, f1_score
from sklearn.model_selection import train_test_split


# ==========================================
# CONFIGURAÇÕES DE CAMINHO E COLUNAS
# ==========================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PATH_INPUT = os.path.abspath(os.path.join(BASE_DIR, "..", "data", "train_cleaned.xlsx"))
PATH_OUTPUT = os.path.abspath(os.path.join(BASE_DIR, "..", "data", "train_cleaned_no_boilerplate.xlsx"))

TEXT_COL = "resp_text"
LABEL_COL = "clarity"


# ==========================================
# 1. FUNÇÃO DE REMOÇÃO DE BOILERPLATE (REGEX)
# ==========================================
def clean_boilerplate_regex(text):
    """Aplica expressões regulares para remover ruídos burocráticos e saudações padrão."""
    if not isinstance(text, str) or not text.strip():
        return ""

    t = text

    # 1. Saudações de início
    pattern_saudacao = r'^(ol[aá]|bom dia|boa tarde|boa noite|prezado[as]?|caro[as]?)\b[:,!\.-]*\s*'
    t = re.sub(pattern_saudacao, '', t, flags=re.IGNORECASE)

    # 2. Protocolos, códigos e números de atendimento
    t = re.sub(r'\b(protocolo|atendimento|chamado|ticket|solicita[çc][ãa]o)\s*n?[º°]?\s*[:\.-]?\s*\d+\b', '', t, flags=re.IGNORECASE)

    # 3. E-mails e URLs
    t = re.sub(r'http[s]?://\S+|www\.\S+', '', t)
    t = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '', t)

    # 4. Despedidas e assinaturas comuns
    pattern_despedida = r'\b(atenciosamente|cordialmente|abra[çc]os?|obrigado[a]?|grato[a]?|atencioso[a]?)\b.*$'
    t = re.sub(pattern_despedida, '', t, flags=re.IGNORECASE)

    # 5. Limpeza de caracteres especiais sobra e múltiplos espaços/quebras
    t = re.sub(r'[\r\n\t]+', ' ', t)
    t = re.sub(r'\s{2,}', ' ', t)

    return t.strip()


# ==========================================
# EXECUÇÃO DO FLUXO
# ==========================================
if __name__ == "__main__":
    if not os.path.exists(PATH_INPUT):
        print(f"❌ Arquivo de entrada não encontrado em: {PATH_INPUT}")
        exit()

    print(f"Carregando dados de: {PATH_INPUT}...")
    df = pd.read_excel(PATH_INPUT)

    # 1. Limpeza de nulos essenciais
    df = df.dropna(subset=[TEXT_COL, LABEL_COL]).copy()
    df[LABEL_COL] = df[LABEL_COL].astype(str).str.strip().str.lower()

    # 2. Aplicação do Regex
    print("Limpando burocracia e boilerplate com Regex...")
    df["resp_text_clean"] = df[TEXT_COL].apply(clean_boilerplate_regex)

    # Remove registros que ficaram completamente vazios após a remoção de boilerplate
    df_filtered = df[df["resp_text_clean"].str.len() > 0].copy()

    # 3. Salvar nova base tratada
    print(f"Salvar nova base limpa em: {PATH_OUTPUT}...")
    df_filtered.to_excel(PATH_OUTPUT, index=False)
    print("✓ Arquivo salvo com sucesso!")

    # 4. Preparação do TF-IDF com N-gramas (1, 3)
    print("\n==================================================")
    print("   AVALIAÇÃO DE MODELO (TF-IDF N-GRAMAS 1-3)      ")
    print("==================================================")

    X_text = df_filtered["resp_text_clean"]
    y = df_filtered[LABEL_COL]

    print("Vetorizando textos com TF-IDF (ngram_range=(1, 3))...")
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 3),
        max_features=25000,
        sublinear_tf=True,
        min_df=2
    )
    X_tfidf = vectorizer.fit_transform(X_text)
    print(f"✓ Matriz TF-IDF gerada com formato: {X_tfidf.shape}")

    # 5. Validação Cruzada Estratificada (5-Fold)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    scoring = {
        'accuracy': 'accuracy',
        'f1_macro': 'f1_macro',
        'f1_weighted': 'f1_weighted'
    }

    models = {
        "Regressão Logística": LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced'),
        "Linear SVC (SVM)": LinearSVC(random_state=42, class_weight='balanced')
    }

    for name, model in models.items():
        print(f"\n--- Avaliando Modelo (5-Fold Cross-Validation): {name} ---")
        scores = cross_validate(model, X_tfidf, y, cv=cv, scoring=scoring, n_jobs=-1)

        acc = np.mean(scores['test_accuracy'])
        f1_m = np.mean(scores['test_f1_macro'])
        f1_w = np.mean(scores['test_f1_weighted'])

        print(f"  • Acurácia Média  : {acc:.4f} ({acc*100:.2f}%)")
        print(f"  • F1-Score Macro  : {f1_m:.4f}")
        print(f"  • F1-Score Weighted: {f1_w:.4f}")

    # 6. Relatório Detalhado no Holdout (Train/Test Split 80/20)
    print("\n--------------------------------------------------")
    print(" Relatório Detalhado por Classe (Holdout Teste 20%)")
    print("--------------------------------------------------")
    X_train, X_test, y_train, y_test = train_test_split(
        X_tfidf, y, test_size=0.2, random_state=42, stratify=y
    )

    clf = LinearSVC(random_state=42, class_weight='balanced')
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    print(classification_report(y_test, y_pred, digits=4))