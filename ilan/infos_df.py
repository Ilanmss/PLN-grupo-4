import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.manifold import TSNE
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
from scipy.spatial.distance import mahalanobis
from scipy.stats import entropy
from scipy.special import rel_entr

# spaCy para Embeddings 300D e Análise Gramatical (POS)
import spacy

# Tenta carregar UMAP como alternativa ao t-SNE
try:
    import umap.umap_ as umap
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False


# ==========================================
# CONFIGURAÇÕES GLOBAIS
# ==========================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PATH_DATA = os.path.abspath(os.path.join(BASE_DIR, "..", "data", "train_cleaned.xlsx"))
TEXT_COL = "resp_text"
LABEL_COL = "clarity"
VALID_CLASSES = {"c1", "c234", "c5"}
EMBEDDING_DIM = 300


# ==========================================
# 1. FUNÇÕES DE DIAGNÓSTICO DA BASE
# ==========================================
def check_label_integrity(data):
    """Valida se os rótulos estão padronizados e encontra inconsistências."""
    print("\n--- Validação Integridade das Classes/Labels ---")

    labels_raw = data[LABEL_COL].tolist()
    invalid_issues = []

    for idx, label in enumerate(labels_raw):
        if not isinstance(label, str):
            invalid_issues.append((idx, label, f"Tipo inválido: {type(label).__name__}"))
            continue

        if label != label.strip():
            invalid_issues.append((idx, label, "Espaços no início ou fim"))

        label_clean = label.strip()
        if label_clean not in VALID_CLASSES:
            if label_clean.lower() in VALID_CLASSES:
                invalid_issues.append((idx, label, "Diferença de Maiúscula/Minúscula"))
            else:
                invalid_issues.append((idx, label, "Valor completamente fora do padrão"))

    if invalid_issues:
        print(f"⚠ FORAM ENCONTRADOS {len(invalid_issues)} RÓTULOS FORA DO PADRÃO:")
        for idx, val, reason in invalid_issues:
            print(f"  - Linha {idx}: Valor '{val}' | Problema: {reason}")
    else:
        print(f"✓ Todos os rótulos estão padronizados estritamente como: {VALID_CLASSES}")


def check_duplicates_and_conflicts(data):
    """Verifica duplicatas totais e duplicatas do texto com rótulos conflitantes."""
    print("\n--- Verificação de Duplicatas e Conflitos ---")

    exact_dups = data.duplicated().sum()
    print(f"• Linhas 100% idênticas: {exact_dups}")

    text_dups = data.duplicated(subset=[TEXT_COL], keep=False)
    df_text_dups = data[text_dups]

    if not df_text_dups.empty:
        conflicting = (
            df_text_dups.groupby(TEXT_COL)[LABEL_COL]
            .nunique()
            .loc[lambda x: x > 1]
        )
        print(f"• Textos idênticos com rótulos DIVERGENTES (Conflitos): {len(conflicting)}")
        if len(conflicting) > 0:
            print("  ⚠ ATENÇÃO: Existem textos iguais categorizados com notas diferentes!")
    else:
        print("• Textos idênticos com rótulos divergentes: 0")


def check_vocabulary_overlap(data):
    """Calcula a sobreposição de vocabulário e similaridade de cosseno TF-IDF
    entre todos os pares de classes (c234 x c1, c234 x c5, c1 x c5).
    """
    print("\n--- Análise de Sobreposição de Vocabulário Lexical (TF-IDF Bruto) ---")

    df_clean = data.dropna(subset=[TEXT_COL, LABEL_COL]).copy()
    df_clean[LABEL_COL] = df_clean[LABEL_COL].astype(str).str.strip().str.lower()
    df_clean[TEXT_COL] = df_clean[TEXT_COL].astype(str)

    classes_presentes = [c for c in ["c1", "c234", "c5"] if c in df_clean[LABEL_COL].unique()]

    if len(classes_presentes) < 2:
        print("❌ Menos de 2 classes válidas encontradas para comparar.")
        return

    vectorizer = TfidfVectorizer(max_features=2000, stop_words=None, ngram_range=(1, 2))
    tfidf_matrix = vectorizer.fit_transform(df_clean[TEXT_COL])

    class_words = {}
    class_centroids = {}

    for cls in classes_presentes:
        mask = (df_clean[LABEL_COL] == cls).values
        words_in_cls = set(" ".join(df_clean.loc[mask, TEXT_COL]).lower().split())
        class_words[cls] = words_in_cls
        class_centroids[cls] = np.asarray(tfidf_matrix[mask].mean(axis=0))

    pairs = [("c234", "c1"), ("c234", "c5"), ("c1", "c5")]

    for cA, cB in pairs:
        if cA not in class_words or cB not in class_words:
            continue

        setA, setB = class_words[cA], class_words[cB]
        inter = setA.intersection(setB)
        union = setA.union(setB)
        jaccard = (len(inter) / len(union)) * 100 if union else 0
        cos_sim = cosine_similarity(class_centroids[cA], class_centroids[cB])[0][0]

        print(f"\n• Comparação [{cA.upper()} vs {cB.upper()}]:")
        print(f"  - Similaridade de Cosseno (TF-IDF Bruto): {cos_sim:.4f}")
        print(f"  - Vocabulário Compartilhado (Jaccard): {jaccard:.2f}% ({len(inter)} palavras em comum)")

        if cos_sim > 0.80:
            print(f"  ⚠️ ALERTA: Altíssima sobreposição entre {cA} e {cB}!")


def analyze_text_length(data):
    """Calcula estatísticas descritivas para caracteres e palavras."""
    texts = data[TEXT_COL].fillna("").astype(str)
    char_lengths = texts.apply(len)
    word_lengths = texts.apply(lambda x: len(x.split()))

    print("\n--- Estatísticas de Tamanho das Sentenças ---")
    print("• Caracteres:")
    print(f"  - Mais curta: {char_lengths.min()} | Mais longa: {char_lengths.max()}")
    print(f"  - Média: {char_lengths.mean():.2f} | Mediana: {char_lengths.median():.1f}")

    print("\n• Palavras:")
    print(f"  - Menos palavras: {word_lengths.min()} | Mais palavras: {word_lengths.max()}")
    print(f"  - Média: {word_lengths.mean():.2f} | Mediana: {word_lengths.median():.1f}")


def check_short_and_empty_texts(data, min_words=5):
    """Analisa textos em branco, nulos e textos com menos de N palavras."""
    print(f"\n--- Análise de Textos Curtos e Em Branco (< {min_words} palavras) ---")

    null_count = data[TEXT_COL].isnull().sum()
    empty_count = data[TEXT_COL].dropna().astype(str).str.strip().eq("").sum()
    
    print(f"• Textos NULOS (NaN): {null_count}")
    print(f"• Textos em BRANCO: {empty_count}")

    word_counts = data[TEXT_COL].fillna("").astype(str).apply(lambda x: len(x.split()))
    short_df = data[word_counts < min_words]
    print(f"• Quantidade de textos com menos de {min_words} palavras: {len(short_df)}")


def check_text_formatting_noise(data):
    """Identifica ruídos de formatação, HTML e espaços múltiplos."""
    print("\n--- Verificação de Ruídos de Formatação ---")
    texts = data[TEXT_COL].fillna("").astype(str)

    print(f"• Linhas não-string originais: {(~data[TEXT_COL].apply(lambda x: isinstance(x, str))).sum()}")
    print(f"• Múltiplos espaços seguidos: {texts.str.contains(r'\\s{2,}', regex=True).sum()}")
    print(f"• Tags HTML: {texts.str.contains(r'<[^>]+>', regex=True).sum()}")
    print(f"• Múltiplas quebras de linha: {texts.str.contains(r'\\n{2,}', regex=True).sum()}")


def show_info_data(data):
    """Executa a bateria de diagnósticos textuais."""
    print("\n==================================================")
    print("        RELATÓRIO DE DIAGNÓSTICO DA BASE          ")
    print("==================================================")

    print("\n--- Distribuição das Classes (Contagem) ---")
    print(data[LABEL_COL].value_counts(dropna=False))

    print("\n--- Distribuição das Classes (Porcentagem) ---")
    print((data[LABEL_COL].value_counts(normalize=True, dropna=False) * 100).round(2).astype(str) + "%")

    check_label_integrity(data)
    check_duplicates_and_conflicts(data)
    check_vocabulary_overlap(data)
    analyze_text_length(data)
    check_short_and_empty_texts(data, min_words=5)
    check_text_formatting_noise(data)


# ==========================================
# 2. EMBEDDINGS 300D PONDERADOS POR TF-IDF
# ==========================================
def compute_weighted_embeddings(df, nlp_model):
    """Calcula os embeddings 300D ponderados por TF-IDF global para cada linha."""
    print("\n==================================================")
    print("  PROCESSAMENTO DE EMBEDDINGS PONDERADOS (300D)   ")
    print("==================================================")
    print("1. Ajustando TfidfVectorizer global sobre todas as linhas...")
    
    texts = df[TEXT_COL].fillna("").astype(str).tolist()

    vectorizer = TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 1),
        max_df=0.95,
        min_df=2
    )
    
    tfidf_matrix = vectorizer.fit_transform(texts)
    feature_names = vectorizer.get_feature_names_out()
    word_to_id = {word: idx for idx, word in enumerate(feature_names)}

    print(f"✓ Vocabulário do TF-IDF: {len(feature_names)} palavras.")
    print("2. Gerando embeddings ponderados por TF-IDF para cada linha...")
    
    doc_vectors = []

    for idx, doc_text in enumerate(texts):
        doc = nlp_model(doc_text)
        
        vector_sum = np.zeros(EMBEDDING_DIM, dtype=np.float32)
        weight_sum = 0.0

        for token in doc:
            token_str = token.text.lower()
            
            if token.has_vector and token_str in word_to_id:
                word_idx = word_to_id[token_str]
                weight = tfidf_matrix[idx, word_idx]
                
                if weight > 0:
                    vector_sum += token.vector * weight
                    weight_sum += weight

        if weight_sum > 0:
            doc_vector = vector_sum / weight_sum
        else:
            doc_vector = np.zeros(EMBEDDING_DIM, dtype=np.float32)

        doc_vectors.append(doc_vector)

    return np.array(doc_vectors), df[LABEL_COL].values


# ==========================================
# 3. SUÍTE DE CÁLCULO DE MÉTRICAS (9 MÉTRICAS)
# ==========================================
def compute_all_metrics(df, X_embeddings, labels, nlp_model):
    """Executa o cálculo das 9 métricas de diagnóstico e separabilidade."""
    print("\n==================================================")
    print("    CÁLCULO DAS MÉTRICAS DE DIAGNÓSTICO E DISTÂNCIA ")
    print("==================================================")

    pairs = [("c234", "c1"), ("c234", "c5"), ("c1", "c5")]
    classes_presentes = list(np.unique(labels))

    # --------------------------------------------------
    # METRICA 1: Mahalanobis (com redução prévia via PCA só aqui)
    # --------------------------------------------------
    print("\n[Métrica 1] Distância de Mahalanobis (Com PCA = 20D):")
    pca = PCA(n_components=min(20, X_embeddings.shape[0], X_embeddings.shape[1] - 1), random_state=42)
    X_pca = pca.fit_transform(X_embeddings)
    
    cov_matrix = np.cov(X_pca, rowvar=False)
    # Adiciona pequena regularização para garantir que a matriz seja invertível
    cov_inv = np.linalg.pinv(cov_matrix + np.eye(cov_matrix.shape[0]) * 1e-6)

    centroids_pca = {c: X_pca[labels == c].mean(axis=0) for c in classes_presentes}

    for cA, cB in pairs:
        if cA in centroids_pca and cB in centroids_pca:
            dist = mahalanobis(centroids_pca[cA], centroids_pca[cB], cov_inv)
            print(f"  • {cA.upper()} vs {cB.upper()}: {dist:.4f}")

    # --------------------------------------------------
    # METRICA 2: Divergências Kullback-Leibler (KL) e Jensen-Shannon (JS)
    # --------------------------------------------------
    print("\n[Métrica 2] Divergência de Jensen-Shannon (JS) e KL (TF-IDF):")
    vec_prob = TfidfVectorizer(max_features=1000, stop_words=None)
    tfidf_prob = vec_prob.fit_transform(df[TEXT_COL].fillna("").astype(str)).toarray()
    
    # Adiciona constante para evitar divisão/log de zero e normaliza em distribuição
    tfidf_prob = tfidf_prob + 1e-9
    
    dist_prob = {}
    for c in classes_presentes:
        mask = (labels == c)
        p = tfidf_prob[mask].sum(axis=0)
        dist_prob[c] = p / p.sum()

    for cA, cB in pairs:
        if cA in dist_prob and cB in dist_prob:
            p, q = dist_prob[cA], dist_prob[cB]
            m = 0.5 * (p + q)
            js_div = 0.5 * entropy(p, m) + 0.5 * entropy(q, m)
            kl_div = np.sum(rel_entr(p, q))
            print(f"  • {cA.upper()} vs {cB.upper()} -> JS Divergence: {js_div:.4f} | KL Divergence: {kl_div:.4f}")

    # --------------------------------------------------
    # METRICA 4: Coeficiente de Overlap (Szymkiewicz-Simpson)
    # --------------------------------------------------
    print("\n[Métrica 4] Coeficiente de Overlap de Vocabulário:")
    vocab_by_class = {}
    for c in classes_presentes:
        texts_c = df[labels == c][TEXT_COL].fillna("").astype(str)
        vocab_by_class[c] = set(" ".join(texts_c).lower().split())

    for cA, cB in pairs:
        if cA in vocab_by_class and cB in vocab_by_class:
            setA, setB = vocab_by_class[cA], vocab_by_class[cB]
            inter = len(setA.intersection(setB))
            overlap = inter / min(len(setA), len(setB)) if min(len(setA), len(setB)) > 0 else 0
            print(f"  • {cA.upper()} vs {cB.upper()}: {overlap:.4f} ({overlap*100:.2f}%)")

    # --------------------------------------------------
    # METRICA 5: Riqueza Lexical Móvel (MATTR - Moving Average TTR)
    # --------------------------------------------------
    print("\n[Métrica 5] Riqueza Lexical Móvel (MATTR - Janela 50 palavras):")
    def compute_mattr(text, window_size=50):
        tokens = text.lower().split()
        if len(tokens) < window_size:
            return len(set(tokens)) / len(tokens) if len(tokens) > 0 else 0
        ttrs = [len(set(tokens[i:i+window_size])) / window_size for i in range(len(tokens) - window_size + 1)]
        return np.mean(ttrs)

    for c in classes_presentes:
        texts_c = df[labels == c][TEXT_COL].fillna("").astype(str)
        mattr_vals = [compute_mattr(t) for t in texts_c]
        print(f"  • Classe {c.upper()}: {np.mean(mattr_vals):.4f}")

    # --------------------------------------------------
    # EXTRAÇÃO DE POS TAGS (Base para Métricas 6 e 7)
    # --------------------------------------------------
    print("\n[Métrica 6 & 7] Extraindo POS Tags via spaCy...")
    texts_list = df[TEXT_COL].fillna("").astype(str).tolist()
    pos_sequences = []
    
    # Processa os documentos no spaCy para capturar as etiquetas gramaticais
    for doc in nlp_model.pipe(texts_list, batch_size=256):
        pos_sequences.append(" ".join([token.pos_ for token in doc]))
        
    df['_pos_sequence'] = pos_sequences

    # --------------------------------------------------
    # METRICA 6: Similaridade sobre N-gramas de POS
    # --------------------------------------------------
    print("\n[Métrica 6] Similaridade de Cosseno sobre N-gramas de POS (1-3):")
    vec_pos = TfidfVectorizer(ngram_range=(1, 3))
    pos_tfidf = vec_pos.fit_transform(df['_pos_sequence'])

    pos_centroids = {}
    for c in classes_presentes:
        mask = (labels == c)
        pos_centroids[c] = np.asarray(pos_tfidf[mask].mean(axis=0))

    for cA, cB in pairs:
        if cA in pos_centroids and cB in pos_centroids:
            sim = cosine_similarity(pos_centroids[cA], pos_centroids[cB])[0][0]
            print(f"  • {cA.upper()} vs {cB.upper()}: {sim:.4f}")

    # --------------------------------------------------
    # METRICA 7: Entropia das Classes Gramaticais (POS Entropy)
    # --------------------------------------------------
    print("\n[Métrica 7] Entropia de Shannon sobre a Distribuição de POS:")
    for c in classes_presentes:
        pos_text_c = " ".join(df[labels == c]['_pos_sequence']).split()
        if pos_text_c:
            _, counts = np.unique(pos_text_c, return_counts=True)
            probs = counts / counts.sum()
            pos_entropy = entropy(probs, base=2)
            print(f"  • Classe {c.upper()}: {pos_entropy:.4f} bits")

    # --------------------------------------------------
    # METRICA 8: Distância de Formato e Densidade de Pontuação
    # --------------------------------------------------
    print("\n[Métrica 8] Densidade Média de Pontuação por Caractere:")
    punct_regex = re.compile(r'[^\w\s]')
    for c in classes_presentes:
        texts_c = df[labels == c][TEXT_COL].fillna("").astype(str)
        punct_densities = [
            len(punct_regex.findall(t)) / len(t) if len(t) > 0 else 0 
            for t in texts_c
        ]
        print(f"  • Classe {c.upper()}: {np.mean(punct_densities):.4f}")

    # --------------------------------------------------
    # METRICA 9: Densidade de Redundância (N-gramas Repetidos)
    # --------------------------------------------------
    print("\n[Métrica 9] Densidade Média de Bigramas Repetidos por Texto:")
    def count_duplicate_bigrams(text):
        tokens = text.lower().split()
        if len(tokens) < 2:
            return 0
        bigrams = [f"{tokens[i]}_{tokens[i+1]}" for i in range(len(tokens)-1)]
        return (len(bigrams) - len(set(bigrams))) / len(bigrams)

    for c in classes_presentes:
        texts_c = df[labels == c][TEXT_COL].fillna("").astype(str)
        redundancy_vals = [count_duplicate_bigrams(t) for t in texts_c]
        print(f"  • Classe {c.upper()}: {np.mean(redundancy_vals):.4f}")

    # --------------------------------------------------
    # METRICA 10: Coeficiente de Silhueta e Davies-Bouldin
    # --------------------------------------------------
    print("\n[Métrica 10] Métricas Globais de Agrupamento (Embeddings 300D):")
    sil_score = silhouette_score(X_embeddings, labels, metric='cosine')
    db_score = davies_bouldin_score(X_embeddings, labels)
    print(f"  • Coeficiente de Silhueta (Cosseno) [-1 a 1]: {sil_score:.4f}")
    print(f"  • Índice Davies-Bouldin (Menor é melhor)   : {db_score:.4f}")


# ==========================================
# 4. ANÁLISE DE SOBREPOSIÇÃO DOS EMBEDDINGS
# ==========================================
def measure_class_overlap(X_embeddings, labels):
    """Calcula a sobreposição dos embeddings entre as classes."""
    print("\n==================================================")
    print("   MÉTRICAS DE SOBREPOSIÇÃO NOS EMBEDDINGS 300D  ")
    print("==================================================")

    unique_classes = np.unique(labels)
    centroids = {}

    for cls in unique_classes:
        mask = (labels == cls)
        centroids[cls] = X_embeddings[mask].mean(axis=0).reshape(1, -1)

    print("\n1. Similaridade de Cosseno entre os Centroides (Embeddings Ponderados):")
    pairs = [("c234", "c1"), ("c234", "c5"), ("c1", "c5")]
    for cA, cB in pairs:
        if cA in centroids and cB in centroids:
            sim = cosine_similarity(centroids[cA], centroids[cB])[0][0]
            print(f"  • {cA.upper()} vs {cB.upper()}: {sim:.4f}")

    # Pureza de vizinhança k-NN
    k = 10
    knn = NearestNeighbors(n_neighbors=k+1, metric='cosine')
    knn.fit(X_embeddings)
    indices = knn.kneighbors(X_embeddings, return_distance=False)

    same_class_count = 0
    total_neighbors = len(labels) * k

    for i in range(len(labels)):
        neighbors_idx = indices[i][1:]
        same_class_count += np.sum(labels[neighbors_idx] == labels[i])

    purity = (same_class_count / total_neighbors) * 100
    print(f"\n2. Pureza de Vizinhança (k-NN, k={k}): {purity:.2f}%")
    print("   (Mede a % de vizinhos mais próximos que compartilham a mesma classe).")


# ==========================================
# 5. VISUALIZAÇÃO DE DISPERSÃO 2D
# ==========================================
def plot_embeddings_2d(X_embeddings, labels, method='tsne'):
    """Reduz a dimensão para 2D e plota o gráfico de dispersão."""
    print(f"\nReduzindo dimensionalidade usando {method.upper()}...")
    
    if method == 'umap' and HAS_UMAP:
        reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric='cosine', random_state=42)
        X_2d = reducer.fit_transform(X_embeddings)
    else:
        reducer = TSNE(n_components=2, perplexity=30, metric='cosine', random_state=42, n_jobs=-1)
        X_2d = reducer.fit_transform(X_embeddings)

    plt.figure(figsize=(10, 7))
    palette = {'c1': '#e74c3c', 'c234': '#f39c12', 'c5': '#2ecc71'}

    sns.scatterplot(
        x=X_2d[:, 0],
        y=X_2d[:, 1],
        hue=labels,
        palette=palette,
        alpha=0.6,
        s=25
    )

    plt.title(f"Dispersão 2D dos Textos ({method.upper()} no Embedding 300D Ponderado por TF-IDF)", fontsize=12)
    plt.xlabel("Dimensão 1")
    plt.ylabel("Dimensão 2")
    plt.legend(title="Classe / Clareza")
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.show()


# ==========================================
# EXECUÇÃO PRINCIPAL FLUXO COMPLETO
# ==========================================
if __name__ == "__main__":
    if os.path.exists(PATH_DATA):
        print(f"Carregando a base de dados de: {PATH_DATA}...")
        df = pd.read_excel(PATH_DATA)

        # 1. Diagnóstico Inicial da Base
        show_info_data(df)

        # 2. Carregar modelo de linguagem de 300D
        print("\nCarregando modelo de linguagem spaCy (300D)...")
        try:
            # Mantemos a extração de POS ativa para as métricas 6 e 7
            nlp = spacy.load("pt_core_news_lg", disable=["ner", "parser"])
        except OSError:
            print("❌ Modelo 'pt_core_news_lg' não encontrado!")
            print("Instale executando no terminal: python -m spacy download pt_core_news_lg")
            exit()

        # 3. Gerar Embeddings Ponderados por TF-IDF
        X_embed, y_labels = compute_weighted_embeddings(df, nlp)

        # 4. Medir sobreposição dos embeddings
        measure_class_overlap(X_embed, y_labels)

        # 5. Executar a Bateria Completa de 9 Métricas
        compute_all_metrics(df, X_embed, y_labels, nlp)

        # 6. Plotar Dispersão em 2D (t-SNE / UMAP)
        plot_embeddings_2d(X_embed, y_labels, method='tsne')

    else:
        print(f"❌ Arquivo não encontrado no caminho: {PATH_DATA}. Verifique o diretório.")