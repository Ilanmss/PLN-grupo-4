import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

from gensim.models import Word2Vec
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, f1_score, confusion_matrix


# ==========================================
# CONFIGURAÇÕES DE CAMINHO E PARÂMETROS
# ==========================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PATH_INPUT = os.path.abspath(os.path.join(BASE_DIR, "..", "data", "train_cleaned_no_boilerplate.xlsx"))
PATH_MODEL_SAVE = os.path.abspath(os.path.join(BASE_DIR, "best_nn_model.pt"))

TEXT_COL = "resp_text_clean"
LABEL_COL = "clarity"
EMBEDDING_DIM = 100
EPOCHS = 100
BATCH_SIZE = 64
LEARNING_RATE = 0.001
PATIENCE = 10  # Early Stopping (parar se não melhorar F1 em N épocas)


# ==========================================
# DEFINIÇÃO DA REDE NEURAL (PyTorch)
# ==========================================
class TextClassifierNN(nn.Module):
    def __init__(self, input_dim=100, hidden_dim=32, num_classes=3):
        super(TextClassifierNN, self).__init__()
        # Requisito 5 & 6: Entradas=100D, Oculta=32 neurônios com ReLU
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        # Requisito 7: Saída = 3 neurônios (Softmax é aplicado via CrossEntropyLoss no treino)
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out


# ==========================================
# EXECUÇÃO DO FLUXO
# ==========================================
if __name__ == "__main__":
    if not os.path.exists(PATH_INPUT):
        print(f"❌ Arquivo de entrada não encontrado em: {PATH_INPUT}")
        exit()

    print(f"Carregando dados de: {PATH_INPUT}...")
    df = pd.read_excel(PATH_INPUT).dropna(subset=[TEXT_COL, LABEL_COL])

    texts = df[TEXT_COL].astype(str).tolist()
    labels = df[LABEL_COL].astype(str).str.strip().str.lower().tolist()

    # Mapeamento de Rótulos para Números (c1, c234, c5)
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(labels)
    classes_list = label_encoder.classes_
    print(f"Classes mapeadas: {list(enumerate(classes_list))}")

    # Tokenização simples por espaço
    tokenized_texts = [text.lower().split() for text in texts]

    # --------------------------------------------------
    # PASSO 1 & 3: TF-IDF (Frequência Ponderada Inversa)
    # --------------------------------------------------
    print("\n[Passo 1 & 3] Calculando matriz TF-IDF...")
    tfidf = TfidfVectorizer(tokenizer=lambda x: x, preprocessor=lambda x: x, token_pattern=None)
    tfidf_matrix = tfidf.fit_transform(tokenized_texts)
    feature_names = tfidf.get_feature_names_out()
    word2tfidf = dict(zip(feature_names, tfidf.idf_))  # IDF dá peso maior para palavras mais raras

    # --------------------------------------------------
    # PASSO 2: Treinando Word2Vec (100D) nas palavras
    # --------------------------------------------------
    print("\n[Passo 2] Treinando embeddings Word2Vec (100D)...")
    w2v_model = Word2Vec(
        sentences=tokenized_texts,
        vector_size=EMBEDDING_DIM,
        window=5,
        min_count=1,
        workers=4,
        seed=42
    )

    # --------------------------------------------------
    # PASSO 3 & 4: Média Ponderada pelo TF-IDF por Linha
    # --------------------------------------------------
    print("\n[Passo 3 & 4] Gerando vetores 100D por média ponderada TF-IDF...")
    X_vectors = []

    for tokens in tokenized_texts:
        vector_acc = np.zeros(EMBEDDING_DIM, dtype=np.float32)
        weight_sum = 0.0

        for token in tokens:
            if token in w2v_model.wv and token in word2tfidf:
                weight = word2tfidf[token]  # Quanto mais rara a palavra, maior o peso
                vector_acc += w2v_model.wv[token] * weight
                weight_sum += weight

        if weight_sum > 0:
            vector_acc /= weight_sum
        
        X_vectors.append(vector_acc)

    X_vectors = np.array(X_vectors, dtype=np.float32)

    # --------------------------------------------------
    # DIVISÃO TREINO / VALIDAÇÃO (80 / 20)
    # --------------------------------------------------
    X_train, X_val, y_train, y_val = train_test_split(
        X_vectors, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )

    # Tensores e Dataloaders do PyTorch
    train_dataset = TensorDataset(torch.tensor(X_train), torch.tensor(y_train, dtype=torch.long))
    val_dataset = TensorDataset(torch.tensor(X_val), torch.tensor(y_val, dtype=torch.long))

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # --------------------------------------------------
    # PASSO 5, 6, 7 & 12: Modelo, Loss e Otimizador Dinâmico
    # --------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TextClassifierNN(input_dim=EMBEDDING_DIM, hidden_dim=32, num_classes=len(classes_list)).to(device)

    criterion = nn.CrossEntropyLoss()
    # Requisito 12: Adam (Ajuste dinâmico das taxas de aprendizado dos pesos)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # --------------------------------------------------
    # PASSO 8, 9, 10 & 11: Treinamento, Early Stopping & Best Model
    # --------------------------------------------------
    print("\n[Passos 8 a 12] Iniciando Treinamento da Rede Neural...")
    
    best_f1 = 0.0
    patience_counter = 0

    for epoch in range(1, EPOCHS + 1):
        # Modo Treino
        model.train()
        running_loss = 0.0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * batch_x.size(0)

        # Modo Avaliação (Validação)
        model.eval()
        val_preds = []
        val_targets = []

        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                outputs = model(batch_x)
                preds = torch.argmax(outputs, dim=1)

                val_preds.extend(preds.cpu().numpy())
                val_targets.extend(batch_y.cpu().numpy())

        # Métricas
        val_acc = accuracy_score(val_targets, val_preds)
        val_f1_macro = f1_score(val_targets, val_preds, average='macro')

        # Requisito 8: Vazamento de C1 -> C5 e C5 -> C1
        cm = confusion_matrix(val_targets, val_preds)
        
        # Mapear índices de c1 e c5
        idx_c1 = np.where(classes_list == 'c1')[0][0]
        idx_c5 = np.where(classes_list == 'c5')[0][0]
        
        vazamento_c1_para_c5 = cm[idx_c1, idx_c5]  # Era C1 mas previu C5
        vazamento_c5_para_c1 = cm[idx_c5, idx_c1]  # Era C5 mas previu C1

        print(f"Época {epoch:03d}/{EPOCHS:03d} | Loss: {running_loss/len(X_train):.4f} | "
              f"Val Acc: {val_acc:.4f} | Val F1 Macro: {val_f1_macro:.4f} | "
              f"Vazamento C1->C5: {vazamento_c1_para_c5} | C5->C1: {vazamento_c5_para_c1}")

        # Requisito 11: Salvar o melhor modelo baseado no F1 Macro
        if val_f1_macro > best_f1:
            best_f1 = val_f1_macro
            patience_counter = 0
            torch.save(model.state_dict(), PATH_MODEL_SAVE)
        else:
            patience_counter += 1

        # Requisito 10: Parar se começar Overfitting / Estagnar (Early Stopping)
        if patience_counter >= PATIENCE:
            print(f"\n✋ Treinamento interrompido na época {epoch} por Early Stopping (Overfitting/Estagnação).")
            break

    # --------------------------------------------------
    # RELATÓRIO FINAL COM O MELHOR MODELO SALVO
    # --------------------------------------------------
    print("\n==================================================")
    print("      RELATÓRIO FINAL (MELHOR MODELO SALVO)       ")
    print("==================================================")
    model.load_state_dict(torch.load(PATH_MODEL_SAVE))
    model.eval()

    val_preds = []
    val_targets = []
    with torch.no_grad():
        for batch_x, batch_y in val_loader:
            batch_x = batch_x.to(device)
            outputs = model(batch_x)
            preds = torch.argmax(outputs, dim=1)
            val_preds.extend(preds.cpu().numpy())
            val_targets.extend(batch_y.numpy())

    print(classification_report(val_targets, val_preds, target_names=classes_list, digits=4))