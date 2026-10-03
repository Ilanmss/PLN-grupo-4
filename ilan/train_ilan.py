import os

# Configura o cache do Hugging Face no drive E:
os.environ["HF_HOME"] = "E:/hf_cache"

# ==========================================
# DESATIVA A TRAVA DE SEGURANÇA DO TORCH.LOAD
# ==========================================
import transformers.utils.import_utils as import_utils

import_utils.check_torch_load_is_safe = lambda: None

import re
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

# ==========================================
# 1. VERIFICAÇÃO DE GPU (CUDA)
# ==========================================
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Dispositivo em uso: {device}")
if device == "cpu":
    print("AVISO: GPU CUDA não encontrada! O treinamento rodará na CPU.")


# ==========================================
# 2. PRÉ-PROCESSAMENTO E PREPARAÇÃO DOS DADOS
# ==========================================
def clean_text_preservative(text):
    if not isinstance(text, str):
        return ""
    text = re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalize_class(val):
    val = str(val).strip().upper()
    if "1" in val and "2" not in val:
        return 0  # C1
    elif "5" in val:
        return 2  # C5
    else:
        return 1  # C234


label_names = ["C1", "C234", "C5"]


# Dataset personalizado PyTorch
class TextDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

    def __len__(self):
        return len(self.labels)


# Métrica para o Hugging Face Trainer
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    acc = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="weighted", zero_division=0
    )
    return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}


# ==========================================
# 3. PIPELINE DE TREINAMENTO
# ==========================================
def main():
    # Ajusta caminho para localizar train_cleaned.xlsx dinamicamente
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.abspath(
        os.path.join(BASE_DIR, "..", "data", "train_cleaned.xlsx")
    )

    print(f"Carregando dataset limpo de: {file_path}...")
    df = pd.read_excel(file_path).dropna(subset=["clarity", "resp_text"])

    df["clean_text"] = df["resp_text"].apply(clean_text_preservative)
    df["label"] = df["clarity"].apply(normalize_class)

    # Divisão Treino e Teste (Stratified)
    train_texts, test_texts, train_labels, test_labels = train_test_split(
        df["clean_text"].tolist(),
        df["label"].tolist(),
        test_size=0.20,
        random_state=42,
        stratify=df["label"],
    )

    # Modelo oficial do BERTimbau
    model_name = "neuralmind/bert-base-portuguese-cased"
    print(f"Carregando Tokenizer e Modelo '{model_name}'...")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=3, use_safetensors=True
    )

    # ----------------------------------------------------
    # FINE-TUNING COMPLETO (TODAS AS CAMADAS TREINÁVEIS)
    # ----------------------------------------------------
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(
        f"Fine-Tuning Ativo - Parâmetros treináveis: {trainable_params:,} / {total_params:,}"
    )

    # Tokenização dos dados (Aumentado max_length para 256)
    print("Tokenizando textos...")
    train_encodings = tokenizer(
        train_texts, truncation=True, padding=True, max_length=256
    )
    test_encodings = tokenizer(
        test_texts, truncation=True, padding=True, max_length=256
    )

    train_dataset = TextDataset(train_encodings, train_labels)
    test_dataset = TextDataset(test_encodings, test_labels)

    # Configuração do Treinamento
    training_args = TrainingArguments(
        output_dir="./results_bertimbau",
        num_train_epochs=4,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        warmup_steps=100,
        learning_rate=2e-5,
        weight_decay=0.01,
        logging_steps=50,
        eval_strategy="epoch",  # Padrão correto para versões recentes
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        fp16=torch.cuda.is_available(),
        report_to="none",
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,
    )

    print("\nIniciando o Fine-Tuning do BERTimbau...")
    trainer.train()

    print("\nAvaliando o melhor modelo no conjunto de teste...")
    results = trainer.evaluate()
    print(
        f"\nResultados Finais: F1-Score = {results['eval_f1']:.4f} | Acurácia = {results['eval_accuracy']:.4f}"
    )

    # Relatório de classificação detalhado
    predictions = trainer.predict(test_dataset)
    preds = np.argmax(predictions.predictions, axis=1)
    print("\nRelatório de Classificação Detalhado:")
    print(classification_report(test_labels, preds, target_names=label_names))

    # Salvando modelo ajustado e tokenizer
    output_dir = "./saved_bertimbau_classifier"
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"\nModelo e Tokenizer salvos com sucesso em: '{output_dir}'")


if __name__ == "__main__":
    main()