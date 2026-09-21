# Leitura dos dados .xlsx

import pandas as pd
from sklearn.model_selection import train_test_split

def load_data(path):  
  data = pd.read_excel(path)

  data['resp_text'] = data['resp_text'].fillna('')  # Substitui valores nulos por texto vazio
  data['resp_text'] = data['resp_text'].astype(str) # Garante que todo o conteúdo seja interpretado como string
    
  return data

def show_info_data(data):
  print("--- Primeiras 5 linhas da base ---")
  print(data.head())

  print("\n--- Distribuição das Classes (Contagem) ---")
  print(data['clarity' ].value_counts())

  print("\n--- Distribuição das Classes (Porcentagem) ---")
  print((data['clarity'].value_counts(normalize=True) * 100).round(2).astype(str) + '%')

def split_data(data, test_size=0.3, random_state=42):
  X = data['resp_text']
  y = data['clarity']
  
  X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)

  print(f"\nTotal de exemplos de treino: {len(X_train)}")
  print(f"Total de exemplos de validação: {len(X_val)}") 
  
  return X_train, X_val, y_train, y_val