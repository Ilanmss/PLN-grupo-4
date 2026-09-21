# 📊 Classificação Automatizada de Textos em Respostas de Formulário

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat&logo=pandas&logoColor=white)](https://pandas.pydata.org/)

Projeto de Machine Learning e Processamento de Linguagem Natural (PLN) focado em avaliar e classificar a utilidade/clareza de respostas em texto de formulários. O sistema lida com um problema de **classificação ternária** baseada no agrupamento de níveis de informação (`c1`, `c234` e `c5`).

---
## 🛠️ Baseline: Métodos de Seleção & Parâmetros Testados

A suíte de experimentos (`treinar_baselines.py`) avalia automaticamente as seguintes dimensões:

| Parâmetro | Valores Testados |
| :--- | :--- |
| **Splits de Dados** | 70/30 e 80/20 |
| **Tamanho do Vocabulário** | 5k, 10k, 15k, 20k, 25k, 30k, 40k, 50k |
| **N-grams (Palavras)** | (1, 2) e (1, 3) |
| **N-grams (Caracteres)** | (3, 7) e (5, 9) |
| **TF-IDF Sublinear** | Ativado / Desativado |
| **Filtros de Frequência** | Ativado (`min_df=2`, `max_df=0.95`) / Desativado |
| **Seleção de Atributos** | `SelectKBest`, `SelectPercentile`, `TruncatedSVD` |

---

## 📁 Estrutura do Repositório

```text
├── data/                      # Dados de entrada (planilhas .xlsx, .csv)
├── modelos/                   # Modelos treinados salvos (.joblib)
│   └── baseline_LR_TF_IDF/
├── src/                       # Código-fonte do projeto
│   ├── __init__.py
│   ├── data_loader.py         # Leitura, limpeza inicial e split estratificado
│   ├── metricas.py            # Cálculo de métricas e plotagem de gráficos
│   └── selecao_atributos.py   # Módulos de seleção e redução de atributos
├── treinar_baselines.py       # Script principal para execução da grade de treinos
├── .gitignore                 # Arquivos ignorados pelo Git
├── README.md                  # Documentação do projeto
