import os
import joblib
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV

# Importação dos seus módulos internos em src/
from src.data_loader import load_data, split_data
import src.selecao_atributos as sa
import src.metricas as met

def main():
    PATH_DATA = "data/train.xlsx" # <--- AJUSTE O CAMINHO DO SEU ARQUIVO AQUI
    DIR_SAIDA = "modelos/baseline_LR_TF_IDF"
    os.makedirs(DIR_SAIDA, exist_ok=True)

    print("Carregando base de dados...")
    df = load_data(PATH_DATA)

    # Parametrização do Grid
    splits = [
        {"test_size": 0.3, "nome": "70_30"},
        {"test_size": 0.2, "nome": "80_20"}
    ]
    vocabularios = [5000, 10000, 15000, 20000, 25000, 30000, 40000, 50000]
    
    # N-gramas com tipo (palavra vs caractere)
    ngrams_config = [
        {"range": (1, 2), "analyzer": "word", "tag": "1_2_palavra"},
        {"range": (1, 3), "analyzer": "word", "tag": "1_3_palavra"},
        {"range": (3, 7), "analyzer": "char", "tag": "3_7_char"},
        {"range": (5, 9), "analyzer": "char", "tag": "5_9_char"}
    ]
    
    filtros_config = [
        {"min_df": 2, "max_df": 0.95, "tag": "on"},
        {"min_df": 1, "max_df": 1.0, "tag": "off"}
    ]
    
    sublinear_config = [
        {"sublinear_tf": True, "tag": "true"},
        {"sublinear_tf": False, "tag": "false"}
    ]

    seletores_nomes = [
        "kbest",
        "selectpercentile",
        "truncatedsvd",
    ]

    melhor_f1_macro = -1.0
    melhor_modelo_info = {}

    TOTAL_MODELOS = (
        len(splits)
        * len(vocabularios)
        * len(ngrams_config)
        * len(filtros_config)
        * len(sublinear_config)
        * len(seletores_nomes)
    )

    contador = 0

    # Treinamento
    for split in splits:
        X_train, X_val, y_train, y_val = split_data(df, test_size=split["test_size"], random_state=42)
        p_tag = split["nome"]

        for vocab in vocabularios:
            x_tag = f"{vocab // 1000}K"

            for ng in ngrams_config:
                yz_tag = ng["tag"]

                for filt in filtros_config:
                    k_tag = filt["tag"]

                    for sub in sublinear_config:
                        t_tag = sub["tag"]

                        vetorizador = TfidfVectorizer(
                            max_features=vocab,
                            ngram_range=ng["range"],
                            analyzer=ng["analyzer"],
                            min_df=filt["min_df"],
                            max_df=filt["max_df"],
                            sublinear_tf=sub["sublinear_tf"]
                        )
                        
                        X_train_vec = vetorizador.fit_transform(X_train)
                        X_val_vec = vetorizador.transform(X_val)

                        # Iteração pelos Métodos de Seleção de Atributos
                        for s_tag in seletores_nomes:
                            nome_modelo = f"vocab_{x_tag}_ngrm_{yz_tag}_filtros_{k_tag}_sublinear_{t_tag}_split_{p_tag}_{s_tag}"

                            print(f"[{contador:04d}/{TOTAL_MODELOS}] Treinando: {nome_modelo}...")

                            try:
                                if s_tag == "kbest":
                                    seletor, X_tr_sel = sa.aplicar_select_k_best(X_train_vec, y_train, k=min(2000, X_train_vec.shape[1]))
                                    X_va_sel = seletor.transform(X_val_vec)

                                elif s_tag == "selectpercentile":
                                    seletor, X_tr_sel = sa.aplicar_select_percentile(X_train_vec, y_train, percentile=10)
                                    X_va_sel = seletor.transform(X_val_vec)

                                elif s_tag == "truncatedsvd":
                                    n_comp = min(300, X_train_vec.shape[1] - 1)
                                    seletor, X_tr_sel = sa.aplicar_truncated_svd(X_train_vec, n_components=n_comp)
                                    X_va_sel = seletor.transform(X_val_vec)

                                # Treinamento da Regressão Logística
                                clf = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)
                                clf.fit(X_tr_sel, y_train)

                                # Avaliação
                                preds = clf.predict(X_va_sel)
                                
                                # Métricas
                                f1_macro = met.f1_score(y_val, preds, average='macro')
                                acc = met.accuracy_score(y_val, preds)
                                kappa = met.cohen_kappa_score(y_val, preds) 

                                print(f"[{nome_modelo}] -> F1-Macro: {f1_macro:.4f} | Acurácia: {acc:.4f} | Cohen Kappa: {kappa:.4f}")

                                # Salva o modelo
                                pipeline_export = {
                                    "vetorizador": vetorizador,
                                    "seletor": seletor,
                                    "modelo": clf
                                }
                                joblib.dump(pipeline_export, os.path.join(DIR_SAIDA, f"{nome_modelo}.joblib"))

                                # Melhor resultado
                                if f1_macro > melhor_f1_macro:
                                    melhor_f1_macro = f1_macro
                                    melhor_modelo_info = {
                                        "nome": nome_modelo,
                                        "modelo": clf,
                                        "X_val": X_va_sel,
                                        "y_val": y_val,
                                        "f1_macro": f1_macro
                                    }

                            except Exception as e:
                                print(f"[ERRO] Falha no modelo {nome_modelo}: {e}")


    # Melhor Resultado
    print("\n" + "="*80)
    print(f"MELHOR RESULTADO GLOBAL: {melhor_modelo_info['nome']}")
    print("="*80)


    met.calcular_todas_metricas(
        melhor_modelo_info["modelo"], 
        melhor_modelo_info["X_val"], 
        melhor_modelo_info["y_val"]
    )

if __name__ == "__main__":
    main()

# parametros utilizados:
    # split 70/30 e 80/20
    # vocabulário [5000, 10000, 15000, 20000, 25000, 30000, 40000, 50000]
    # ngramas palavras [(1, 2), (1, 3)]
    # ngramas characteres [(3, 7), (5, 9)]
    # filtros ON/OFF
    # sublinear TF ON/OFF
    # selecao de atributos: kbest, select percentile, truncated svd (rfe e arvore iam demorar de mais)
    
# ================================================================================
# MELHOR RESULTADO GLOBAL: vocab_50K_ngrm_1_3_palavra_filtros_off_sublinear_true_split_80_20_selectpercentile
# ================================================================================
# =============================================
#       RELATÓRIO DE MÉTRICAS DO MODELO       
# =============================================
# 1. F1-Score (Macro):  0.4470
# 2. Acurácia:          0.4501
# 3. Cohen's Kappa:     0.1752
# 4. ROC-AUC (OvR):     0.6298
# 5. Log Loss:          1.0457
# =============================================
