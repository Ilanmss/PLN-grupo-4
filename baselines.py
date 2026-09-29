import os
import joblib
import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.metrics import f1_score


def main():
    PATH_DATA = "data/train.xlsx"
    DIR_SAIDA = "modelos/baselines"
    os.makedirs(DIR_SAIDA, exist_ok=True)

    print("Carregando base de dados...")
    df = pd.read_excel(PATH_DATA)
    X_text = df["resp_text"].fillna("").astype(str)
    Y = df["clarity"]

    x_train, x_test, y_train, y_test = train_test_split(
        X_text,
        Y,
        test_size=0.2,
        random_state=123,
        stratify=Y
    )

    resultados = {}

    # BASELINE CLASSE MAJORITÁRIA
    print("\nTreinando Dummy Classifier...")

    clf_maj = DummyClassifier(
        strategy="most_frequent",
        random_state=123
    )

    clf_maj.fit(x_train.to_frame(), y_train)

    predicted_maj = clf_maj.predict(x_test.to_frame())

    score_maj = f1_score(
        y_test,
        predicted_maj,
        average="macro"
    )

    joblib.dump(
        {
            "modelo": clf_maj
        },
        os.path.join(DIR_SAIDA, "baseline_classe_majoritaria.joblib")
    )

    resultados["Classe majoritária"] = score_maj

    print(f"F1-Macro: {score_maj:.4f}")
    print("Modelo salvo com sucesso.")



    # BASELINE BAG OF WORDS + REGRESSÃO LOGÍSTICA

    print("\nTreinando BoW + Logistic Regression...")

    vect_bow = CountVectorizer()

    X_train_bow = vect_bow.fit_transform(x_train)
    X_test_bow = vect_bow.transform(x_test)

    clf_bow = LogisticRegression(
        class_weight="balanced",
        max_iter=3000,
        random_state=123
    )

    clf_bow.fit(X_train_bow, y_train)

    predicted_bow = clf_bow.predict(X_test_bow)

    score_bow = f1_score(
        y_test,
        predicted_bow,
        average="macro"
    )

    pipeline_bow = {
        "vetorizador": vect_bow,
        "modelo": clf_bow
    }

    joblib.dump(
        pipeline_bow,
        os.path.join(DIR_SAIDA, "baseline_bow_logistic_regression.joblib")
    )

    resultados["BoW + LogReg"] = score_bow

    print(f"F1-Macro: {score_bow:.4f}")
    print("Modelo salvo com sucesso.")



    # BASELINE TF-IDF + REGRESSÃO LOGÍSTICA

    print("\nTreinando TF-IDF + Logistic Regression...")

    vect_tfidf = TfidfVectorizer()

    X_train_tfidf = vect_tfidf.fit_transform(x_train)
    X_test_tfidf = vect_tfidf.transform(x_test)

    clf_tfidf = LogisticRegression(
        class_weight="balanced",
        max_iter=3000,
        random_state=123
    )

    clf_tfidf.fit(X_train_tfidf, y_train)

    predicted_tfidf = clf_tfidf.predict(X_test_tfidf)

    score_tfidf = f1_score(
        y_test,
        predicted_tfidf,
        average="macro"
    )

    pipeline_tfidf = {
        "vetorizador": vect_tfidf,
        "modelo": clf_tfidf
    }

    joblib.dump(
        pipeline_tfidf,
        os.path.join(DIR_SAIDA, "baseline_tfidf_logistic_regression.joblib")
    )

    resultados["TF-IDF + LogReg"] = score_tfidf

    print(f"F1-Macro: {score_tfidf:.4f}")
    print("Modelo salvo com sucesso.")

    # RESUMO FINAL

    print("\n" + "=" * 50)
    print("RESUMO DOS BASELINES (F1-Macro)")
    print("=" * 50)

    for nome, score in resultados.items():
        print(f"{nome:<25}: {score:.4f}")

    print("=" * 50)

    print(f"\nModelos salvos em: {DIR_SAIDA}")

    PATH_RESULTADOS = os.path.join(DIR_SAIDA, "resultados_baselines.txt")
    with open(PATH_RESULTADOS, "w", encoding="utf-8") as arquivo:
        arquivo.write("RESULTADOS DOS BASELINES (F1-Macro)\n")
        arquivo.write("=" * 50 + "\n\n")

        for nome, score in resultados.items():
            arquivo.write(f"{nome:<25}: {score:.4f}\n")

        arquivo.write("\n" + "=" * 50 + "\n")
        arquivo.write("Configurações utilizadas:\n")
        arquivo.write("- Divisão treino/teste: 80/20\n")
        arquivo.write("- Random State: 123\n")
        arquivo.write("- Métrica: F1-Score Macro\n")
        arquivo.write("- Regressão Logística: class_weight='balanced'\n")
        arquivo.write("- Max Iterations: 3000\n")

    print(f"\nResultados salvos em: {PATH_RESULTADOS}")


if __name__ == "__main__":
    main()