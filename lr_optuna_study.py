import optuna
import pandas as pd
from sklearn.svm import SVC
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import sys



def objective(trial):
    df = pd.read_csv("filtered_train.csv", index_col=False)
    df = df.rename(columns={"target": "is_disaster"})

    min_df = trial.suggest_int("min_df", 0, 20)
    ngram_range = trial.suggest_int("ngram_range", 1, 4)

    vectorizer = CountVectorizer(binary=True, min_df=min_df, ngram_range=(1, ngram_range))
    X = vectorizer.fit_transform(df.text.values)
    bag_of_words = pd.DataFrame(X.toarray(), columns=vectorizer.get_feature_names())
    df = pd.concat([df.drop(columns="Unnamed: 0"), bag_of_words], axis=1, join="inner")

    use_keywords = trial.suggest_int("use_keywords", 0, 1)
    if use_keywords:
        df = pd.concat([df.drop(columns=["keyword", "text"]), pd.get_dummies(df.keyword, dummy_na=True, prefix="keyword")], axis=1)
    else:
        df = df.drop(columns=["keyword", "text"])

    X_train, X_dev, y_train, y_dev = train_test_split(df.drop(columns=["is_disaster"]), df.is_disaster, train_size=0.7)

    C = trial.suggest_loguniform("C", 1e-10, 1e10)
    tol = trial.s

    clf = SVC(C=C, kernel=kernel, degree=degree, gamma=gamma).fit(X_train, y_train)

    return svm.score(X_dev, y_dev)

if __name__ == "__main__":
    study = optuna.create_study(direction="maximize", study_name=sys.argv[1], storage=sys.argv[2], load_if_exists=True)
    study.optimize(objective, n_trials=1000)