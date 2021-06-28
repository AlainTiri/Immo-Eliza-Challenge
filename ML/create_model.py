
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split
import time
import pickle
import preprocess


path_model: str = "Eliza_XGB_Model.pkl"


def knn(data_input: pd.DataFrame) -> np.array:
    start = time.perf_counter()
    print("start knn")

    imputer = KNNImputer(n_neighbors = 10)
    df_knn = imputer.fit_transform(data_input)

    finish = time.perf_counter()
    print(f"KNN fini en {round(finish-start, 2)} secondes")
    return df_knn


def run():
    df: pd.DataFrame = pd.read_csv("./Data/database.csv", index_col=0)
    df = preprocess(df)

    model = train(df)

    with open(path_model, 'wb') as file:
        pickle.dump(model, file)


def train(df: pd.DataFrame):

    # It's not relevant to train and test without target (Y)
    df = df[df['Price'].notna()]

    # Splitting data into train and test split before using KNN
    df_train, df_test = train_test_split(df, random_state=41, test_size=0.2)

    df_train: pd.DataFrame = preprocess(df_train)
    df_knn: np.array = knn(df_train)
    np.save("fichierKNN.data", df_knn)

    y_train = df_knn[:, 0]
    X_train = df_knn[:, 1:]
    y_train = y_train.T

    model = xgb.XGBRegressor(random_state=0, n_jobs=6, max_depth=8, grow_policy = 'lossguide', max_leaves = 500,
                        max_bin = 512, reg_alpha = 5, reg_lambda = 5,
                        n_estimators = 6000, learning_rate=0.1, tree_method = 'gpu_hist')

    model.fit(X_train, y_train)

    df_test: pd.DataFrame = preprocess(df_test)
    y_test = df_test.Price
    X_test = df_test.drop(columns='Price')

    print(model.score(X_train, y_train), model.score(X_test, y_test), (model.score(X_train, y_train)- model.score(X_test, y_test)))

    return model

