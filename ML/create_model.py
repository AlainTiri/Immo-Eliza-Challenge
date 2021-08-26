import os
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split
import time
import pickle
from preprocess import preprocess_data

data_path = os.path.join(os.path.dirname(__file__), "Data", "database.csv")
path_model: str = "Eliza_XGB_Model.pkl"


def print_entetes():
    df: pd.DataFrame = pd.read_csv(data_path, index_col=0)

    df = df.drop(axis=1, labels="Url")
    df = df.drop(axis=1, labels="Source")
    df = df.drop(axis=1, labels="Surface area of the plot of land")
    df = df.drop(axis=1, labels="Region")
    df = df.drop(axis=1, labels="Province")
    df = df.drop(axis=1, labels="Type of sale")
    df = df.drop("Price", axis=1)

    print(df.columns)
    print(df["Subtype of property"].value_counts())


def knn(data_input: pd.DataFrame) -> np.ndarray:
    start = time.perf_counter()
    print("start knn")

    imputer = KNNImputer(n_neighbors = 10)
    df_knn = imputer.fit_transform(data_input)

    finish = time.perf_counter()
    print(f"KNN fini en {round(finish - start, 2)} secondes")
    return df_knn


def run():
    df: pd.DataFrame = pd.read_csv(data_path, index_col=0)

    model = train(df)

    with open(path_model, 'wb') as file:
        pickle.dump(model, file)


def train(df: pd.DataFrame) -> xgb.XGBRegressor:
    # It's not relevant to train or test without target (Y)
    df = df[df['Price'].notna()]

    # Splitting data into train and test split before using KNN
    df_train, df_test = train_test_split(df, random_state=41, test_size=0.2)

    df_train: pd.DataFrame = preprocess_data(df_train)
    df_knn: np.array = knn(df_train)
    np.save("fichierKNN.data", df_knn)

    y_train = df_knn[:, 0]
    X_train = df_knn[:, 1:]
    y_train = y_train.T

    model = xgb.XGBRegressor(random_state=0, n_jobs=6, max_depth=8, grow_policy = 'lossguide', max_leaves = 500,
                        max_bin = 512, reg_alpha = 5, reg_lambda = 5,
                        n_estimators = 6000, learning_rate=0.1, tree_method = 'gpu_hist')

    model.fit(X_train, y_train)

    df_test: pd.DataFrame = preprocess_data(df_test)
    y_test = df_test.Price
    X_test = df_test.drop(columns='Price')

    print("score train")
    print(model.score(X_train, y_train))
    print("score test")
    print(model.score(X_test, y_test))

    print("MSE : ", np.sqrt(((y_test - model.predict(X_test)) ** 2).mean()))

    return model


def mean_med_prices(df):
    df_prices = df.groupby('locality').count()
    print(df_prices)
