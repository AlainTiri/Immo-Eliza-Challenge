import os
import numpy as np
import pandas as pd
import xgboost
import xgboost as xgb
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split
import time
import pickle
from preprocess import preprocess_data

data_path = os.path.join(os.path.dirname(__file__), "Data", "database.csv")
path_model: str = "Eliza_XGB_Model.pkl"


def knn(data_input: pd.DataFrame) -> np.ndarray:
    start = time.perf_counter()
    print("start knn")

    imputer = KNNImputer(n_neighbors = 10)
    df_knn = imputer.fit_transform(data_input)

    finish = time.perf_counter()
    print(f"KNN fini en {round(finish-start, 2)} secondes")
    return df_knn


def run():
    df: pd.DataFrame = pd.read_csv(data_path, index_col=0)

    model = train(df)

    with open(path_model, 'wb') as file:
        pickle.dump(model, file)


def train(df: pd.DataFrame) -> xgboost.XGBRegressor:

    # It's not relevant to train and test without target (Y)
    df = df[df['Price'].notna()]
    print("df")
    print(df.shape)
    print(df.columns)

    # Splitting data into train and test split before using KNN
    df_train, df_test = train_test_split(df, random_state=41, test_size=0.2)
    print ("train")
    print(df_train.shape)

    df_train: pd.DataFrame = preprocess_data(df_train)
    df_test: pd.DataFrame = preprocess_data(df_test)

    print(df_train.shape)  #42
    print(df_train.columns)


    #df_knn: np.ndarray = knn(df_train)
    #np.save("fichierKNN.data", df_knn)
    df_knn = np.load("fichierKNN.data.npy")

    y_train: np.ndarray = df_knn[:, 0]
    X_train: np.ndarray = df_knn[:, 1:]
    print("x y train")
    print(y_train.shape, X_train.shape)  # 41
    #y_train = y_train.T
    print(y_train.shape, X_train.shape)

    model = xgb.XGBRegressor(random_state=0, n_jobs=6, max_depth=8, grow_policy = 'lossguide', max_leaves = 500,
                        max_bin = 512, reg_alpha = 5, reg_lambda = 5,
                        n_estimators = 100, learning_rate=0.1, tree_method = 'gpu_hist')

    start = time.perf_counter()
    print("start training")
    model.fit(X_train, y_train)

    finish = time.perf_counter()
    print(f"Training fini en {round(finish-start, 2)} secondes")

    print("test")
    print(df_test.columns)
    print(df_test.shape)  # 43
    y_test: pd.Series = df_test["Price"]
    X_test: pd.DataFrame = df_test.drop(columns='Price')
    print("x y test")
    print(X_test.shape, y_test.shape)

    print(model.score(X_train, y_train))
    print(model.score(X_test, y_test))

    return model

