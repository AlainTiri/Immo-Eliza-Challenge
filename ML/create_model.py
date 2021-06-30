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

    imputer = KNNImputer(n_neighbors = 3)
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

    # It's not relevant to train or test without target (Y)
    df = df[df['Price'].notna()]

    with open("df_empty_pre.pkl", 'wb') as file:
        df_empty = df.iloc[:0]
        pickle.dump(df_empty, file)

    df = preprocess_data(df)

    with open("df_empty_post.pkl", 'wb') as file:
        df_empty = df.iloc[:0]
        pickle.dump(df_empty, file)

    y = df.Price
    X = df.drop(columns="Price")

    # Splitting data into train and test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=41, test_size=0.2)

    model = xgb.XGBRegressor(random_state=0, n_jobs=6, max_depth=8, grow_policy = 'lossguide', max_leaves = 100,
                             max_bin = 512, reg_alpha = 0, reg_lambda = 0, n_estimators = 1000, learning_rate=0.1,
                             tree_method = 'gpu_hist')

    start = time.perf_counter()
    print("start training")
    model.fit(X_train, y_train)

    finish = time.perf_counter()
    print(f"Training fini en {round(finish-start, 2)} secondes")

    print("score train")
    print(model.score(X_train, y_train))
    print("score test")
    print(model.score(X_test, y_test))

    print("MSE : ", np.sqrt(((y_test - model.predict(X_test)) ** 2).mean()) )

    return model


    """
    df_train: pd.DataFrame = preprocess_data(df_train)
    df_test: pd.DataFrame = preprocess_data(df_test)

    for col in df_train.columns:
        if not col in df_test.columns:
            print("train to test, ", col)
            df_test[col] = 0
    for col in df_test.columns:
        if not col in df_train.columns:
            print("test to train", col)
            df_train[col] = 0"""
    """

    df_knn: np.ndarray = knn(df_train)
    np.save("fichierKNN.data", df_knn)
    # df_knn = np.load("fichierKNN.data.npy")

    y_train: np.ndarray = df_knn[:, 0]
    X_train: np.ndarray = df_knn[:, 1:]
    
    
    
    #df_knn = knn(df)
    #np.save("fichierKNN.data", df_knn)
    df_knn = np.load("fichierKNN.data.npy")

    y = df_knn[:, 0]  # .reshape(-1,1)
    X = df_knn[:, 1:]
    # y = y.T
    print(y.shape)
    
    
    
    """