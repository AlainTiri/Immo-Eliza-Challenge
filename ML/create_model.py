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
zip_path = os.path.join(os.path.dirname(__file__), "Data", "zipcodes.csv")
mm_zip_path = os.path.join(os.path.dirname(__file__), "Data", "mean_median_zipcodes.csv")

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


def zip_trunc(zip_code: int) -> int:
    """
    :param zip_code: Postal code (ex. 1341)
    :return: Postal code of pincipal locality (ex. 1340)
    """
    rest = zip_code % 10
    zip_code = zip_code - rest
    return zip_code


def mean_and_median_by_locality(df: pd.DataFrame, df_zip_code: pd.DataFrame) -> pd.DataFrame:
    """
    Calcul mean and median price on each locality. If there are too less prices for to calculate a significante price
    (<5 choosen abritrary), it's the mean/median price for the entities.
    TO Do : Process must be optimized
    :param df: Principal DataFrame with all data of real estate
    :param df_zip_code: DataFrame with list of all postal code of Belgium
    :return: A DataFrame with : ZipeCode, Locality Name, ZipCode Truncatened, Mean Price on each zipcode (exact or
    estimated), Median Price on each zipcode (exact estimated).
    """

    # To truncate zip code
    df_zip_code["zip_trunck"] = df_zip_code.zipcode.apply(zip_trunc)

    # count numbers of goods by locality
    df_zip_code["count_CP"] = 0
    df_prices = df.groupby("Locality").count()
    s_count = df_prices.Province

    # Add on a new df
    new_df = pd.DataFrame(s_count.rename("count_CP", inplace=True))
    new_df.reset_index(level=0, inplace=True)

    # Compute mean and median for each zipcode
    for row in new_df.itertuples():
        df_zip_code.loc[df_zip_code["zipcode"] == row.Locality, 'count_CP'] = row.count_CP
        select_zip = df[df['Locality'] == row.Locality]
        moyenne = select_zip.mean()
        moyenne = moyenne['Price']
        mediane = select_zip.median()
        mediane = mediane['Price']
        new_df.loc[new_df["Locality"] == row.Locality, 'moyenne'] = moyenne
        df_zip_code.loc[df_zip_code["zip_trunck"] == row.Locality, 'moyenne'] = moyenne

        new_df.loc[new_df["Locality"] == row.Locality, 'mediane'] = mediane
        df_zip_code.loc[df_zip_code["zip_trunck"] == row.Locality, 'mediane'] = mediane

    # Add zip truncated on principal df
    df["zip_trunck"] = df["Locality"].apply(zip_trunc)

    # Compute mean and median price by zip simplified for each zipcode which have less than 5 goods to sell
    # Then whose have 5 or more goods keep they prices mean/median calculated above
    for row in df_zip_code.itertuples():
        if row.count_CP < 5:
            select_zip = df[df['zip_trunck'] == row.zip_trunck]
            moyenne = select_zip.mean()
            moyenne = moyenne['Price']
            mediane = select_zip.median()
            mediane = mediane['Price']
            new_df.loc[new_df["Locality"] == row.zipcode, 'moyenne'] = moyenne
            df_zip_code.loc[df_zip_code["zipcode"] == row.zipcode, 'moyenne'] = moyenne

            new_df.loc[new_df["Locality"] == row.zipcode, 'mediane'] = mediane
            df_zip_code.loc[df_zip_code["zip_trunck"] == row.zipcode, 'mediane'] = mediane

    return df_zip_code


def run():
    df: pd.DataFrame = pd.read_csv(data_path, index_col=0)

    model = train(df)

    with open(path_model, 'wb') as file:
        pickle.dump(model, file)


def train(df: pd.DataFrame) -> xgb.XGBRegressor:
    # It's not relevant to train or test without target (Y)
    df = df[df['Price'].notna()]

    zip_code: pd.DataFrame = pd.read_csv(zip_path, index_col=0)
    df_zip_code = mean_and_median_by_locality(df, zip_code)
    df_zip_code.to_csv(mm_zip_path)

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

