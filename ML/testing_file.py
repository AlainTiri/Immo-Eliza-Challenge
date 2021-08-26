import os
import pandas as pd


data_path = os.path.join(os.path.dirname(__file__), "Data", "database.csv")
zip_path = os.path.join(os.path.dirname(__file__), "Data", "zipcodes.csv")


def count_by_locality(df):
    df_prices = df.groupby("Locality").count()
    s_count = df_prices.Province

    new_df = pd.DataFrame(s_count.rename("count_CP", inplace=True))
    new_df.reset_index(level=0, inplace=True)
    print(new_df)

    for row in new_df.itertuples():
        suite = df[df['Locality'] == row.Locality]
        moyenne = suite.mean()
        moyenne = moyenne['Price']
        print('moyenne:', moyenne)
        df.loc[df["Locality"] == row.Locality, 'moyenne'] = moyenne

        print(row.Locality, row.count_CP, moyenne)

    return df


df: pd.DataFrame = pd.read_csv(data_path, index_col=0)
zip: pd.DataFrame = pd.read_csv(zip_path, index_col=0)

print(df.shape)
counter = count_by_locality(df, zip)
