import os
import pandas as pd


data_path = os.path.join(os.path.dirname(__file__), "Data", "database.csv")
zip_path = os.path.join(os.path.dirname(__file__), "Data", "zipcodes.csv")


def zip_trunc(zip_code: int) -> int:
    rest = zip_code % 10
    zip_code = zip_code - rest
    return zip_code


def count_by_locality(df, df_zip_code):
    df_zip_code["zip_trunck"] = df_zip_code.zipcode.apply(zip_trunc)
    df_zip_code["count_CP"] = 0
    print(df_zip_code.head())

    df_prices = df.groupby("Locality").count()
    s_count = df_prices.Province

    new_df = pd.DataFrame(s_count.rename("count_CP", inplace=True))
    new_df.reset_index(level=0, inplace=True)
    df["zip_trunck"] = df["Locality"].apply(zip_trunc)
    print(new_df)

    for row in new_df.itertuples():
        df_zip_code.loc[df_zip_code["zipcode"] == row.Locality, 'count_CP'] = row.count_CP
        select_zip = df[df['Locality'] == row.Locality]
        moyenne = select_zip.mean()
        moyenne = moyenne['Price']
        new_df.loc[new_df["Locality"] == row.Locality, 'moyenne'] = moyenne
        df_zip_code.loc[df_zip_code["zip_trunck"] == row.Locality, 'moyenne'] = moyenne

    for row in df_zip_code.itertuples():
        if row.count_CP < 4 :
            select_zip = df[df['zip_trunck'] == row.zip_trunck]
            moyenne = select_zip.mean()
            moyenne = moyenne['Price']
            new_df.loc[new_df["Locality"] == row.zipcode, 'moyenne'] = moyenne
            df_zip_code.loc[df_zip_code["zipcode"] == row.zipcode, 'moyenne'] = moyenne

    return df_zip_code


df: pd.DataFrame = pd.read_csv(data_path, index_col=0)
zip_code: pd.DataFrame = pd.read_csv(zip_path, index_col=0)

print(zip_code.columns)
counter = count_by_locality(df, zip_code)
print(counter.head(100))
