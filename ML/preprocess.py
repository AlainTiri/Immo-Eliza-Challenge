from lat_long import get_lat_from_zip, get_long_from_zip
import pandas as pd
import os

mm_zip_path = os.path.join(os.path.dirname(__file__), "Data", "mean_median_zipcodes.csv")


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:

    # Nettoyer les type et sub type null
    df.loc[df["Type of sale"] == "public sale", 'Type of sale'] = None
    df = df[df['Type of sale'].notna()]
    df = df.drop(axis=1, labels="Type of sale")

    df.loc[df["Subtype of property"] == "villa", 'Type of property'] = "house"
    df.loc[df["Subtype of property"] == "loft", 'Type of property'] = "apartment"
    df.loc[df["Subtype of property"] == "mansion", 'Type of property'] = "house"
    df.loc[df["Subtype of property"] == "penthouse", 'Type of property'] = "apartment"
    df.loc[df["Subtype of property"] == "duplex", 'Type of property'] = "apartment"
    df.loc[df["Subtype of property"] == "investment property", 'Type of property'] = "house"

    df.loc[df["Url"].str.contains("immeuble-de-rapport-a-vendre"), 'Type of property'] = "house"
    df.loc[df["Url"].str.contains("immeuble-de-rapport-a-vendre"), 'Subtype of property'] = "investment property"

    df.loc[df["Number of rooms"] == 0, 'Number of rooms'] = None
    df.loc[((df["Number of rooms"].isnull()) & (df["Subtype of property"] == "studio")), 'Number of rooms'] = 0
    df.loc[((df["Number of rooms"].isnull()) & (df["Subtype of property"] == "student")), 'Number of rooms'] = 0
    df.loc[((df["Number of facades"].isnull()) & (df["Type of property"] == "apartment")), 'Number of facades'] = 2
    df.loc[(df["Garden Area"].isnull()) & (df["Garden"] == 0), "Garden Area"] = 0
    df.loc[((df["Subtype of property"].isnull()) & (
            df["Type of property"] == "apartment")), "Subtype of property"] = "apartment"
    df.loc[((df["Subtype of property"].isnull()) & (
            df["Type of property"] == "house")), "Subtype of property"] = "house"

    df.loc[df["Terrace"] == 0, 'Terrace Area'] = 0
    df.loc[(df["Swimming pool"].isnull()), "Swimming pool"] = 0
    df.loc[(df["Furnished"].isnull()), "Furnished"] = 0
    df.loc[(df["Open fire"].isnull()), "Open fire"] = 0

    df.loc[df["Garden"] == 0, 'Garden Area'] = 0

    df = df.drop(axis=1, labels="Url")
    df = df.drop(axis=1, labels="Source")
    df = df.drop(axis=1, labels="Surface area of the plot of land")
    df = df.drop(axis=1, labels="Region")
    df = df.drop(axis=1, labels="Province")

    df = df.sort_values("Price", ascending=False)
    df = df[df['Price'] < 30000000]

    # Add mean and median prices by locality
    mm_zip = pd.read_csv(mm_zip_path, index_col=0)
    for row in mm_zip.itertuples():
        df.loc[df["Locality"] == row.zipcode, 'moyenne'] = row.moyenne
        df.loc[df["Locality"] == row.zipcode, 'mediane'] = row.mediane

    to_replace = {None: "good"}
    df["State of the building"] = df["State of the building"].replace(to_replace)

    df["Locality"] = df["Locality"].astype(str)
    df['lat'] = df["Locality"].apply(get_lat_from_zip)
    df['long'] = df["Locality"].apply(get_long_from_zip)
    df = df[df["lat"] != 0]
    df = df.drop(columns='Locality')
    df = pd.get_dummies(df, drop_first=False)

    return df
