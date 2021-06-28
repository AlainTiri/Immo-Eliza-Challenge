import os

# from numba import jit
import pandas as pd
import numpy as np
from typing import Tuple

from Core.datautils import DataStruct

__core_path: str = os.path.dirname(__file__)
__root_path: str = os.path.dirname(__core_path)
__per_site_api_path: str = os.path.join(__root_path, "PerSiteApi")
__data_path: str = os.path.join(__root_path, "Data")
__data_immoweb_path: str = os.path.join(__data_path, "data_immoweb")
__database_file: str = os.path.join(__data_path, "database.csv")
__logic_immo_data: str = os.path.join(__data_path, "logic_immo_clean_data.csv")
__immoweb_data: str = os.path.join(__data_immoweb_path, "datas_immoweb.csv")
__non_url_data_cols: Tuple = (
    "Price",
    "Number of rooms",
    "Area",
    "Fully equipped kitchen",
    "Terrace Area",
    "Garden Area",
    "Surface of the land",
    "Surface area of the plot of land",
    "Number of facades",
    "State of the building",
)
pd.options.display.max_columns = 22


def convert_datatypes(df):
    """
    Function used to convert the database in a version +-30% the weight of the original
    :param df: the original version of the database Dataframe
    :type df: pd.DataFrame
    :return: a lightweight version of the database DataFrame (without dataloss)
    :rtype: pd.DataFrame
    """
    property_subtype = (
        "penthouse",
        "building",
        "studio",
        "duplex",
        "triplex",
        "loft",
        "ground floor",
        "student",
        "investment property",
        "villa",
        "mansion",
        "mixed",
        "apartments row",
        "farmhouse",
        "cottage",
        "floor",
        "town",
        "service flat",
        "manor",
        "castle",
        "pavilion",
    )
    df["Source"] = pd.Categorical(df["Source"])
    # for column in df.columns:
    #     print(column, df[column].unique())

    df["Type of property"] = pd.Categorical(
        df["Type of property"], categories=("apartment", "house"), ordered=True
    )
    df["Type of sale"] = pd.Categorical(
        df["Type of sale"], categories=("regular sale", "public sale")
    )
    df["Subtype of property"] = pd.Categorical(
        df["Subtype of property"], categories=property_subtype
    )
    df["State of the building"] = pd.Categorical(
        df["State of the building"],
        categories=("to renovate", "good", "new"),
        ordered=True,
    )
    df["Province"] = pd.Categorical(
        df["Province"],
        categories=list(sorted(DataStruct.get_zipcode_extended_data()["province"].unique()))
    )
    df["Region"] = pd.Categorical(
        df["Region"],
        categories=list(sorted(DataStruct.get_zipcode_extended_data()["region"].unique()))
    )
    df["Fully equipped kitchen"] = df["Fully equipped kitchen"].astype(np.float16)
    df["Furnished"] = df["Furnished"].astype(np.float16)
    df["Locality"] = df["Locality"].astype(np.int16)
    df["Open fire"] = df["Open fire"].astype(np.float16)
    df["Swimming pool"] = df["Swimming pool"].astype(np.float16)
    df["Garden"] = df["Garden"].astype(np.float16)
    df["Terrace Area"] = df["Terrace Area"].astype(np.float32)
    df["Surface of the land"] = df["Surface of the land"].astype(np.float32)
    df["Surface area of the plot of land"] = df[
        "Surface area of the plot of land"
    ].astype(np.float32)
    df["Garden Area"] = df["Garden Area"].astype(np.float32)
    df["Number of facades"] = df["Number of facades"].astype(np.float16)
    df["Area"] = df["Area"].astype(np.float32)
    df["Terrace"] = df["Terrace"].astype(np.float32)
    df["Number of rooms"] = df["Number of rooms"].astype(np.float16)
    df["Fully equipped kitchen"] = df["Fully equipped kitchen"].astype(np.float16)
    df.reset_index(drop=True, inplace=True)
    return df


def create_database():
    """
    Load the database files and clean, type, store it
    """
    df1: pd.DataFrame = pd.read_csv(__logic_immo_data)
    df1["Locality"] = df1["Zip"]
    df1.drop_duplicates(subset=["Url"], keep="first", inplace=True)
    df1.drop(columns=["Id", "Zip"], inplace=True)
    for column in df1.columns:
        df1[column].where(df1[column] != -1, inplace=True)
        if column == "State of the building":
            df1[column].where(df1[column] != "none", inplace=True)

    df1: pd.DataFrame = df1[
        np.logical_and(
            df1["Type of sale"] != "viager", df1["Type of sale"] != "life_annuity"
        )
    ]
    df1.dropna(subset=__non_url_data_cols, how="all", inplace=True)
    # print(df1.head())
    # print(df1.info())

    df2: pd.DataFrame = pd.read_csv(__immoweb_data, index_col=0)
    df2: pd.DataFrame = df2[
        np.logical_and(
            df2["Type of sale"] != "viager", df2["Type of sale"] != "life_annuity"
        )
    ]
    df2["Locality"] = df2["Zip"]
    df2["Url"] = df2["Url"].str.split('?').str[0]
    df2.drop_duplicates(subset=["Url"], keep="first", inplace=True)
    df2.drop(columns=["Zip"], inplace=True)
    df2.dropna(subset=__non_url_data_cols, how="all")





    filters = list()
    for column in ("Furnished", "Open fire", "Garden", "Swimming pool"):
        df2[column] = df2[column].astype(str)
        df2.loc[df2[column] == "False", column] = 0
        df2.loc[df2[column] == "True", column] = 1

    for column, dtype in zip(df2.columns, df2.dtypes):
        if dtype == "object":
            df2[column] = df2[column].astype(str).str.lower()

    # print(df2.head())
    # print(df2.info())

    df: pd.DataFrame = pd.concat([df1, df2], axis=0, ignore_index=True)
    df: pd.DataFrame = df[df["Type of sale"] != "life_annuity"]
    # print(df.info())
    df["Source"] = pd.Categorical(df["Source"])
    df.loc[df["Type of property"] == "maison", "Type of property"] = "house"
    df.loc[df["Type of property"] == "appartment", "Type of property"] = "apartment"
    df: pd.DataFrame = df[df["Type of property"].str.find("_group") == -1]
    # print(df["Type of property"].unique())
    df["Subtype of property"] = df["Subtype of property"].str.replace("_house", "")
    df.loc[df["Subtype of property"] == "étage", "Subtype of property"] = "floor"
    df.loc[df["Subtype of property"] == "immeuble", "Subtype of property"] = "building"
    df.loc[
        df["Subtype of property"] == "mixed_use_building", "Subtype of property"
    ] = "building"
    df.loc[
        df["Subtype of property"] == "apartment_block", "Subtype of property"
    ] = "building"
    df.loc[
        df["Subtype of property"] == "rez-de-chaussée", "Subtype of property"
    ] = "ground floor"
    df.loc[
        df["Subtype of property"] == "ground_floor", "Subtype of property"
    ] = "ground floor"
    df.loc[df["Subtype of property"] == "flat_studio", "Subtype of property"] = "studio"
    df.loc[df["Subtype of property"] == "château", "Subtype of property"] = "castle"
    df.loc[df["Subtype of property"] == "bungalow", "Subtype of property"] = "chalet"
    df.loc[df["Subtype of property"] == "maître", "Subtype of property"] = "mansion"
    df.loc[
        df["Subtype of property"] == "rangée", "Subtype of property"
    ] = "apartments row"
    df.loc[
        df["Subtype of property"] == "façades", "Subtype of property"
    ] = "apartments row"
    df.loc[df["Subtype of property"] == "ferme", "Subtype of property"] = "farmhouse"
    df.loc[df["Subtype of property"] == "mixte", "Subtype of property"] = "mixed"
    df.loc[
        df["Subtype of property"] == "country_cottage", "Subtype of property"
    ] = "cottage"
    df.loc[
        df["Subtype of property"] == "rapport", "Subtype of property"
    ] = "investment property"
    df.loc[df["Subtype of property"] == "kot", "Subtype of property"] = "student"
    df["Subtype of property"] = df["Subtype of property"].str.replace("_", " ")
    df.loc[df["Type of sale"] == "vente", "Type of sale"] = "regular sale"
    df.loc[df["Type of sale"] == "buy_regular", "Type of sale"] = "regular sale"
    df["Type of sale"] = df["Type of sale"].str.replace("_", " ")
    df["Fully equipped kitchen"] = df["Fully equipped kitchen"].str.replace("usa_", "")
    df.loc[df["Fully equipped kitchen"] == "nan", "Fully equipped kitchen"] = None
    df.loc[
        df["Fully equipped kitchen"] == "not_installed", "Fully equipped kitchen"
    ] = 0
    df.loc[df["Fully equipped kitchen"] == "uninstalled", "Fully equipped kitchen"] = 0
    df.loc[
        df["Fully equipped kitchen"] == "semi_equipped", "Fully equipped kitchen"
    ] = 0
    df.loc[df["Fully equipped kitchen"] == "installed", "Fully equipped kitchen"] = 0
    df.loc[
        df["Fully equipped kitchen"] == "hyper_equipped", "Fully equipped kitchen"
    ] = 1

    df.loc[
        df["State of the building"] == "to_renovate", "State of the building"
    ] = "to renovate"
    df.loc[
        df["State of the building"] == "to_be_done_up", "State of the building"
    ] = "to renovate"
    df.loc[
        df["State of the building"] == "to_restore", "State of the building"
    ] = "to renovate"
    df.loc[
        df["State of the building"] == "just_renovated", "State of the building"
    ] = "good"
    df.loc[
        df["State of the building"] == "to_renovate", "State of the building"
    ] = "good"
    df.loc[
        df["State of the building"] == "just_renovated", "State of the building"
    ] = "good"
    df.loc[df["State of the building"] == "as_new", "State of the building"] = "good"
    df.loc[df["Terrace"] == "true", "Terrace"] = 1
    # df = df[np.logical_or(df["Surface of the land"] >= 0, df["Surface of the land"] == np.nan)]
    df["Surface of the land"].where(df["Surface of the land"] >=0, inplace=True)
    df["Surface area of the plot of land"] = df["Surface of the land"]
    df = df.loc[np.in1d(df["Type of property"],("commercial", "garage", "office", "land"), invert=True), :]
    df.drop_duplicates(subset=df.columns[1:], inplace=True)


    # Data values override when "is not mentionned when false"
    subset: Tuple = ("Swimming pool","Open fire","Garden","Terrace", "Fully equipped kitchen")
    df.loc[:, subset] = df.loc[:, subset].astype(float).fillna(0)


    df["Province"] = df["Locality"].map(DataStruct.get_province)
    df["Region"] = df["Locality"].map(DataStruct.get_region)
    df = df.loc[np.logical_and(df["Province"].notna(), df["Region"].notna()), :]



    df: pd.DataFrame = convert_datatypes(df)

    # print(df["Source"].unique())
    # print(df.info())
    df.to_csv(__database_file)


def load_database(lightweight=True):
    """
    Load the final database file in a lightweight type
    :param bool lightweight: force the data conversion into a lightweight version of the database
    """
    df: pd.DataFrame = pd.read_csv(__database_file, index_col=0)
    if lightweight:
        df: pd.DataFrame = convert_datatypes(df)
    return df

if __name__ == "__main__":
    create_database()
    df = load_database()
    print(df.info())
    print(df.describe())
