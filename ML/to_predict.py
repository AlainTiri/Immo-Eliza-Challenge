
from preprocess import preprocess_data
import pandas as pd
import pickle


path_model: str = "Eliza_XGB_Model.pkl"

def load_model():
    # Load the Model back from file
    with open(path_model, 'rb') as file:
        model = pickle.load(file)
    return model


def to_predict(to_predict: dict):
    model = load_model()

    df_to_predict = pd.DataFrame.from_dict(to_predict)

    df_to_predict['Price'] = 0
    df_to_predict["Type of sale"] = "predict"
    df_to_predict["Url"] = "predict"

    with open("df_empty_pre.pkl", 'rb') as file:
        df_empty = pickle.load(file)

    df_to_predict = df_empty.append(df_to_predict)

    df_to_predict = preprocess_data(df_to_predict)
    print(df_to_predict.T)

    with open("df_empty_post.pkl", 'rb') as file:
        df_empty = pickle.load(file)

    df_to_predict = df_empty.append(df_to_predict)
    df_to_predict = df_to_predict.drop("Price", axis=1)

    df_to_predict = df_to_predict.fillna(0)

    prediction = model.predict(df_to_predict)
    print(prediction, '€')

