
import preprocess
import pandas as pd
import pickle


path_model = "Eliza_XGB_Model.pkl"

def load_model():
    # Load the Model back from file
    with open(path_model, 'rb') as file:
        model = pickle.load(file)
    return model


def to_predict(to_predict: dict):
    model = load_model()

    df_empty = pd.DataFrame()

    df_to_predict = pd.DataFrame.from_dict(to_predict)
    df_to_predict['Price'] = 0
    df_to_predict["Type of sale"] = "predict"
    df_to_predict["Url"] = "predict"
    df_to_predict = df_empty.append(df_to_predict)

    df_to_predict = preprocess(df_to_predict)

    df_empty_long = df.iloc[0:0]
    df_to_predict = df_empty_long.append(df_to_predict)

    df_to_predict2 = df_to_predict.drop("Price", axis=1)
    df_to_predict2.shape
    prediction = model.predict(df_to_predict2)
    print(prediction, '€')


