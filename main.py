import os
import sys
sys.path.extend(os.path.join(os.path.dirname(__file__), "ML"))
sys.path.extend(os.path.join(os.path.dirname(__file__), "scrap"))
import ML
import scrap
# todo : re write srapping for to take in case the new captcha on immoweb


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')

    ask = input("Voulez-vous créer un nouveau modèle : (y/n)")
    if ask.lower().startswith("y"):
        new_model = ML.create_model.ModelImmoweb()
        new_model.fit()

    ask = input("Voulez-vous tester le modèle : (y/n)")
    if ask.lower().startswith("y"):

        # Han
        to_predict = {"Number of rooms": [2], "Fully equipped kitchen": [1], "Terrace": [1], "Garden": [0],
                      "Swimming pool": [0], "Locality": [1160], "Area": [85], "State of the building": ["good"],
                      "Type of property": ["apartment"]}
        ML.to_predict.to_predict(to_predict)

        # Olivier
        to_predict = {"Number of rooms": [1], "Fully equipped kitchen": [1], "Terrace": [0], "Garden": [0],
                      "Locality": [1000], "Area": [70], "State of the building": ["good"],
                      "Type of property": ["apartment"], "Number of facades": [2]}
        ML.to_predict.to_predict(to_predict)

        # WSL Rue de la Cambre
        to_predict = {"Number of rooms": [2], "Fully equipped kitchen": [1], "Terrace": [1], "Terrace Area": [30],
                      "Garden": [0], "Locality": [1200], "Area": [120], "State of the building": ["good"],
                      "Type of property": ["apartment"], "Number of facades": [2]}
        ML.to_predict.to_predict(to_predict)


        # Maison blanche
        to_predict = {"Number of rooms": [5], "Fully equipped kitchen": [1], "Terrace": [1], "Terrace Area": [8],
                      "Garden": [1], "Locality": [1341], "Area": [200], "State of the building": ["good"],
                      "Surface of the land": [1500], "Type of property": ["house"], "Number of facades": [4]}
        ML.to_predict.to_predict(to_predict)

    ask = input("Voulez-vous afficher les entêtes : (y/n)")
    if ask.lower().startswith("y"):
        ML.create_model.print_entetes()
