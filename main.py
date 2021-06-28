import os
import sys
sys.path.extend(os.path.join(os.path.dirname(__file__), "ML"))
sys.path.extend(os.path.join(os.path.dirname(__file__), "scrap"))
import ML
import scrap


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')

    ask = input("Voulez-vous créer un nouveau modèle : (y/n)")
    if ask.lower().startswith("y"):
        ML.create_model.run()
    else:
        ask = input("Vous êtes sur ? (y/n)")
        ML.create_model.run()


# See PyCharm help at https://www.jetbrains.com/help/pycharm/
