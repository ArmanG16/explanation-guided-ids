import sys
import os
import pandas as pd

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))

sys.path.insert(0, os.path.join(BASE_DIR, "pyIDS"))

data_dir = os.path.join(BASE_DIR, "data/processed/beth_preprocessed.csv")
cars_dir = os.path.join(BASE_DIR, "data/cars")
output_path = os.path.join(BASE_DIR, "data/rules/bethdataset_rules.csv")
cars_csv_name = "bethdataset.csv"

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.pyIDS_Functions.Mining_Cars_Func import Mine_Cars
from src.pyIDS_Functions.Training_Func import Train

def Beth_Train(max_rows):

    df = pd.read_csv(data_dir, nrows=max_rows)

    cars = Mine_Cars(50, df, cars_dir + "/" + cars_csv_name)

    lambda_array = [1, 1, 1, 1, 1, 1, 1]

    Train("SLS", lambda_array, cars, df, output_path)

if __name__ == "__main__":
    Beth_Train(1000)