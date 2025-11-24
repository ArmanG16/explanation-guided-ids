import sys
import os

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))

sys.path.insert(0, os.path.join(BASE_DIR, "pyIDS"))

data_dir = os.path.join(BASE_DIR, "data/processed/UNR-IDD")
cars_dir = os.path.join(BASE_DIR, "data/cars")
output_path = os.path.join(BASE_DIR, "data/rules/UNR-IDD_rules.csv")
cars_csv_name = "UNR-IDD.csv"

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.utils.Mining_Cars_Func import Mine_Cars
from src.utils.Training_Func import Train

def UNR_IDD_Train(max_rows):

    cars = Mine_Cars(max_rows, 50, data_dir, cars_dir + "/" + cars_csv_name)

    lambda_array = [1, 1, 1, 1, 1, 1, 1]

    Train("SLS", lambda_array, cars, max_rows, data_dir, output_path)

if __name__ == "__main__":
    UNR_IDD_Train(10000)