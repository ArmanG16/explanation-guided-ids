import sys
import os
import pandas as pd
from sklearn.model_selection import train_test_split

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))

sys.path.insert(0, os.path.join(BASE_DIR, "pyIDS"))

data_dir = os.path.join(BASE_DIR, "data/processed/UNR-IDD")
cars_dir = os.path.join(BASE_DIR, "data/cars")
output_path = os.path.join(BASE_DIR, "data/rules/UNR-IDD_rules.csv")
lambdas_path = os.path.join(BASE_DIR, "data/lambdas/UNR-IDD_best_lambdas.csv")
data_dir_name = "UNR-IDD_preprocessed.csv"
cars_csv_name = "UNR-IDD.csv"

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.utils.Mining_Cars_Func import Mine_Cars
from src.utils.Training_Func import Train
from src.utils.Optimizing_Lambdas import Optimize_Lambdas
from src.utils.Print_Helper import MyPrint

def UNR_IDD_Train(max_rows, val_fraction = 0.2, random_state=42):
    MyPrint("Train_UNR-IDD", "Beginning to Train UNR-IDD")

    full_df = pd.read_csv(os.path.join(data_dir, data_dir_name))
    full_df = full_df.head(max_rows)
    full_df["class"] = full_df["class"].astype(str)

    train_df, val_df = train_test_split(
        full_df,
        test_size=val_fraction,
        stratify=full_df["class"],
        random_state=random_state
    )

    cars = Mine_Cars(100, train_df, cars_dir + "/" + cars_csv_name)

    lambda_array = Optimize_Lambdas(
        algorithm="SLS",
        cars=cars,
        df=val_df,
        output_path=lambdas_path,
        precision=50,
        iterations=3
    )

    Train("SLS", lambda_array, cars, train_df, output_path)

    MyPrint("Train_UNR-IDD", "Training complete!")

if __name__ == "__main__":
    UNR_IDD_Train(10000)