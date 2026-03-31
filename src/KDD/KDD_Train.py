import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))

data_dir = os.path.join(BASE_DIR, "data/processed/KDD_preprocessed.csv")
cars_dir = os.path.join(BASE_DIR, "data/cars/KDD.csv")
output_path = os.path.join(BASE_DIR, "data/rules/KDD_rules.csv")
lambdas_path = os.path.join(BASE_DIR, "data/lambdas/KDD_best_lambdas.csv")

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.pyIDS_Functions.Run_pyIDS import Run_pyIDS

def KDD_Train(val_fraction = 0.2, random_state = 42):
    full_df = pd.read_csv(data_dir)
    full_df["class"] = full_df["class"].astype(str)

    train_df, val_df = train_test_split(
        full_df,
        test_size=val_fraction,
        stratify=full_df["class"],
        random_state=random_state
    )

    Run_pyIDS(algorithm="SLS", train_df=train_df, val_df=val_df, lambdas_path=lambdas_path, cars_path=cars_dir, output_path=output_path)
    


if __name__ == "__main__":
    KDD_Train()