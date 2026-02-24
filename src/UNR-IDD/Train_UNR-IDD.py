import sys
import os
import pandas as pd
from sklearn.model_selection import train_test_split

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))

# Make sure pyIDS is importable
sys.path.insert(0, os.path.join(BASE_DIR, "pyIDS"))
sys.path.append(BASE_DIR)

from src.pyIDS_Functions.Mining_Cars_Func import Mine_Cars
from src.pyIDS_Functions.Training_Func import Train
from src.utils.Print_Helper import MyPrint

# ---- Paths (MAKE SURE THESE MATCH YOUR PREPROCESS OUTPUT NAMES) ----
data_path = os.path.join(BASE_DIR, "data", "processed", "unridd_preprocessed.csv")
cars_path = os.path.join(BASE_DIR, "data", "cars", "UNR-IDD.csv")
rules_out_path = os.path.join(BASE_DIR, "data", "rules", "UNR-IDD_rules.csv")


def UNR_IDD_Train(max_rows=10000, val_fraction=0.2, random_state=42, rule_cutoff=100):
    MyPrint("Train_UNR-IDD", "Beginning to Train UNR-IDD")

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Processed data not found: {data_path}")

    full_df = pd.read_csv(data_path).head(max_rows)
    full_df["class"] = full_df["class"].astype(str)

    # Split for reproducibility / future lambda optimization
    train_df, val_df = train_test_split(
        full_df,
        test_size=val_fraction,
        stratify=full_df["class"],
        random_state=random_state
    )

    MyPrint("Train_UNR-IDD", f"Train rows: {len(train_df)} | Val rows: {len(val_df)}")

    # 1) Mine CARs on the training split
    cars = Mine_Cars(rule_cutoff, train_df, cars_path)

    # 2) Pick a lambda array (simple default)
    # NOTE: This is just a placeholder default; tune later if you want.
    lambda_array = [1, 1, 1, 1, 1, 1, 1, 1]

    # 3) Train pyIDS and save selected rules
    Train("SLS", lambda_array, cars, train_df, rules_out_path)

    MyPrint("Train_UNR-IDD", f"Training complete! Rules saved to: {rules_out_path}")


if __name__ == "__main__":
    UNR_IDD_Train(max_rows=10000)
