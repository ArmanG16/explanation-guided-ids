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
from src.pyIDS_Functions.Optimizing_Lambdas import Optimize_Lambdas

# ---- Paths (MAKE SURE THESE MATCH YOUR PREPROCESS OUTPUT NAMES) ----
data_path = os.path.join(BASE_DIR, "data", "processed", "unridd_preprocessed.csv")
cars_path = os.path.join(BASE_DIR, "data", "cars", "UNR-IDD.csv")
rules_out_path = os.path.join(BASE_DIR, "data", "rules", "UNR-IDD_rules.csv")

def UNR_IDD_Train(
    max_rows=10000,
    val_fraction=0.2,
    random_state=42,
    num_cars=100):

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
    
    cars = Mine_Cars(num_cars, train_df, cars_path)
    MyPrint("Train_UNR-IDD", f"Train rows: {len(train_df)} | Val rows: {len(val_df)}")

    lambda_array = Optimize_Lambdas(
        algorithm="SLS",
        cars=cars,
        df=val_df,
        individual_precision=50,
        individiual_iterations=3,
        precision=50,
        iterations=1,
        grid_step=200,
        search_type="coordinate"
    )

    # 3) Train pyIDS and save selected rules
    Train("SLS", lambda_array, cars, train_df, rules_out_path)

    MyPrint("Train_UNR-IDD", f"Training complete! Rules saved to: {rules_out_path}")


if __name__ == "__main__":
    UNR_IDD_Train(max_rows=10000)
