import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

from src.pyIDS_Functions.Run_pyIDS import Run_pyIDS
from src.utils.Print_Helper import MyPrint


def Run_USNW_pyIDS():
    input_csv = os.path.join(
        BASE_DIR, "data", "processed", "usnw_dataset", "usnw_preprocessed.csv"
    )
    cars_path = os.path.join(
        BASE_DIR, "data", "processed", "usnw_dataset", "usnw_cars.txt"
    )
    output_path = os.path.join(
        BASE_DIR, "data", "processed", "usnw_dataset", "usnw_pyids_output.txt"
    )
    lambdas_path = os.path.join(
        BASE_DIR, "data", "processed", "usnw_dataset", "usnw_lambdas.txt"
    )

    LABEL_COL = "class"

    MyPrint("Run_USNW_pyIDS", f"Input CSV: {input_csv}")

    df = pd.read_csv(input_csv)

    # sample dataset to avoid OOM during rule mining
    df = df.sample(n=20000, random_state=42)

    if LABEL_COL not in df.columns:
        raise ValueError(f"Expected '{LABEL_COL}' column in preprocessed UNSW dataset.")

    # Drop columns that should not be used as predictors
    drop_cols = [col for col in ["id", "attack_cat"] if col in df.columns]
    if drop_cols:
        MyPrint("Run_USNW_pyIDS", f"Dropping columns: {drop_cols}")
        df = df.drop(columns=drop_cols)

    MyPrint("Run_USNW_pyIDS", f"Dataset shape after drops: {df.shape}")
    MyPrint("Run_USNW_pyIDS", f"Class distribution:\n{df[LABEL_COL].value_counts()}")

    train_df, val_df = train_test_split(
        df,
        test_size=0.2,
        random_state=42,
        stratify=df[LABEL_COL]
    )

    MyPrint("Run_USNW_pyIDS", f"Training rows: {len(train_df)}")
    MyPrint("Run_USNW_pyIDS", f"Validation rows: {len(val_df)}")

    Run_pyIDS(
        algorithm="SLS",
        train_df=train_df,
        cars_path=cars_path,
        output_path=output_path,
        val_df=val_df,
        lambdas_path=lambdas_path
    )

    MyPrint("Run_USNW_pyIDS", "UNSW PyIDS run complete.")


if __name__ == "__main__":
    Run_USNW_pyIDS()