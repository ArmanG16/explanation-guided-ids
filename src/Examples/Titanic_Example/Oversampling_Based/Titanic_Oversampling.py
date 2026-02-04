from sklearn.model_selection import train_test_split
from sklearn.utils import resample

import sys
import os
import pandas as pd
import glob

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))

sys.path.insert(0, os.path.join(BASE_DIR, "pyIDS"))
sys.path.append(BASE_DIR)

from src.utils.Print_Helper import MyPrint
from src.pyIDS_Functions.Run_pyIDS import Run_pyIDS

data_dir = os.path.join(BASE_DIR, "pyIDS/data/")

titanic_file = os.path.join(data_dir, "titanic.csv")

df = pd.read_csv(titanic_file)


# ----- oversample minority class -----
df_majority = df[df["Died"] == df["Died"].mode()[0]]  # majority class
df_minority = df[df["Died"] != df["Died"].mode()[0]]  # minority class

df_minority_upsampled = resample(
    df_minority,
    replace=True,                  # sample with replacement
    n_samples=len(df_majority),    # match majority class size
    random_state=42
)

df_balanced = pd.concat([df_majority, df_minority_upsampled])
df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)

# ----- basic split -----
train_df, val_df = train_test_split(
    df_balanced,
    test_size=0.2,
    stratify=df_balanced["Died"],
    random_state=42
)

BASE_OUTPUT_DIR = os.path.join(BASE_DIR, "src/Examples/Titanic_Example/Oversampling_Based")

# We just train on the full balanced dataset (no clusters)
MyPrint("Oversample", f"Training on oversampled Titanic data: {len(train_df)} train, {len(val_df)} val")

cars_path = os.path.join(BASE_OUTPUT_DIR, "oversample_cars.csv")
output_path = os.path.join(BASE_OUTPUT_DIR, "oversample_rules.csv")
lambdas_path = os.path.join(BASE_OUTPUT_DIR, "oversample_lambdas.csv")

Run_pyIDS(
    algorithm="SLS",
    train_df=train_df,
    val_df=None,
    cars_path=cars_path,
    output_path=output_path,
    lambdas_path=None
)
