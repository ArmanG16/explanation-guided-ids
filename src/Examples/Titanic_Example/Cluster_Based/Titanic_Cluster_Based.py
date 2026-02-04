from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans

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


# ----- basic split -----
train_df, val_df = train_test_split(
    df,
    test_size=0.2,
    stratify=df["surv"],
    random_state=42
)

# Make a copy of train_df for clustering
cluster_df = train_df.copy()

# One-hot encode the categorical columns using the correct CSV headers
cluster_df = pd.get_dummies(
    cluster_df,
    columns=["Passenger_Cat", "Age_Cat", "Gender"],  # match your CSV headers
    drop_first=False
)


# ----- cluster -----
kmeans = KMeans(n_clusters=3, random_state=42)
train_df["cluster"] = kmeans.fit_predict(cluster_df)
val_df["cluster"] = kmeans.predict(
    pd.get_dummies(val_df, columns=["passenger_class", "age", "sex"])
        .reindex(columns=cluster_df.columns, fill_value=0)
)

BASE_OUTPUT_DIR = os.path.join(BASE_DIR, "src/Examples/Titanic_Example")

for cluster_id in sorted(train_df["cluster"].unique()):
    MyPrint("Cluster", f"Training cluster {cluster_id}")

    cluster_train_df = train_df[train_df["cluster"] == cluster_id].drop(columns="cluster")
    cluster_val_df   = val_df[val_df["cluster"] == cluster_id].drop(columns="cluster")

    cars_path = os.path.join(
        BASE_OUTPUT_DIR,
        f"cluster_{cluster_id}_cars.csv"
    )

    output_path = os.path.join(
        BASE_OUTPUT_DIR,
        f"cluster_{cluster_id}_rules.csv"
    )

    lambdas_path = os.path.join(
        BASE_OUTPUT_DIR,
        f"cluster_{cluster_id}_lambdas.csv"
    )

    Run_pyIDS(
        algorithm="SLS",
        train_df=cluster_train_df,
        val_df=None,
        cars_path=cars_path,
        output_path=output_path,
        lambdas_path=None
    )