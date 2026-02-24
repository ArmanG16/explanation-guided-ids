import os
import glob
import json

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import VarianceThreshold

from src.utils.Print_Helper import MyPrint


def preprocess_data(
    input_path,
    output_path,
    class_column,
    columns=None,
    variance_threshold=0.01,
    metadata_output_path=None,   # NEW: save mappings/scaler/columns for XAI rule translation
):
    """
    Loads all CSVs in input_path, preprocesses them for PyIDS, saves a single processed CSV to output_path,
    and (optionally) saves preprocessing metadata to metadata_output_path for XAI rule translation.

    Metadata saved:
      - kept_columns (final feature columns)
      - dropped_high_cardinality
      - variance_threshold
      - label_encoders (per categorical feature: int_code -> original category)
      - scaler (per numeric feature: mean/scale, so you can inverse-transform rule thresholds)
      - class_column info
    """

    csv_files = glob.glob(os.path.join(input_path, "*.csv"))
    MyPrint("Preprocessing_Func.py", f"Number of CSV files found: {len(csv_files)}")

    df_list = []
    total_rows = 0

    for file in csv_files:
        temp_df = pd.read_csv(file)
        df_list.append(temp_df)
        total_rows += len(temp_df)

    if total_rows == 0:
        MyPrint(
            "Preprocessing_Func.py",
            "Error, no rows found in input path: " + input_path,
            error=True,
            line_num=24,
        )
        return

    df = pd.concat(df_list, ignore_index=True)

    MyPrint(
        "Preprocessing_Func.py",
        "Creating a processed file with " + str(total_rows) + " rows at input path: " + input_path,
    )

    # --- class column handling ---
    if class_column in df.columns:
        df = df.rename(columns={class_column: "class"})
        label_col = df["class"].copy()
        df = df.drop(columns=["class"])  # drop during processing
    else:
        MyPrint(
            "Preprocessing_Func.py",
            "Error, class name " + class_column + " not found",
            error=True,
            line_num=16,
        )
        return

    # Optional column whitelist
    if columns is not None:
        allowed_cols = [c for c in columns if c in df.columns]
        missing = set(columns) - set(allowed_cols)
        if missing:
            MyPrint(
                "Preprocessing_Func.py",
                f"Warning: columns not found and skipped: {missing}",
                error=True,
                line_num=22,
            )
        df = df[allowed_cols]

    # Drop columns with too many missing values
    df = df.dropna(axis=1, how="all")
    df = df.dropna(thresh=len(df) * 0.8, axis=1)

    # Drop high-cardinality / near-unique columns that pyIDS cannot handle well
    high_card_cols = [col for col in df.columns if df[col].nunique(dropna=True) / max(len(df), 1) > 0.9]
    df = df.drop(columns=high_card_cols, errors="ignore")
    MyPrint("Preprocessing_Func.py", f"Dropped high-cardinality columns: {high_card_cols}")

    # Fill missing values
    for col in df.columns:
        if df[col].dtype == "object":
            # if mode is empty (all NaN), fallback to empty string
            mode_vals = df[col].mode(dropna=True)
            fill_val = mode_vals.iloc[0] if len(mode_vals) > 0 else ""
            df[col] = df[col].fillna(fill_val)
        else:
            df[col] = df[col].fillna(df[col].mean())

    # --- Encode categorical features (SAVE mapping for XAI) ---
    label_encoders = {}  # {col: {int_code: original_value}}
    categorical = df.select_dtypes(include=["object"]).columns

    for col in categorical:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = {int(i): cls for i, cls in enumerate(le.classes_)}

    # --- Scale numeric features (SAVE scaler params for XAI inverse-transform) ---
    numeric = df.select_dtypes(include=[np.number]).columns
    scaler = StandardScaler()
    df[numeric] = scaler.fit_transform(df[numeric])

    scaler_params = {
        "numeric_columns": list(numeric),
        "mean": {col: float(m) for col, m in zip(numeric, scaler.mean_)},
        "scale": {col: float(s) for col, s in zip(numeric, scaler.scale_)},
    }

    # --- Variance threshold on numeric columns ---
    selector = VarianceThreshold(threshold=variance_threshold)
    reduced = selector.fit_transform(df[numeric])
    kept_numeric = numeric[selector.get_support(indices=True)]

    # Keep: variance-filtered numeric + all (encoded) categorical
    df = df[kept_numeric.tolist() + list(categorical)]

    # Cleanup: duplicated cols, constant cols, and any remaining NaNs
    df = df.loc[:, ~df.columns.duplicated()]
    df = df.loc[:, df.nunique() > 1]  # removes columns with same value throughout
    df = df.dropna()

    # Reattach class column at end
    df["class"] = label_col
    df = df[[c for c in df.columns if c != "class"] + ["class"]]

    # Save processed dataset
    df.to_csv(output_path, index=False)
    MyPrint("Preprocessing_Func.py", f"Saved {output_path} | Rows: {df.shape[0]} | Cols: {df.shape[1]}")

    # Return the dataframe and the final kept feature columns
    kept_columns = [col for col in df.columns if col != "class"]

    # --- NEW: Save metadata for XAI rule translation ---
    if metadata_output_path is not None:
        meta = {
            "class_column": "class",
            "original_class_column": class_column,
            "kept_columns": kept_columns,
            "dropped_high_cardinality": high_card_cols,
            "variance_threshold": variance_threshold,
            "label_encoders": label_encoders,
            "scaler": scaler_params,
        }
        os.makedirs(os.path.dirname(metadata_output_path), exist_ok=True)
        with open(metadata_output_path, "w") as f:
            json.dump(meta, f, indent=2)
        MyPrint("Preprocessing_Func.py", f"Saved preprocessing metadata to {metadata_output_path}")

    return df, kept_columns
