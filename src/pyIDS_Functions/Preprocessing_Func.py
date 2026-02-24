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
    safe_name=None,
    malicious_name=None,
    safe_values=None,
    malicious_values=None,
	metadata_output_path=None):

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

    # Ensure class column is named 'class' for pyIDS compatibility and rename class values if specified
    df = df.rename(columns={class_column: "class"})

    if safe_name is not None and malicious_name is not None:

        MyPrint("Preprocessing_Func.py",
                "Renaming class values to: " + safe_name +
                " (safe) and " + malicious_name + " (malicious)")

        if safe_values is None:
            MyPrint("Preprocessing_Func.py",
                    "Error: safe_values must be provided when renaming class values",
                    error=True, line_num=30)
            return

        # If malicious_values is None, treat as empty list
        if malicious_values is None:
            malicious_values = []

        # Check overlap only if malicious_values provided
        overlap = set(safe_values) & set(malicious_values)
        if overlap:
            MyPrint("Preprocessing_Func.py",
                    f"Error: Values cannot appear in both classes: {overlap}",
                    error=True, line_num=35)
            return

        # -------------------------------------------------
        # Case 1: malicious_values provided → strict mapping
        # -------------------------------------------------
        if len(malicious_values) > 0:

            mapping = {}

            for val in safe_values:
                mapping[val] = safe_name

            for val in malicious_values:
                mapping[val] = malicious_name

            unique_vals = set(df["class"].unique())
            unknown_vals = unique_vals - set(mapping.keys())

            if unknown_vals:
                MyPrint("Preprocessing_Func.py",
                        f"Error: Unexpected class values found: {unknown_vals}",
                        error=True, line_num=45)
                return

            df["class"] = df["class"].map(mapping)

        # -------------------------------------------------
        # Case 2: malicious_values empty → everything else malicious
        # -------------------------------------------------
        else:

            df["class"] = np.where(
                df["class"].isin(safe_values),
                safe_name,
                malicious_name
            )

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
