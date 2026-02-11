import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import VarianceThreshold
import os
import glob
from src.utils.Print_Helper import MyPrint

def preprocess_data(input_path, output_path, class_column, columns=None, variance_threshold=0.01):

    csv_files = glob.glob(os.path.join(input_path, "*.csv"))
    MyPrint("Preprocessing_Func.py", f"Number of CSV files found: {len(csv_files)}")

    df_list = []
    total_rows = 0

    for file in csv_files:
        temp_df = pd.read_csv(file)
        df_list.append(temp_df)
        total_rows += len(temp_df)

    df = pd.concat(df_list, ignore_index=True)

    if (total_rows > 0):
        MyPrint("Preprocessing_Func.py", "Creating a processed file with " + str(total_rows) + " rows at input path: " + input_path)
    else:
        MyPrint("Preprocessing_Func.py", "Error, no rows found in input path: " + input_path, error=True, line_num=24)
        return

    df = df.rename(columns={class_column: "class"})

    if columns is not None:
        allowed_cols = [c for c in columns if c in df.columns]
        missing = set(columns) - set(allowed_cols)
        if missing:
            MyPrint("Preprocessing_Func.py", f"Warning: columns not found and skipped: {missing}", error=True, line_num=22)
        df = df[allowed_cols]

    df = df.dropna(axis=1, how='all')
    df = df.dropna(thresh=len(df) * 0.8, axis=1)

    # Drop high-cardinality / near-unique columns that pyIDS cannot handle
    high_card_cols = [col for col in df.columns if df[col].nunique() / len(df) > 0.9]
    df = df.drop(columns=high_card_cols)
    MyPrint("Preprocessing_Func.py", f"Dropped high-cardinality columns: {high_card_cols}")

    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].fillna(df[col].mode()[0])
        else:
            df[col] = df[col].fillna(df[col].mean())

    categorical = df.select_dtypes(include=['object']).columns
    for col in categorical:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))

    numeric = df.select_dtypes(include=[np.number]).columns
    scaler = StandardScaler()
    df[numeric] = scaler.fit_transform(df[numeric])

    selector = VarianceThreshold(threshold=variance_threshold)
    reduced = selector.fit_transform(df[numeric])
    kept_columns = numeric[selector.get_support(indices=True)]
    df = df[kept_columns.tolist() + list(categorical)]

    df = df[[c for c in df.columns if c != 'class'] + ['class']] # make the class column the last column
    df.to_csv(output_path, index=False)
    MyPrint("Preprocessing_Func.py",  f"Saved {output_path} | Rows: {df.shape[0]} | Cols: {df.shape[1]}")
    
    #return the array and the names of the column
    kept_columns = [col for col in df.columns if col != "class"]
    return df, kept_columns
