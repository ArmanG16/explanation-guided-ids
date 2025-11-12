import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import VarianceThreshold
import os
from src.utils.Print_Helper import MyPrint

def preprocess_data(input_path, output_path, class_column, variance_threshold=0.01):
    df = pd.read_csv(input_path)

    if class_column in df.columns:
        df = df.rename(columns={class_column: "class"})
        label_col = df["class"].copy()
        df = df.drop(columns=["class"]) #drop the class column to avoid it during processing
    else:
        MyPrint("Preprocessing_Func.py",  "Error, class name " + class_column + " not found", error=True, line_num=15)
        return

    df = df.dropna(axis=1, how='all')
    df = df.dropna(thresh=len(df) * 0.8, axis=1)
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

    df = df.loc[:, ~df.columns.duplicated()]
    df = df.loc[:, df.nunique() > 1] # removes columns with the same value throughout
    df = df.dropna()
    df["class"] = label_col #add back in the class column
    df = df[[c for c in df.columns if c != 'class'] + ['class']]
    df.to_csv(output_path, index=False)
    MyPrint("Preprocessing_Func.py",  f"Saved {output_path} | Rows: {df.shape[0]} | Cols: {df.shape[1]}")
