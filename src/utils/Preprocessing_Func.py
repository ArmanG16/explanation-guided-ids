import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import VarianceThreshold
import os
from src.utils.Print_Helper import MyPrint

def preprocess_data(input_path, output_path, variance_threshold=0.01):
    df = pd.read_csv(input_path)
    label_col = None
    for possible_label in ['Label', 'label', 'Attack', 'Class', 'target']:
        if possible_label in df.columns:
            label_col = possible_label
            break
    if label_col:
        y = df[label_col]
        df = df.drop(columns=[label_col])
    else:
        y = None

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
    df = df.dropna()
    if y is not None:
        df[label_col] = y
    df.to_csv(output_path, index=False)
    MyPrint("Preprocessing_Func.py",  f"Saved {output_path} | Rows: {df.shape[0]} | Cols: {df.shape[1]}")
