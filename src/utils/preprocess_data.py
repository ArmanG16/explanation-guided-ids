import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import VarianceThreshold

def preprocess_data(input_path, output_path, variance_threshold=0.01):
    # Load dataset
    df = pd.read_csv(input_path)
    print(f"Loaded dataset with {df.shape[0]} rows and {df.shape[1]} columns.")

    # Drop fully empty or mostly-empty columns
    df = df.dropna(axis=1, how='all')              # Drop columns entirely empty
    df = df.dropna(thresh=len(df) * 0.8, axis=1)   # Drop columns with >20% missing values

    # Fill remaining NaNs (numeric -> mean, categorical -> mode)
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].fillna(df[col].mode()[0])
        else:
            df[col] = df[col].fillna(df[col].mean())

    # Encode categorical columns
    categorical = df.select_dtypes(include=['object']).columns
    for col in categorical:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))

    # Scale numeric columns
    numeric = df.select_dtypes(include=[np.number]).columns
    scaler = StandardScaler()
    df[numeric] = scaler.fit_transform(df[numeric])

    # Remove low-variance features
    selector = VarianceThreshold(threshold=variance_threshold)
    reduced = selector.fit_transform(df[numeric])
    kept_columns = numeric[selector.get_support(indices=True)]
    df = df[kept_columns.union(categorical)]  # Keep high-variance numeric + all categorical

    # Final cleanup
    df = df.loc[:, ~df.columns.duplicated()]
    df = df.dropna()  # remove any remaining incomplete rows just in case

    # Save processed dataset
    df.to_csv(output_path, index=False)
    print(f"Saved preprocessed dataset to {output_path} with {df.shape[0]} rows and {df.shape[1]} columns.")
    print(f"Variance threshold: {variance_threshold} | Kept {len(kept_columns)} numeric features.")

    return df
