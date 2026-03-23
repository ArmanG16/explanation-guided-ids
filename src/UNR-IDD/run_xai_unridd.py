import pandas as pd
from src.Explainable_AI.explain import explain_row

RULES = "data/cars/UNR-IDD.csv"
VAL   = "data/processed/unridd_preprocessed.csv"

df = pd.read_csv(VAL)

result = explain_row(df.iloc[0], RULES, meta=None, top_k=5)
print(result)
