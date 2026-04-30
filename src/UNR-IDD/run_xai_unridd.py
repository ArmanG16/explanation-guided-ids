import pandas as pd
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.Explainable_AI.explain import explain_row

RULES = "data/cars/UNR-IDD.csv"
VAL   = "data/processed/unridd_preprocessed.csv"

df = pd.read_csv(VAL)

result = explain_row(df.iloc[0], RULES, meta=None, top_k=5)
print(result)
