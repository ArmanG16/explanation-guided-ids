import pandas as pd
from src.Explainable_AI.explain import explain_row

DATA_PATH = "data/processed/usnw_dataset/usnw_preprocessed.csv"
RULES_PATH = "data/processed/usnw_dataset/usnw_cars.txt"

df = pd.read_csv(DATA_PATH)

# optional: remove class column from the row passed to explainer
# if the rule antecedents do not include class, this is cleaner
if "class" in df.columns:
    feature_df = df.drop(columns=["class"])
else:
    feature_df = df

row_idx = 0
row = feature_df.iloc[row_idx]

results = explain_row(
    row=row,
    rules_csv_path=RULES_PATH,
    meta=None,
    top_k=3
)

print(f"\nTop explanations for row {row_idx}:\n")

if not results:
    print("No rules fired for this row.")
else:
    for i, r in enumerate(results, 1):
        print(f"--- Explanation {i} ---")
        print("Rule Index:", r["Rule_Index"])
        print("Readable Rule:", r["Readable_Rule"])
        print("Predicted Class:", r["Predicted_Class"])
        print(r["Explanation"])
        print()