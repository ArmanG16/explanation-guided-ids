import json
import pandas as pd
from src.Explainable_AI.explain import explain_row

DATA_PATH = "data/processed/usnw_dataset/usnw_preprocessed.csv"
RULES_PATH = "data/processed/usnw_dataset/usnw_cars.txt"

OUT_JSON = "data/processed/usnw_dataset/usnw_xai_results.json"
OUT_CSV = "data/processed/usnw_dataset/usnw_xai_summary.csv"

TOP_K = 3
MAX_ROWS = 5000   # set to an int like 5000 for testing, or keep None for all rows


def main():
    print("Loading processed dataset...")
    df = pd.read_csv(DATA_PATH)

    # features only for explanation
    X = df.drop(columns=["class"], errors="ignore")

    if MAX_ROWS is not None:
        X = X.iloc[:MAX_ROWS]
        df = df.iloc[:MAX_ROWS]

    print(f"Rows to process: {len(X)}")

    all_results = []
    summary_rows = []

    matched_rows = 0
    unmatched_rows = 0

    for i in range(len(X)):
        if i % 1000 == 0:
            print(f"Processing row {i}/{len(X)}")

        row = X.iloc[i]

        try:
            explanations = explain_row(
                row=row,
                rules_csv_path=RULES_PATH,
                meta=None,
                top_k=TOP_K
            )
        except Exception as e:
            summary_rows.append({
                "row_index": i,
                "true_class": df.iloc[i]["class"] if "class" in df.columns else None,
                "matched_rules": 0,
                "status": f"error: {str(e)}"
            })
            continue

        if explanations:
            matched_rows += 1

            all_results.append({
                "row_index": i,
                "true_class": float(df.iloc[i]["class"]) if "class" in df.columns else None,
                "matched_rules": len(explanations),
                "top_explanations": explanations
            })

            top1 = explanations[0]
            summary_rows.append({
                "row_index": i,
                "true_class": df.iloc[i]["class"] if "class" in df.columns else None,
                "matched_rules": len(explanations),
                "top_predicted_class": top1.get("Predicted_Class"),
                "top_confidence": top1.get("Confidence"),
                "top_support": top1.get("Support"),
                "top_f1": top1.get("F1"),
                "top_rule": top1.get("Readable_Rule"),
                "status": "matched"
            })
        else:
            unmatched_rows += 1
            summary_rows.append({
                "row_index": i,
                "true_class": df.iloc[i]["class"] if "class" in df.columns else None,
                "matched_rules": 0,
                "top_predicted_class": None,
                "top_confidence": None,
                "top_support": None,
                "top_f1": None,
                "top_rule": None,
                "status": "no_rule_fired"
            })

    print("\nDone.")
    print(f"Matched rows: {matched_rows}")
    print(f"Unmatched rows: {unmatched_rows}")

    with open(OUT_JSON, "w") as f:
        json.dump(all_results, f, indent=2)

    pd.DataFrame(summary_rows).to_csv(OUT_CSV, index=False)

    print(f"Saved JSON results to: {OUT_JSON}")
    print(f"Saved CSV summary to: {OUT_CSV}")


if __name__ == "__main__":
    main()