import pandas as pd
import numpy as np

from src.Explainable_AI.rule_translation import parse_antecedent_dict, parse_consequent_label
from src.Explainable_AI.explanation_metrics import rule_fires_mask

RULES_IN  = "data/cars/UNR-IDD.csv"
DATASET   = "data/processed/unridd_preprocessed.csv"
OUT_CSV   = "data/cars/UNR-IDD_with_f1.csv"
CLASS_COL = "class"

def safe_div(a, b):
    return float(a) / float(b) if b else 0.0

def main():
    rules = pd.read_csv(RULES_IN)
    df = pd.read_csv(DATASET)

    if CLASS_COL not in df.columns:
        raise SystemExit(f"Dataset missing label column '{CLASS_COL}'. Columns: {df.columns.tolist()}")

    y = df[CLASS_COL].astype(str)

    if "antecedent" not in rules.columns or "consequent" not in rules.columns:
        raise SystemExit(f"Rules file missing 'antecedent'/'consequent'. Columns: {rules.columns.tolist()}")

    tp_list, fp_list, fn_list = [], [], []
    prec_list, rec_list, f1_list = [], [], []

    for idx, r in rules.iterrows():
        antecedent = parse_antecedent_dict(r["antecedent"])
        rule_label = str(parse_consequent_label(r["consequent"]))

        fires = rule_fires_mask(df, antecedent).astype(bool)
        is_pos = (y == rule_label)

        tp = int((fires & is_pos).sum())
        fp = int((fires & ~is_pos).sum())
        fn = int((~fires & is_pos).sum())

        precision = safe_div(tp, tp + fp)
        recall    = safe_div(tp, tp + fn)
        f1        = safe_div(2 * precision * recall, precision + recall) if (precision + recall) else 0.0

        tp_list.append(tp); fp_list.append(fp); fn_list.append(fn)
        prec_list.append(precision); rec_list.append(recall); f1_list.append(f1)

        if (idx + 1) % 200 == 0:
            print(f"Processed {idx+1}/{len(rules)} rules...")

    out = rules.copy()
    out["tp"] = tp_list
    out["fp"] = fp_list
    out["fn"] = fn_list
    out["precision"] = prec_list
    out["recall"] = rec_list
    out["f1"] = f1_list

    out.to_csv(OUT_CSV, index=False)
    print("Wrote:", OUT_CSV)

    # Show a quick sanity summary
    top = out.sort_values("f1", ascending=False).head(10)
    print("\nTop 10 rules by F1:")
    cols = [c for c in ["antecedent","consequent","support","confidence","precision","recall","f1","tp","fp","fn"] if c in top.columns]
    print(top[cols].to_string(index=False))

if __name__ == "__main__":
    main()
