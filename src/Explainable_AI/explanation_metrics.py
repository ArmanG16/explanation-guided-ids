import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from .rule_translation import parse_antecedent_dict, parse_condition

def _match_condition(series: pd.Series, cond) -> pd.Series:
    if cond.kind == "interval" and cond.lo is not None and cond.hi is not None:
        lo_ok = series >= cond.lo if cond.lo_inclusive else series > cond.lo
        hi_ok = series <= cond.hi if cond.hi_inclusive else series < cond.hi
        return lo_ok & hi_ok
    if cond.kind == "eq":
        return series == cond.eq_value
    # fallback: can't evaluate, return all False
    return pd.Series([False] * len(series), index=series.index)

def rule_fires_mask(df: pd.DataFrame, antecedent: dict, eps: float = 1e-6) -> pd.Series:
    """
    Returns a boolean mask for rows in df that satisfy all conditions in antecedent.

    Handles:
      - numeric equality with tolerance (float-ish values)
      - exact match for non-numeric values
    """
    mask = pd.Series(True, index=df.index)

    for feat, raw_val in antecedent.items():
        if feat not in df.columns:
            # feature missing -> rule can't fire
            return pd.Series(False, index=df.index)

        col = df[feat]

        # Try numeric compare first
        try:
            v = float(raw_val)
            # cast column to float safely
            col_num = pd.to_numeric(col, errors="coerce")
            mask &= (col_num - v).abs() <= eps
        except Exception:
            # fallback string compare
            mask &= col.astype(str) == str(raw_val)

        if not mask.any():
            break

    return mask

def compute_rule_stats(rules_csv_path: str, df: pd.DataFrame, class_col: str="class") -> pd.DataFrame:
    rules_df = pd.read_csv(rules_csv_path)
    out_rows = []

    base_rate = df[class_col].value_counts(normalize=True).to_dict()

    for _, r in rules_df.iterrows():
        antecedent = parse_antecedent_dict(r["Antecedent"])
        consequent = str(r["Consequent"])
        fired = rule_fires_mask(df, antecedent)

        fires = int(fired.sum())
        if fires == 0:
            precision = 0.0
        else:
            precision = float((df.loc[fired, class_col].astype(str) == consequent).mean())

        coverage = fires / len(df) if len(df) else 0.0
        lift = precision / base_rate.get(consequent, 1e-9)

        out_rows.append({
            "Rule_Index": r.get("Rule_Index"),
            "Consequent": consequent,
            "Rule_Length": len(antecedent),
            "Fires": fires,
            "Coverage": coverage,
            "Precision_When_Fires": precision,
            "Lift_vs_BaseRate": lift,
            "Support": r.get("Support"),
            "Confidence": r.get("Confidence"),
            "F1": r.get("F1"),
        })

    return pd.DataFrame(out_rows).sort_values(["Precision_When_Fires","Fires"], ascending=[False, False])
