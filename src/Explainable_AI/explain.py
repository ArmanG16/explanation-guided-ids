import json
import pandas as pd
from typing import Dict, Any, List
from .rule_translation import (parse_antecedent_dict,
    rule_to_text,
    parse_condition,
    condition_to_text,
    parse_consequent_label,
    feature_label,
    feature_unit,
    maybe_inverse_scale)
from .explanation_metrics import rule_fires_mask

def fmt_num(x: float) -> str:
    """Human-friendly numeric formatting that won't collapse small values to 0."""
    ax = abs(x)
    if ax == 0:
        return "0"
    if ax < 0.001:
        return f"{x:.2e}"   # scientific for tiny values
    if ax < 1:
        return f"{x:.4f}"   # keep detail for small numbers
    if ax < 1000:
        return f"{x:.3f}"   # normal
    return f"{x:,.0f}"      # big numbers as integers with commas

def load_meta(meta_path: str) -> Dict[str, Any]:
    with open(meta_path, "r") as f:
        return json.load(f)

def explain_row(row: pd.Series,
                rules_csv_path: str,
                meta: Dict = None,
                top_k: int = 3) -> List[Dict[str, Any]]:

    rules = pd.read_csv(rules_csv_path)
    df_one = row.to_frame().T
    fired_rules = []

    meta = meta or {}

    for _, r in rules.iterrows():
        antecedent = parse_antecedent_dict(r["antecedent"])

        if bool(rule_fires_mask(df_one, antecedent).iloc[0]):

            raw_consequent = str(r["consequent"])
            label = parse_consequent_label(raw_consequent)

            # ---------- Build readable rule ----------
            readable_rule = rule_to_text(antecedent, raw_consequent, meta)

            # ---------- Build "Triggered because" ----------
            because_lines = []

            for feat, raw_val in antecedent.items():
                c = parse_condition(feat, raw_val)
                rule_txt = condition_to_text(c, meta)

                obs = row.get(feat, None)
                obs_str = "N/A"

                try:
                    if obs is not None and pd.notna(obs):
                        feat_name = feature_label(feat, meta)
                        unit = feature_unit(feat, meta)

                        obs_f = float(obs)
                        inv_obs = maybe_inverse_scale(feat, obs_f, meta)

                        if inv_obs is not None:
                            unit_txt = f" {unit}" if unit else ""
                            obs_str = f"{inv_obs:.4g}{unit_txt}"
                        else:
                            obs_str = f"{obs_f:.3g}"
                    else:
                        feat_name = feature_label(feat, meta)

                except Exception:
                    feat_name = feature_label(feat, meta)
                    obs_str = str(obs)

                because_lines.append(
                    f"- {feat_name}: observed {obs_str}; rule says {rule_txt}"
                )

            # ---------- Metrics ----------
            conf = r.get("confidence")
            sup  = r.get("support")
            f1_val = r.get("f1")

            # Clean numeric values for JSON
            conf_val = float(conf) if pd.notna(conf) else None
            sup_val  = float(sup) if pd.notna(sup) else None
            f1_val   = float(f1_val) if pd.notna(f1_val) else None

            conf_str = f"{conf_val:.2f}" if conf_val is not None else "N/A"
            sup_pct  = f"{sup_val * 100:.2f}%" if sup_val is not None else "N/A"

            explanation = (
                f"\nPredicted {label} "
                f"(rule confidence={conf_str}, support={sup_pct}"
                + (f", F1={f1_val:.2f}" if f1_val is not None else "")
                + ") because:\n"
                "Triggered because:\n"
                + "\n".join(because_lines)
            )

            fired_rules.append({
                "Rule_Index": r.get("Rule_Index"),
                "Readable_Rule": readable_rule,
                "Predicted_Class": label,
                "Explanation": explanation,
                "Support": sup_val,
                "Confidence": conf_val,
                "F1": f1_val,
            })

    # ---------- Rank rules ----------
    fired_rules.sort(
        key=lambda x: (
            x["F1"] if x["F1"] is not None else 0,
            x["Confidence"] if x["Confidence"] is not None else 0,
            x["Support"] if x["Support"] is not None else 0,
        ),
        reverse=True
    )

    return fired_rules[:top_k]