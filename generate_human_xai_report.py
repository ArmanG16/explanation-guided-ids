import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd


def fmt_num(x: Any, digits: int = 4) -> str:
    if x is None:
        return "N/A"
    try:
        xf = float(x)
    except Exception:
        return str(x)
    if math.isnan(xf):
        return "N/A"
    if abs(xf) >= 1000:
        return f"{xf:,.0f}"
    return f"{xf:.{digits}f}".rstrip("0").rstrip(".")


def pct(x: Any, digits: int = 2) -> str:
    if x is None:
        return "N/A"
    try:
        xf = float(x)
    except Exception:
        return str(x)
    if math.isnan(xf):
        return "N/A"
    return f"{xf * 100:.{digits}f}%"


def clean_label(label: Any, positive_name: str, negative_name: str) -> str:
    if label is None:
        return "Unknown"
    s = str(label).strip()
    if s in {"1", "1.0", "np.float64(1.0)"}:
        return positive_name
    if s in {"0", "0.0", "np.float64(0.0)"}:
        return negative_name
    return s


def load_json_results(path: Path) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("JSON input must be a list of row-level explanation objects.")
    return data


def load_csv_results(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def summarize_json(rows: List[Dict[str, Any]], positive_name: str, negative_name: str) -> Dict[str, Any]:
    total_rows = len(rows)
    matched_rows = sum(1 for r in rows if r.get("matched_rules", 0) > 0)
    total_rules = sum(len(r.get("top_explanations", [])) for r in rows)

    true_class_counts = Counter()
    pred_class_counts = Counter()
    top_rule_counts = Counter()
    all_rule_counts = Counter()
    support_vals = []
    confidence_vals = []

    for r in rows:
        true_class_counts[clean_label(r.get("true_class"), positive_name, negative_name)] += 1
        explanations = r.get("top_explanations", [])
        if explanations:
            pred = clean_label(explanations[0].get("Predicted_Class"), positive_name, negative_name)
            pred_class_counts[pred] += 1
            top_rule_counts[explanations[0].get("Readable_Rule", "<missing>")] += 1
        for ex in explanations:
            rule = ex.get("Readable_Rule", "<missing>")
            all_rule_counts[rule] += 1
            if ex.get("Support") is not None:
                support_vals.append(float(ex["Support"]))
            if ex.get("Confidence") is not None:
                confidence_vals.append(float(ex["Confidence"]))

    return {
        "total_rows": total_rows,
        "matched_rows": matched_rows,
        "coverage": (matched_rows / total_rows) if total_rows else 0,
        "avg_rules_per_matched_row": (total_rules / matched_rows) if matched_rows else 0,
        "true_class_counts": true_class_counts,
        "pred_class_counts": pred_class_counts,
        "top_rule_counts": top_rule_counts,
        "all_rule_counts": all_rule_counts,
        "avg_support": (sum(support_vals) / len(support_vals)) if support_vals else None,
        "avg_confidence": (sum(confidence_vals) / len(confidence_vals)) if confidence_vals else None,
    }


def summarize_csv(df: pd.DataFrame, positive_name: str, negative_name: str) -> Dict[str, Any]:
    total_rows = len(df)
    status_col = "status" if "status" in df.columns else None
    matched_mask = df[status_col].eq("matched") if status_col else df["matched_rules"].fillna(0).gt(0)
    matched_rows = int(matched_mask.sum())
    true_class_counts = Counter(clean_label(v, positive_name, negative_name) for v in df.get("true_class", []))
    pred_class_counts = Counter(clean_label(v, positive_name, negative_name) for v in df.get("top_predicted_class", []))
    top_rule_counts = Counter(str(v) for v in df.get("top_rule", []) if pd.notna(v))

    support_series = pd.to_numeric(df.get("top_support"), errors="coerce") if "top_support" in df.columns else pd.Series(dtype=float)
    conf_series = pd.to_numeric(df.get("top_confidence"), errors="coerce") if "top_confidence" in df.columns else pd.Series(dtype=float)
    matched_rules_series = pd.to_numeric(df.get("matched_rules"), errors="coerce") if "matched_rules" in df.columns else pd.Series(dtype=float)

    return {
        "total_rows": total_rows,
        "matched_rows": matched_rows,
        "coverage": (matched_rows / total_rows) if total_rows else 0,
        "avg_rules_per_matched_row": float(matched_rules_series[matched_mask].mean()) if matched_rows and len(matched_rules_series) else 0,
        "true_class_counts": true_class_counts,
        "pred_class_counts": pred_class_counts,
        "top_rule_counts": top_rule_counts,
        "avg_support": float(support_series[matched_mask].mean()) if len(support_series) else None,
        "avg_confidence": float(conf_series[matched_mask].mean()) if len(conf_series) else None,
    }


def build_markdown_from_json(rows: List[Dict[str, Any]], summary: Dict[str, Any], positive_name: str, negative_name: str, examples_per_rule: int, max_rows: int) -> str:
    lines: List[str] = []
    lines.append("# Human-Readable XAI Report")
    lines.append("")
    lines.append("## What this file is")
    lines.append("This report translates the raw XAI output into plain language so a reviewer can quickly understand what the rules are doing, how often they appear, and what they mean for individual rows in the processed dataset.")
    lines.append("")
    lines.append("## Executive Summary")
    lines.append(f"- Total explained rows in input file: **{summary['total_rows']:,}**")
    lines.append(f"- Rows where at least one rule fired: **{summary['matched_rows']:,}**")
    lines.append(f"- Rule coverage in this file: **{pct(summary['coverage'])}**")
    lines.append(f"- Average number of returned explanations per matched row: **{fmt_num(summary['avg_rules_per_matched_row'], 2)}**")
    if summary.get("avg_confidence") is not None:
        lines.append(f"- Average rule confidence: **{fmt_num(summary['avg_confidence'], 3)}**")
    if summary.get("avg_support") is not None:
        lines.append(f"- Average rule support: **{pct(summary['avg_support'])}**")
    lines.append("")

    if summary.get("true_class_counts"):
        lines.append("## True Class Distribution")
        for cls, cnt in summary["true_class_counts"].most_common():
            if cls != "Unknown":
                lines.append(f"- **{cls}**: {cnt:,}")
        lines.append("")

    if summary.get("pred_class_counts"):
        lines.append("## Predicted Class Distribution (Top Fired Rule)")
        for cls, cnt in summary["pred_class_counts"].most_common():
            if cls != "Unknown":
                lines.append(f"- **{cls}**: {cnt:,}")
        lines.append("")

    lines.append("## Most Common Fired Rules")
    for idx, (rule, cnt) in enumerate(summary["top_rule_counts"].most_common(10), start=1):
        lines.append(f"{idx}. **{rule}** — top explanation for {cnt:,} row(s)")
    lines.append("")

    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        for ex in row.get("top_explanations", []):
            grouped[ex.get("Readable_Rule", "<missing rule>")].append({
                "row_index": row.get("row_index"),
                "true_class": clean_label(row.get("true_class"), positive_name, negative_name),
                "predicted_class": clean_label(ex.get("Predicted_Class"), positive_name, negative_name),
                "support": ex.get("Support"),
                "confidence": ex.get("Confidence"),
                "explanation": ex.get("Explanation", "").strip(),
            })

    lines.append("## Rule-by-Rule Interpretation")
    for idx, (rule, examples) in enumerate(sorted(grouped.items(), key=lambda kv: len(kv[1]), reverse=True)[:10], start=1):
        lines.append(f"### Rule {idx}")
        lines.append(f"**Rule text:** {rule}")
        lines.append(f"**How often it appeared in the returned explanations:** {len(examples):,} row(s)")

        supports = [e["support"] for e in examples if e["support"] is not None]
        confs = [e["confidence"] for e in examples if e["confidence"] is not None]
        preds = Counter(e["predicted_class"] for e in examples)
        trues = Counter(e["true_class"] for e in examples)

        if supports:
            lines.append(f"**Support:** around {pct(sum(supports)/len(supports))}")
        if confs:
            lines.append(f"**Confidence:** around {fmt_num(sum(confs)/len(confs), 3)}")
        if preds:
            lines.append("**Predicted class when this rule fired:** " + ", ".join(f"{k} ({v:,})" for k, v in preds.items()))
        if trues:
            lines.append("**True class among shown rows:** " + ", ".join(f"{k} ({v:,})" for k, v in trues.items()))
        lines.append("")
        lines.append("**Example rows and plain-language explanations:**")
        for ex in examples[:examples_per_rule]:
            lines.append(f"- Row **{ex['row_index']}** | true class: **{ex['true_class']}** | predicted class: **{ex['predicted_class']}**")
            for expl_line in ex["explanation"].splitlines():
                if expl_line.strip():
                    lines.append(f"  - {expl_line.strip()}")
        lines.append("")

    lines.append("## Individual Row Explanations")
    lines.append(f"Below are the first **{min(max_rows, len(rows))}** explained rows from the JSON file, rewritten in a reviewer-friendly format.")
    lines.append("")
    for row in rows[:max_rows]:
        row_idx = row.get("row_index")
        true_cls = clean_label(row.get("true_class"), positive_name, negative_name)
        matched = row.get("matched_rules", 0)
        lines.append(f"### Row {row_idx}")
        lines.append(f"- True class: **{true_cls}**")
        lines.append(f"- Number of matched rules returned: **{matched}**")
        explanations = row.get("top_explanations", [])
        if not explanations:
            lines.append("- No explanation was returned for this row.")
            lines.append("")
            continue
        for i, ex in enumerate(explanations, start=1):
            pred_cls = clean_label(ex.get("Predicted_Class"), positive_name, negative_name)
            lines.append(f"- Explanation {i}: predicts **{pred_cls}** | confidence **{fmt_num(ex.get('Confidence'), 3)}** | support **{pct(ex.get('Support'))}**")
            lines.append(f"  - Rule: {ex.get('Readable_Rule', '')}")
            expl = (ex.get("Explanation") or "").strip()
            for expl_line in expl.splitlines():
                if expl_line.strip():
                    lines.append(f"  - {expl_line.strip()}")
        lines.append("")

    lines.append("## How to Read This Report")
    lines.append("- **Support** tells you how common a rule is in the mined rule set context. Higher support means the pattern appeared more broadly.")
    lines.append("- **Confidence** tells you how often the rule’s consequent was correct when that rule fired in the mined rules output.")
    lines.append("- **Matched rules** is how many of the returned top explanations fired for that row in this report.")
    lines.append("- Repeated rules across many rows usually mean the model found a recurring pattern in the processed feature space.")
    lines.append("")
    lines.append("## Caution for Reviewers")
    lines.append("The feature values shown in the rules appear to be processed/scaled values rather than raw original network units. This means the report is accurate about which processed patterns fired, but a separate metadata mapping would be needed to translate each feature value back into original human units.")
    return "\n".join(lines)


def build_markdown_from_csv(df: pd.DataFrame, summary: Dict[str, Any], positive_name: str, negative_name: str, max_rows: int) -> str:
    lines: List[str] = []
    lines.append("# Human-Readable XAI Summary Report")
    lines.append("")
    lines.append("## Executive Summary")
    lines.append(f"- Total rows in CSV summary: **{summary['total_rows']:,}**")
    lines.append(f"- Matched rows: **{summary['matched_rows']:,}**")
    lines.append(f"- Coverage: **{pct(summary['coverage'])}**")
    lines.append(f"- Average matched rules per matched row: **{fmt_num(summary['avg_rules_per_matched_row'], 2)}**")
    if summary.get("avg_confidence") is not None:
        lines.append(f"- Average top-rule confidence: **{fmt_num(summary['avg_confidence'], 3)}**")
    if summary.get("avg_support") is not None:
        lines.append(f"- Average top-rule support: **{pct(summary['avg_support'])}**")
    lines.append("")

    lines.append("## Most Common Top Rules")
    for idx, (rule, cnt) in enumerate(summary["top_rule_counts"].most_common(10), start=1):
        lines.append(f"{idx}. **{rule}** — {cnt:,} row(s)")
    lines.append("")

    lines.append("## Reviewer-Friendly Row Summary")
    show_df = df.head(max_rows)
    for _, row in show_df.iterrows():
        row_idx = row.get("row_index", "N/A")
        true_cls = clean_label(row.get("true_class"), positive_name, negative_name)
        pred_cls = clean_label(row.get("top_predicted_class"), positive_name, negative_name)
        lines.append(f"### Row {row_idx}")
        lines.append(f"- Status: **{row.get('status', 'unknown')}**")
        lines.append(f"- True class: **{true_cls}**")
        lines.append(f"- Top predicted class: **{pred_cls}**")
        lines.append(f"- Matched rules: **{row.get('matched_rules', 0)}**")
        lines.append(f"- Top rule confidence: **{fmt_num(row.get('top_confidence'), 3)}**")
        lines.append(f"- Top rule support: **{pct(row.get('top_support'))}**")
        top_rule = row.get("top_rule")
        if pd.notna(top_rule):
            lines.append(f"- Top rule: {top_rule}")
        lines.append("")

    lines.append("## Caution for Reviewers")
    lines.append("This CSV-based report is a summary view. For the full row-level explanations with all triggered conditions, use the JSON-based report generator on the raw XAI JSON output.")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Turn raw XAI JSON or CSV output into a human-readable Markdown report.")
    parser.add_argument("--input", required=True, help="Path to the XAI JSON or CSV file.")
    parser.add_argument("--output", required=True, help="Path to the output Markdown file.")
    parser.add_argument("--positive-name", default="Attack / Positive Class", help="Human name for label 1.0")
    parser.add_argument("--negative-name", default="Normal / Negative Class", help="Human name for label 0.0")
    parser.add_argument("--examples-per-rule", type=int, default=3, help="How many example rows to show per common rule in JSON mode.")
    parser.add_argument("--max-rows", type=int, default=50, help="How many row sections to print in the report.")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if input_path.suffix.lower() == ".json":
        rows = load_json_results(input_path)
        summary = summarize_json(rows, args.positive_name, args.negative_name)
        report = build_markdown_from_json(rows, summary, args.positive_name, args.negative_name, args.examples_per_rule, args.max_rows)
    elif input_path.suffix.lower() == ".csv":
        df = load_csv_results(input_path)
        summary = summarize_csv(df, args.positive_name, args.negative_name)
        report = build_markdown_from_csv(df, summary, args.positive_name, args.negative_name, args.max_rows)
    else:
        raise ValueError("Input must be a .json or .csv file")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report)

    print(f"Saved human-readable report to: {output_path}")


if __name__ == "__main__":
    main()
