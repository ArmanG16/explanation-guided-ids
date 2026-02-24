import ast
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

@dataclass
class Condition:
    feature: str
    raw: Any  # usually string after CSV
    kind: str # "interval" | "eq" | "other"
    lo: Optional[float] = None
    hi: Optional[float] = None
    lo_inclusive: bool = False
    hi_inclusive: bool = False
    eq_value: Optional[Any] = None

_INTERVAL_RE = re.compile(r"^([\(\[])\s*([^,]+)\s*,\s*([^\]\)]+)\s*([\]\)])$")

def _try_float(x: str) -> Optional[float]:
    x = x.strip()
    if x in ("inf", "+inf", "Infinity", "+Infinity"):
        return float("inf")
    if x in ("-inf", "-Infinity"):
        return float("-inf")
    try:
        return float(x)
    except:
        return None

def feature_label(feature: str, meta: Optional[Dict] = None) -> str:
    meta = meta or {}
    # if you later add meta["feature_display"], it'll use it; else default
    return meta.get("feature_display", {}).get(feature, feature)

def feature_unit(feature: str, meta: Optional[Dict] = None) -> str:
    meta = meta or {}
    return meta.get("units", {}).get(feature, "")

def z_qual(z: float) -> str:
    # quick human-friendly interpretation of a z-score
    if z <= -2.0: return "far below global average"
    if z <= -1.0: return "below global average"
    if z <  1.0:  return "near global average"
    if z <  2.0:  return "above global average"
    return "far above global average"

def maybe_inverse_scale(feature: str, z: float, meta: Dict) -> Optional[float]:
    """Return original-unit value if mean/scale exist; else None."""
    scaler = (meta or {}).get("scaler", {})
    mean = scaler.get("mean", {}).get(feature)
    scale = scaler.get("scale", {}).get(feature)
    if mean is None or scale is None:
        return None
    return z * float(scale) + float(mean)


def parse_antecedent_dict(s: str) -> Dict[str, Any]:
    """Parse antecedent strings from pyIDS CAR mining outputs.

    CSV examples:
      Antecedent(('feat','val'))
      Antecedent(('feat','val'), ('feat2','val2'))
      (('feat','val'), ('feat2','val2'))

    Returns {feature: value, ...}
    """
    if s is None:
        return {}
    s = str(s).strip()
    if s.startswith('Antecedent(') and s.endswith(')'):
        s = s[len('Antecedent('):-1].strip()

    obj = ast.literal_eval(s)

    # obj can be ('A','1') or (('A','1'), ('B','2'))
    if isinstance(obj, tuple) and len(obj) == 2 and all(isinstance(x, str) for x in obj):
        pairs = [obj]
    else:
        pairs = list(obj)

    return {k: v for (k, v) in pairs}


def parse_condition(feature: str, val: Any) -> Condition:
    # Most likely val is a string like "(0.1, 0.9]" or a scalar
    if isinstance(val, str):
        m = _INTERVAL_RE.match(val.strip())
        if m:
            lo_inc = (m.group(1) == "[")
            hi_inc = (m.group(4) == "]")
            lo = _try_float(m.group(2))
            hi = _try_float(m.group(3))
            return Condition(feature=feature, raw=val, kind="interval",
                            lo=lo, hi=hi, lo_inclusive=lo_inc, hi_inclusive=hi_inc)
    # fallback = equality
    return Condition(feature=feature, raw=val, kind="eq", eq_value=val)

def inverse_scale(feature: str, x: float, meta: Dict) -> float:
    scaler = meta.get("scaler", {})
    mean = scaler.get("mean", {}).get(feature)
    scale = scaler.get("scale", {}).get(feature)
    if mean is None or scale is None:
        return x
    return x * float(scale) + float(mean)

def decode_category(feature: str, code: Any, meta: Dict) -> Any:
    enc = meta.get("label_encoders", {}).get(feature)
    if enc is None:
        return code
    try:
        return enc.get(int(code), code)
    except:
        return code

def condition_to_text(c: Condition, meta: Optional[Dict] = None) -> str:
    meta = meta or {}

    def fmt_num(x: float) -> str:
        """Human-friendly numeric formatting that won't collapse small values to 0."""
        try:
            x = float(x)
        except Exception:
            return str(x)

        ax = abs(x)
        if ax == 0:
            return "0"
        if ax < 0.001:
            return f"{x:.2e}"      # scientific for tiny values
        if ax < 1:
            return f"{x:.4f}"      # keep detail for small numbers
        if ax < 1000:
            return f"{x:.3f}"      # normal
        return f"{x:,.0f}"         # big numbers as integers with commas

    if c.kind == "interval" and c.lo is not None and c.hi is not None:
        lo = inverse_scale(c.feature, c.lo, meta) if c.lo not in (float("inf"), float("-inf")) else c.lo
        hi = inverse_scale(c.feature, c.hi, meta) if c.hi not in (float("inf"), float("-inf")) else c.hi
        lo_br = "≥" if c.lo_inclusive else ">"
        hi_br = "≤" if c.hi_inclusive else "<"

        # Handle infinite bounds nicely
        if c.lo == float("-inf"):
            return f"{feature_label(c.feature, meta)} {hi_br} {fmt_num(hi)}"
        if c.hi == float("inf"):
            return f"{feature_label(c.feature, meta)} {lo_br} {fmt_num(lo)}"

        feat_name = feature_label(c.feature, meta)
        return f"{fmt_num(lo)} {lo_br} {feat_name} {hi_br} {fmt_num(hi)}"

    if c.kind == "eq":
        feat_name = feature_label(c.feature, meta)
        unit = feature_unit(c.feature, meta)

        v = decode_category(c.feature, c.eq_value, meta)

        # If numeric-like string, treat as z-score and show inverse-scaled value with honest formatting
        if isinstance(v, str):
            z = _try_float(v)
            if z is not None:
                qual = z_qual(z)
                inv = maybe_inverse_scale(c.feature, z, meta)
                if inv is not None:
                    unit_txt = f" {unit}" if unit else ""
                    return f"{feat_name} ≈ {fmt_num(inv)}{unit_txt} ({qual})"
                else:
                    return f"{feat_name} is {qual} (z={fmt_num(z)})"

        # Non-numeric / categorical
        return f"{feat_name} = {v}"

    return f"{feature_label(c.feature, meta)} matches {c.raw}"

def parse_consequent_label(s: Any) -> str:
    """
    Converts strings like:
      "Consequent{('class', 'PortScan')}"
    into:
      "PortScan"
    """
    if s is None:
        return ""
    s = str(s).strip()

    if s.startswith("Consequent{") and s.endswith("}"):
        inner = s[len("Consequent{"):-1].strip()  # "('class', 'PortScan')"
        try:
            k, v = ast.literal_eval(inner)
            return str(v)
        except Exception:
            # fallback: just return inner if parsing fails
            return inner

    return s

def rule_to_text(antecedent: Dict[str, Any], consequent: str, meta: Optional[Dict] = None) -> str:
    conds = [parse_condition(k, v) for k, v in antecedent.items()]
    cond_text = " AND ".join(condition_to_text(c, meta) for c in conds)

    label = parse_consequent_label(consequent)
    return f"IF {cond_text} THEN class = {label}"
