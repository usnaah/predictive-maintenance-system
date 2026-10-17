"""Shared logic for the Predictive Maintenance Intelligence System.

Used by BOTH the training script (train_model.py) and the Flask app (app.py),
so feature engineering, validation and risk categories can never drift apart.
"""
import json
import logging
import math
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

log = logging.getLogger("pm.core")

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "predictive_maintenance_model.joblib"
OUTPUT_DIR = BASE_DIR / "outputs"

# ---------------------------------------------------------------- schema ---
TARGET = "Machine failure"
CATEGORICAL = "Type"
TYPE_VALUES = ["H", "L", "M"]
RAW_NUMERIC = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]
RAW_FEATURES = [CATEGORICAL] + RAW_NUMERIC          # the ONLY inputs the model needs
ENGINEERED = ["temp_diff_K", "power_W", "wear_x_torque"]
NUMERIC_MODEL = RAW_NUMERIC + ENGINEERED

# Columns deliberately NOT used as inputs (see README / notebook for the evidence).
LEAKAGE_COLUMNS = ["TWF", "HDF", "PWF", "OSF", "RNF"]
IDENTIFIER_COLUMNS = ["UDI", "Product ID"]

# Input sanity limits used by the app. These are plausibility checks chosen for
# this project - NOT industrial standards. Values outside the range seen in the
# training data still trigger an extra "extrapolation" warning.
FIELD_SPECS = {
    "Type": {"label": "Product quality type", "unit": "", "help": "L = low, M = medium, H = high quality variant"},
    "Air temperature [K]": {"label": "Air temperature", "unit": "K", "min": 250, "max": 350, "step": 0.1, "example": 300.0},
    "Process temperature [K]": {"label": "Process temperature", "unit": "K", "min": 250, "max": 400, "step": 0.1, "example": 310.0},
    "Rotational speed [rpm]": {"label": "Rotational speed", "unit": "rpm", "min": 100, "max": 10000, "step": 1, "example": 1500},
    "Torque [Nm]": {"label": "Torque", "unit": "Nm", "min": 0, "max": 300, "step": 0.1, "example": 40.0},
    "Tool wear [min]": {"label": "Tool wear", "unit": "min", "min": 0, "max": 1000, "step": 1, "example": 100},
}
MAX_BATCH_ROWS = 5000


class BatchError(ValueError):
    """Raised when an uploaded CSV cannot be processed at all."""


# ----------------------------------------------------- feature engineering ---
def add_features(X: pd.DataFrame) -> pd.DataFrame:
    """Derive physically motivated features from the raw inputs.

    temp_diff_K   : process - air temperature (heat dissipation indicator)
    power_W       : torque x angular speed  (mechanical power, watts)
    wear_x_torque : tool wear x torque      (overstrain indicator)
    """
    X = X.copy()
    X["temp_diff_K"] = X["Process temperature [K]"] - X["Air temperature [K]"]
    X["power_W"] = X["Torque [Nm]"] * X["Rotational speed [rpm]"] * 2 * np.pi / 60
    X["wear_x_torque"] = X["Tool wear [min]"] * X["Torque [Nm]"]
    return X


# ----------------------------------------------------------- validation ---
def validate_record(raw: dict):
    """Validate one machine record (e.g. from the web form).

    Returns (clean_record, errors). `errors` maps field name -> message.
    """
    errors, clean = {}, {}
    t = str(raw.get("Type", "")).strip().upper()
    if t not in TYPE_VALUES:
        errors["Type"] = "Choose L, M or H."
    else:
        clean["Type"] = t
    for col in RAW_NUMERIC:
        spec = FIELD_SPECS[col]
        text = str(raw.get(col, "")).strip()
        if text == "":
            errors[col] = "This field is required."
            continue
        try:
            val = float(text)
        except ValueError:
            errors[col] = "Enter a number."
            continue
        if not math.isfinite(val):
            errors[col] = "Value must be a finite number."
        elif not spec["min"] <= val <= spec["max"]:
            errors[col] = f"Must be between {spec['min']} and {spec['max']} {spec['unit']}."
        else:
            clean[col] = val
    return (None if errors else clean), errors


def validate_batch(df: pd.DataFrame):
    """Validate an uploaded DataFrame. Returns (valid_df, issues).

    valid_df keeps the original CSV line number in `csv_line`.
    issues is a list of {"csv_line": n, "problem": "..."} for rejected rows.
    Raises BatchError for file-level problems.
    """
    if df is None or df.empty:
        raise BatchError("The file contains no data rows.")
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    dupes = sorted({c for c in RAW_FEATURES if list(df.columns).count(c) > 1})
    if dupes:
        raise BatchError("Duplicate column(s) in header: " + ", ".join(dupes))
    missing = [c for c in RAW_FEATURES if c not in df.columns]
    if missing:
        raise BatchError("Missing required column(s): " + ", ".join(missing))
    if len(df) > MAX_BATCH_ROWS:
        raise BatchError(f"Too many rows ({len(df)}). The limit is {MAX_BATCH_ROWS} rows per upload.")

    work = df[RAW_FEATURES].copy()
    work.insert(0, "csv_line", np.arange(len(work)) + 2)      # header is line 1
    problems = [[] for _ in range(len(work))]

    work["Type"] = work["Type"].astype(str).str.strip().str.upper()
    for i in np.where(~work["Type"].isin(TYPE_VALUES))[0]:
        problems[i].append("Type must be L, M or H")
    for col in RAW_NUMERIC:
        num = pd.to_numeric(work[col], errors="coerce")
        spec = FIELD_SPECS[col]
        bad_num = num.isna() | ~np.isfinite(num.fillna(np.nan).astype(float))
        out_rng = ~bad_num & ((num < spec["min"]) | (num > spec["max"]))
        for i in np.where(bad_num)[0]:
            problems[i].append(f"{col}: not a valid number")
        for i in np.where(out_rng)[0]:
            problems[i].append(f"{col}: outside {spec['min']}-{spec['max']}")
        work[col] = num
    bad_rows = np.array([bool(p) for p in problems])
    issues = [{"csv_line": int(work["csv_line"].iloc[i]), "problem": "; ".join(problems[i])}
              for i in np.where(bad_rows)[0]]
    valid = work.loc[~bad_rows].reset_index(drop=True)
    return valid, issues


# ------------------------------------------------------------ prediction ---
def risk_category(prob: float, threshold: float) -> str:
    """Project-defined (NOT an industry standard) risk bands.

    High   : probability >= decision threshold chosen during training
    Medium : probability >= half of that threshold
    Low    : otherwise
    """
    if prob >= threshold:
        return "High"
    if prob >= threshold / 2:
        return "Medium"
    return "Low"


class Predictor:
    """Loads the saved artifact once and serves single/batch predictions."""

    def __init__(self, path: Path = MODEL_PATH):
        if not Path(path).exists():
            raise FileNotFoundError(f"Model artifact not found at {path}. Run `python train_model.py` first.")
        art = joblib.load(path)
        required = {"model", "threshold", "model_name", "raw_features", "training_ranges", "medians"}
        if not required.issubset(art):
            raise ValueError(f"Artifact is missing keys: {sorted(required - set(art))}")
        if list(art["raw_features"]) != RAW_FEATURES:
            raise ValueError("Artifact feature schema does not match core.RAW_FEATURES - retrain the model.")
        self.model = art["model"]
        self.threshold = float(art["threshold"])
        self.model_name = art["model_name"]
        self.ranges = art["training_ranges"]
        self.medians = art["medians"]
        self.meta = {k: art[k] for k in ("trained_at", "sklearn_version", "random_state") if k in art}

    def predict_frame(self, df: pd.DataFrame) -> pd.DataFrame:
        X = df[RAW_FEATURES]                         # enforces the feature order
        prob = self.model.predict_proba(X)[:, 1]
        out = df.copy()
        out["failure_probability"] = prob
        out["predicted_failure"] = (prob >= self.threshold).astype(int)
        out["risk_category"] = [risk_category(p, self.threshold) for p in prob]
        return out

    def predict_one(self, record: dict) -> dict:
        row = pd.DataFrame([record])[RAW_FEATURES]
        res = self.predict_frame(row).iloc[0]
        eng = add_features(row).iloc[0]
        return {
            "probability": float(res["failure_probability"]),
            "predicted_failure": int(res["predicted_failure"]),
            "risk": res["risk_category"],
            "threshold": self.threshold,
            "engineered": {k: float(eng[k]) for k in ENGINEERED},
            "drivers": self.local_effects(record),
            "out_of_range": self.out_of_range(record),
        }

    def local_effects(self, record: dict):
        """What-if analysis: change in probability if one input were set to the
        training-set median (most common Type for Type). Simple, model-agnostic,
        and an *approximation* - not a causal statement."""
        base = pd.DataFrame([record])[RAW_FEATURES]
        variants = [base]
        for col in RAW_FEATURES:
            v = base.copy()
            v[col] = self.medians[col]
            variants.append(v)
        probs = self.model.predict_proba(pd.concat(variants, ignore_index=True))[:, 1]
        effects = [{"feature": col, "delta": float(probs[0] - probs[i + 1])} for i, col in enumerate(RAW_FEATURES)]
        return sorted(effects, key=lambda e: -abs(e["delta"]))

    def out_of_range(self, record: dict):
        notes = []
        for col in RAW_NUMERIC:
            lo, hi = self.ranges[col]
            if not lo <= float(record[col]) <= hi:
                notes.append(f"{col} = {record[col]} is outside the range seen in training data ({lo}-{hi}).")
        return notes


def load_json(name: str):
    path = OUTPUT_DIR / name
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)
