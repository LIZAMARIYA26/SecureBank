"""
Module 1 + Module 2 helpers: load BankSim, clean it, encode, scale, split.

BankSim stores categorical values wrapped in single quotes
(e.g. 'C1093826151'). pandas is told to treat ' as the quote character
so those quotes disappear automatically.
"""

from __future__ import annotations

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

from config import DATASET_PATH, RANDOM_STATE, TEST_SIZE


def load_banksim(path=None) -> pd.DataFrame:
    """Load the BankSim CSV and return a clean pandas DataFrame."""
    csv_path = path or DATASET_PATH
    df = pd.read_csv(csv_path, quotechar="'")
    # Strip leftover whitespace on string columns
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].astype(str).str.strip()
    return df


def dataset_overview(df: pd.DataFrame) -> dict:
    """
    Module 1 summary used by train_model.py.
    Returns JSON-serialisable facts about the raw table.
    """
    missing = df.isnull().sum()
    return {
        "rows": int(len(df)),
        "columns": int(df.shape[1]),
        "column_names": list(df.columns),
        "dtypes": {c: str(t) for c, t in df.dtypes.items()},
        "missing_values": {c: int(v) for c, v in missing.items()},
        "duplicate_rows": int(df.duplicated().sum()),
        "fraud_count": int((df["fraud"] == 1).sum()) if "fraud" in df.columns else 0,
        "genuine_count": int((df["fraud"] == 0).sum()) if "fraud" in df.columns else 0,
        "head": df.head(10).to_dict(orient="records"),
    }


def print_dataset_report(df: pd.DataFrame) -> None:
    """Pretty-print Module 1 facts in the terminal."""
    info = dataset_overview(df)
    print("=" * 70)
    print("MODULE 1 — DATASET LOADING")
    print("=" * 70)
    print(f"Rows: {info['rows']:,}   Columns: {info['columns']}")
    print("\nFirst 10 records:")
    print(df.head(10).to_string(index=False))
    print("\nData types:")
    print(df.dtypes)
    print("\nMissing values per column:")
    print(df.isnull().sum())
    print(f"\nDuplicate rows: {info['duplicate_rows']}")
    print(f"Fraud: {info['fraud_count']:,}   Genuine: {info['genuine_count']:,}")
    print()


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Drop exact duplicate transactions."""
    before = len(df)
    cleaned = df.drop_duplicates().reset_index(drop=True)
    print(f"Removed {before - len(cleaned):,} duplicate rows.")
    return cleaned


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fill missing numbers with the median and missing categories with the mode.
    BankSim is already complete, but this keeps the pipeline general.
    """
    out = df.copy()
    for col in out.columns:
        if out[col].isnull().any():
            if pd.api.types.is_numeric_dtype(out[col]):
                out[col] = out[col].fillna(out[col].median())
            else:
                mode = out[col].mode()
                out[col] = out[col].fillna(mode.iloc[0] if len(mode) else "Unknown")
    return out


def encode_age(series: pd.Series) -> pd.Series:
    """
    BankSim age is a code: 0–6 plus 'U' (unknown).
    Convert to integers so the model can use it as a numeric feature.
    """
    mapped = pd.to_numeric(series.replace({"U": -1, "u": -1}), errors="coerce")
    return mapped.fillna(-1).astype(int)


def undersample_majority(df: pd.DataFrame, max_genuine: int, random_state: int) -> pd.DataFrame:
    """
    Keep every fraud row and randomly sample genuine rows.
    This speeds up training without throwing away the rare class.
    """
    fraud = df[df["fraud"] == 1]
    genuine = df[df["fraud"] == 0]
    if len(genuine) > max_genuine:
        genuine = genuine.sample(n=max_genuine, random_state=random_state)
    balanced = pd.concat([fraud, genuine], ignore_index=True)
    return balanced.sample(frac=1.0, random_state=random_state).reset_index(drop=True)


def fit_label_encoders(df: pd.DataFrame, categorical_cols: list[str]) -> dict:
    """Fit one sklearn LabelEncoder per categorical column."""
    encoders: dict[str, LabelEncoder] = {}
    for col in categorical_cols:
        enc = LabelEncoder()
        enc.fit(df[col].astype(str))
        encoders[col] = enc
    return encoders


def apply_label_encoders(df: pd.DataFrame, encoders: dict) -> pd.DataFrame:
    """
    Transform categorical columns. Unseen labels (from the web form)
    are mapped to the first known class so prediction never crashes.
    """
    out = df.copy()
    for col, enc in encoders.items():
        values = out[col].astype(str)
        known = set(enc.classes_)
        fallback = enc.classes_[0]
        values = values.where(values.isin(known), fallback)
        out[col] = enc.transform(values)
    return out


def split_xy(df: pd.DataFrame, target: str = "fraud"):
    """Separate features (X) from the fraud label (y)."""
    X = df.drop(columns=[target])
    y = df[target].astype(int)
    return X, y


def train_test_split_xy(X, y):
    """70:30 stratified split as described in the research paper."""
    return train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )


def fit_scaler(X_train: pd.DataFrame) -> StandardScaler:
    """Fit StandardScaler on the training features only (no data leakage)."""
    scaler = StandardScaler()
    scaler.fit(X_train)
    return scaler


def transform_features(X: pd.DataFrame, scaler: StandardScaler) -> pd.DataFrame:
    """Scale columns and keep the original column names."""
    scaled = scaler.transform(X)
    return pd.DataFrame(scaled, columns=X.columns, index=X.index)
