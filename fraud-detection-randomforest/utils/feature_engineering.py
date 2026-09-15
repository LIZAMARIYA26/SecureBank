"""
Module 4 — Feature engineering inspired by the JETIR research paper.

Each new column is documented below. Customer-level statistics are
computed from the full dataset first, then attached to every row so
the Random Forest can see behaviour, not just a single payment.
"""

from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd

from config import DEVICE_TYPES, LOCATIONS, PAYMENT_MODES, SPENDING_CATEGORIES


def _stable_index(text: str, modulo: int) -> int:
    """Turn a string into a stable integer (same input → same output)."""
    digest = hashlib.md5(str(text).encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % modulo


def add_transaction_frequency(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transaction Frequency
    ---------------------
    How many payments this customer made in the BankSim window.
    Fraudsters often burst many payments in a short period, so a high
    (or unusually low) frequency can be a useful signal.
    """
    out = df.copy()
    freq = out.groupby("customer")["amount"].transform("count")
    out["tx_frequency"] = freq.astype(int)
    return out


def add_average_transaction_amount(df: pd.DataFrame) -> pd.DataFrame:
    """
    Average Transaction Amount
    --------------------------
    Mean amount spent by this customer. A single payment that is far
    above the customer's own average is more suspicious than the same
    amount from a high-spending customer.
    """
    out = df.copy()
    out["avg_tx_amount"] = out.groupby("customer")["amount"].transform("mean")
    return out


def add_merchant_risk_score(df: pd.DataFrame) -> pd.DataFrame:
    """
    Merchant Risk Score
    -------------------
    Historical fraud rate of the merchant (0 to 1).
    Leisure / travel merchants in BankSim have much higher fraud rates
    than transportation, so this is one of the strongest features.
    """
    out = df.copy()
    risk = out.groupby("merchant")["fraud"].transform("mean")
    out["merchant_risk_score"] = risk.astype(float)
    return out


def add_hour_of_transaction(df: pd.DataFrame) -> pd.DataFrame:
    """
    Hour of Transaction
    -------------------
    BankSim's `step` is a day index (0–179). We derive a 0–23 hour so
    the web form (which asks for hour) matches the model. Combining
    step with the customer id spreads hours realistically.
    """
    out = df.copy()
    customer_shift = out["customer"].astype(str).map(lambda c: _stable_index(c, 24))
    out["hour_of_tx"] = ((out["step"].astype(int) + customer_shift) % 24).astype(int)
    return out


def add_weekend_flag(df: pd.DataFrame) -> pd.DataFrame:
    """
    Weekend Transaction
    -------------------
    1 if the simulated day-of-week is Saturday or Sunday.
    Some fraud patterns concentrate on weekends when monitoring is lighter.
    """
    out = df.copy()
    out["is_weekend"] = (out["step"].astype(int) % 7 >= 5).astype(int)
    return out


def add_customer_spending_category(df: pd.DataFrame) -> pd.DataFrame:
    """
    Customer Spending Category
    --------------------------
    Bins the customer's average ticket into Low / Medium / High / Premium.
    Gives the model a coarse lifestyle segment instead of a raw rupee value.
    """
    out = df.copy()
    if "avg_tx_amount" not in out.columns:
        out = add_average_transaction_amount(out)
    # Quantile bins are data-driven and work even if amounts change
    try:
        out["spending_category"] = pd.qcut(
            out["avg_tx_amount"],
            q=4,
            labels=SPENDING_CATEGORIES,
            duplicates="drop",
        ).astype(str)
    except ValueError:
        out["spending_category"] = "Medium"
    return out


def add_transaction_velocity(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transaction Velocity
    --------------------
    Amount spent per historical transaction (amount / frequency).
    A sudden large payment from a customer who usually makes many tiny
    ones produces a high velocity spike.
    """
    out = df.copy()
    freq = out["tx_frequency"].replace(0, 1)
    out["tx_velocity"] = (out["amount"] / freq).astype(float)
    return out


def add_amount_to_avg_ratio(df: pd.DataFrame) -> pd.DataFrame:
    """How large this payment is compared with the customer's usual spend."""
    out = df.copy()
    avg = out["avg_tx_amount"].replace(0, np.nan)
    out["amount_to_avg_ratio"] = (out["amount"] / avg).fillna(1.0)
    return out


def add_synthetic_context_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Payment Mode, Device Type, Location
    -----------------------------------
    BankSim does not include these columns, but the web application does.
    We assign them deterministically from customer + category so they are
    stable and the trained model can accept the same fields from the form.
    """
    out = df.copy()
    keys_pay = (out["customer"].astype(str) + out["category"].astype(str)).map(
        lambda k: _stable_index(k, len(PAYMENT_MODES))
    )
    out["payment_mode"] = np.where(out["amount"] >= 400, "Credit Card", pd.Series(PAYMENT_MODES).take(keys_pay).to_numpy())
    keys_dev = out["customer"].astype(str).map(lambda c: _stable_index(c + "device", len(DEVICE_TYPES)))
    devices = pd.Series(DEVICE_TYPES).take(keys_dev).to_numpy()
    out["device_type"] = np.where(out["category"].astype(str).str.contains("transportation", na=False), "POS Terminal", devices)
    keys_loc = out["customer"].astype(str).map(lambda c: _stable_index(c + "city", len(LOCATIONS)))
    out["location"] = pd.Series(LOCATIONS).take(keys_loc).to_numpy()
    return out


def add_category_risk_score(df: pd.DataFrame) -> pd.DataFrame:
    """Fraud rate of the merchant *category* (used when a new merchant appears)."""
    out = df.copy()
    out["category_risk_score"] = out.groupby("category")["fraud"].transform("mean")
    return out


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Run every feature step in a safe order and return the enriched table."""
    out = df.copy()
    out = add_transaction_frequency(out)
    out = add_average_transaction_amount(out)
    out = add_merchant_risk_score(out)
    out = add_category_risk_score(out)
    out = add_hour_of_transaction(out)
    out = add_weekend_flag(out)
    out = add_customer_spending_category(out)
    out = add_transaction_velocity(out)
    out = add_amount_to_avg_ratio(out)
    out = add_synthetic_context_features(out)
    return out


def build_lookups(engineered: pd.DataFrame) -> dict:
    """
    Store category / merchant statistics so the Flask app can score a
    brand-new transaction that has no customer history.
    """
    merchant_risk = (
        engineered.groupby("merchant")["fraud"].mean().to_dict() if "fraud" in engineered else {}
    )
    category_risk = engineered.groupby("category")["merchant_risk_score"].mean().to_dict()
    if "category_risk_score" in engineered.columns:
        category_risk = engineered.groupby("category")["category_risk_score"].mean().to_dict()

    spending_avg = engineered.groupby("spending_category")["avg_tx_amount"].mean().to_dict()

    return {
        "merchant_risk": merchant_risk,
        "category_risk": {str(k): float(v) for k, v in category_risk.items()},
        "global_fraud_rate": float(engineered["fraud"].mean()) if "fraud" in engineered else 0.01,
        "spending_avg": {str(k): float(v) for k, v in spending_avg.items()},
        "categories": sorted(engineered["category"].astype(str).unique().tolist()),
        "genders": sorted(engineered["gender"].astype(str).unique().tolist()),
        "payment_modes": PAYMENT_MODES,
        "device_types": DEVICE_TYPES,
        "locations": LOCATIONS,
        "spending_categories": SPENDING_CATEGORIES,
        "amount_mean": float(engineered["amount"].mean()),
        "amount_std": float(engineered["amount"].std() or 1.0),
        "freq_median": float(engineered["tx_frequency"].median()),
    }


def form_to_feature_row(form: dict, lookups: dict) -> pd.DataFrame:
    """
    Convert a web-form dictionary into a one-row DataFrame that uses the
    same column names as the training matrix (before encoding/scaling).
    """
    amount = float(form.get("amount", 0) or 0)
    freq = int(float(form.get("tx_frequency", lookups.get("freq_median", 10)) or 10))
    hour = int(float(form.get("hour", 12) or 12)) % 24
    category = str(form.get("category", "es_transportation"))
    spending = str(form.get("spending_category", "Medium"))
    age_raw = form.get("age", 2)

    # Web form may send a real age (18–80) or a BankSim code (0–6)
    try:
        age_val = float(age_raw)
    except (TypeError, ValueError):
        age_val = 2
    if age_val >= 18:
        if age_val < 26:
            age_code = 1
        elif age_val < 36:
            age_code = 2
        elif age_val < 46:
            age_code = 3
        elif age_val < 56:
            age_code = 4
        elif age_val < 66:
            age_code = 5
        else:
            age_code = 6
    else:
        age_code = int(age_val) if age_val >= 0 else 2

    category_risk = float(lookups.get("category_risk", {}).get(category, lookups.get("global_fraud_rate", 0.01)))
    avg_tx = float(lookups.get("spending_avg", {}).get(spending, amount if amount else 30.0))
    is_weekend = int(str(form.get("is_weekend", "0")) in {"1", "true", "True", "yes", "Yes", "Weekend"})

    row = {
        "age": age_code,
        "gender": str(form.get("gender", "M")),
        "category": category,
        "amount": amount,
        "tx_frequency": freq,
        "avg_tx_amount": avg_tx,
        "merchant_risk_score": category_risk,
        "category_risk_score": category_risk,
        "hour_of_tx": hour,
        "is_weekend": is_weekend,
        "spending_category": spending,
        "tx_velocity": amount / max(freq, 1),
        "amount_to_avg_ratio": amount / max(avg_tx, 1e-6),
        "payment_mode": str(form.get("payment_mode", "Debit Card")),
        "device_type": str(form.get("device_type", "Mobile")),
        "location": str(form.get("location", "Madrid")),
    }
    return pd.DataFrame([row])


def age_code_to_years(code: int) -> int:
    """Representative age in years for a BankSim age code (used by the simulator)."""
    mapping = {0: 16, 1: 22, 2: 30, 3: 40, 4: 50, 5: 60, 6: 72, -1: 35}
    return mapping.get(int(code), 35)
