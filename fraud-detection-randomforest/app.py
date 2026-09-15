"""
SecureBank AI — Fraud Detection Web Application (Flask).

Run after training:

    python app.py

Then open http://127.0.0.1:5000
"""

from __future__ import annotations

import json
import random
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
from flask import Flask, flash, jsonify, redirect, render_template, request, send_file, session, url_for

import config
from utils.feature_engineering import form_to_feature_row
from utils.preprocessing import apply_label_encoders, transform_features

app = Flask(__name__)
app.secret_key = "securebank-ai-student-project-key"
app.config["MAX_CONTENT_LENGTH"] = 8 * 1024 * 1024

# ---------------------------------------------------------------------------
# Load trained artefacts once at startup (re-loaded if missing after train)
# ---------------------------------------------------------------------------
ARTEFACTS = {}


def artefacts_ready() -> bool:
    return config.MODEL_PATH.exists() and config.SCALER_PATH.exists()


def load_artefacts() -> dict:
    global ARTEFACTS
    if ARTEFACTS.get("model") is not None:
        return ARTEFACTS
    if not artefacts_ready():
        return {}
    ARTEFACTS = {
        "model": joblib.load(config.MODEL_PATH),
        "scaler": joblib.load(config.SCALER_PATH),
        "encoders": joblib.load(config.ENCODERS_PATH),
        "lookups": joblib.load(config.LOOKUPS_PATH),
        "feature_cols": joblib.load(config.FEATURE_COLUMNS_PATH),
        "metrics": json.loads(config.METRICS_PATH.read_text()) if config.METRICS_PATH.exists() else {},
        "comparison": json.loads(config.COMPARISON_PATH.read_text()) if config.COMPARISON_PATH.exists() else {},
    }
    return ARTEFACTS


def load_history() -> list:
    if not config.HISTORY_PATH.exists():
        return []
    try:
        return json.loads(config.HISTORY_PATH.read_text())
    except json.JSONDecodeError:
        return []


def save_history(rows: list) -> None:
    config.OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    config.HISTORY_PATH.write_text(json.dumps(rows[-80:], indent=2))


def get_last_session_result() -> dict | None:
    result = session.get("last_result")
    if result is not None:
        return result
    history = load_history()
    if history:
        return history[-1]
    return None


def explain_in_words(row: pd.Series, probability: float, is_fraud: bool) -> str:
    """
    Module 10 — plain-language reason a student / analyst can read.
    Uses the same signals the research paper highlights.
    """
    amount = float(row.get("amount", 0))
    risk = float(row.get("merchant_risk_score", 0))
    ratio = float(row.get("amount_to_avg_ratio", 1))
    hour = int(row.get("hour_of_tx", 12))
    category = str(row.get("category", "this merchant category"))
    reasons = []
    if amount >= 200:
        reasons.append("the transaction amount is unusually high")
    if risk >= 0.15:
        reasons.append(f"merchant category {category} has a high historical fraud rate")
    elif risk >= 0.05:
        reasons.append(f"merchant category {category} shows elevated fraud risk")
    if ratio >= 3:
        reasons.append("the amount is several times this customer's typical spend")
    if hour <= 5 or hour >= 23:
        reasons.append("the payment occurred at an unusual hour")
    if int(row.get("is_weekend", 0)) == 1:
        reasons.append("the payment was made on a weekend")
    if float(row.get("tx_velocity", 0)) >= 80:
        reasons.append("transaction velocity is high relative to frequency")

    if is_fraud:
        if not reasons:
            reasons.append("several weak risk signals combined above the decision threshold")
        joined = " and ".join(reasons[:3])
        return f"This transaction is predicted as FRAUD because {joined}."
    if probability >= 0.35:
        return (
            "This transaction is predicted as GENUINE, but the fraud probability is elevated. "
            "A fraud analyst should review it before releasing the payment."
        )
    return (
        "This transaction is predicted as GENUINE. Amount, merchant risk and "
        "customer spending pattern are consistent with normal behaviour."
    )


def score_frame(raw_rows: pd.DataFrame) -> pd.DataFrame:
    """Encode, scale, and score one or more engineered rows."""
    art = load_artefacts()
    model = art["model"]
    cols = art["feature_cols"]
    encoded = apply_label_encoders(raw_rows.copy(), art["encoders"])
    X = encoded[cols]
    X_s = transform_features(X, art["scaler"])
    proba = model.predict_proba(X_s)[:, 1]
    pred = (proba >= 0.50).astype(int)
    out = raw_rows.copy()
    out["fraud_probability"] = (proba * 100).round(2)
    out["prediction"] = pred
    out["label"] = np.where(pred == 1, "Fraudulent", "Genuine")
    out["confidence"] = (np.maximum(proba, 1 - proba) * 100).round(2)
    return out


def score_form(form: dict) -> dict:
    art = load_artefacts()
    raw = form_to_feature_row(form, art["lookups"])
    scored = score_frame(raw).iloc[0]
    is_fraud = int(scored["prediction"]) == 1
    result = {
        "is_fraud": is_fraud,
        "label": scored["label"],
        "probability": float(scored["fraud_probability"]),
        "confidence": float(scored["confidence"]),
        "explanation": explain_in_words(raw.iloc[0], float(scored["fraud_probability"]) / 100.0, is_fraud),
        "inputs": form,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    history = load_history()
    history.append(result)
    save_history(history)
    session["last_result"] = result
    session["last_form"] = form
    return result


def dashboard_stats() -> dict:
    history = load_history()
    total = len(history)
    fraud = sum(1 for r in history if r.get("is_fraud"))
    genuine = total - fraud
    pct = (fraud / total * 100) if total else 0.0
    metrics = load_artefacts().get("metrics", {}) if artefacts_ready() else {}
    comparison = load_artefacts().get("comparison", {}) if artefacts_ready() else {}
    overview = metrics.get("overview", {})
    return {
        "history": list(reversed(history[-25:])),
        "total": total,
        "fraud": fraud,
        "genuine": genuine,
        "fraud_pct": round(pct, 2),
        "model_accuracy": round(metrics.get("after_tuning", {}).get("accuracy", 0) * 100, 2),
        "model_precision": round(metrics.get("after_tuning", {}).get("precision", 0) * 100, 2),
        "model_recall": round(metrics.get("after_tuning", {}).get("recall", 0) * 100, 2),
        "model_f1": round(metrics.get("after_tuning", {}).get("f1", 0) * 100, 2),
        "model_auc": round(metrics.get("after_tuning", {}).get("roc_auc", 0) * 100, 2),
        "dataset_rows": overview.get("rows", 0),
        "dataset_fraud": overview.get("fraud_count", 0),
        "best_params": metrics.get("best_params", {}),
        "comparison": comparison,
        "feature_ranking": metrics.get("feature_ranking", [])[:12],
        "model_ready": artefacts_ready(),
    }


def form_choices() -> dict:
    if artefacts_ready():
        lookups = load_artefacts()["lookups"]
        categories = lookups.get("categories", [])
        genders = [g for g in lookups.get("genders", ["M", "F"]) if g in {"M", "F", "E", "U"}]
    else:
        categories = ["es_transportation", "es_health", "es_food", "es_travel"]
        genders = ["M", "F"]
    return {
        "categories": categories,
        "genders": genders or ["M", "F"],
        "payment_modes": config.PAYMENT_MODES,
        "device_types": config.DEVICE_TYPES,
        "locations": config.LOCATIONS,
        "spending_categories": config.SPENDING_CATEGORIES,
    }


def random_transaction() -> dict:
    """Module 11 — realistic fake payment the model can score immediately."""
    art = load_artefacts()
    lookups = art.get("lookups", {})
    categories = lookups.get("categories") or ["es_transportation", "es_leisure", "es_travel"]
    # Mix easy genuine cases with a few high-risk ones
    risky = [c for c in categories if any(k in c for k in ("leisure", "travel", "hotel", "sports"))]
    if random.random() < 0.28 and risky:
        category = random.choice(risky)
        amount = round(random.uniform(180, 1800), 2)
        hour = random.choice(list(range(0, 5)) + list(range(22, 24)))
        freq = random.randint(1, 8)
        spending = random.choice(["Low", "Medium"])
        weekend = "1"
    else:
        category = random.choice(categories)
        amount = round(random.uniform(4, 80), 2)
        hour = random.randint(8, 21)
        freq = random.randint(20, 160)
        spending = random.choice(["Medium", "High"])
        weekend = "0"
    return {
        "age": random.choice([22, 28, 34, 41, 48, 55, 63]),
        "gender": random.choice(["M", "F"]),
        "amount": amount,
        "category": category,
        "payment_mode": random.choice(config.PAYMENT_MODES),
        "hour": hour,
        "tx_frequency": freq,
        "device_type": random.choice(config.DEVICE_TYPES),
        "location": random.choice(config.LOCATIONS),
        "spending_category": spending,
        "is_weekend": weekend,
    }


@app.context_processor
def inject_globals():
    return {
        "model_ready": artefacts_ready(),
        "year": datetime.now().year,
    }


@app.route("/")
def index():
    stats = dashboard_stats()
    result = get_last_session_result()
    return render_template("index.html", stats=stats, choices=form_choices(), result=result, last_form=result["inputs"] if result else {})


@app.route("/predict", methods=["GET", "POST"])
def predict():
    if not artefacts_ready():
        flash("Train the model first: python train_model.py", "error")
        return redirect(url_for("index"))

    result = get_last_session_result()
    if request.method == "POST":
        required = ["amount", "category", "gender"]
        missing = [f for f in required if not str(request.form.get(f, "")).strip()]
        if missing:
            flash("Please fill in amount, merchant category and gender.", "error")
        else:
            try:
                form = {k: request.form.get(k) for k in request.form}
                result = score_form(form)
                flash("Prediction complete.", "success")
            except Exception as exc:
                flash(f"Could not score this transaction: {exc}", "error")
    return render_template(
        "prediction.html",
        result=result,
        choices=form_choices(),
        last_form=result["inputs"] if result else session.get("last_form", {}),
    )


@app.route("/generate", methods=["GET", "POST"])
def generate():
    if not artefacts_ready():
        flash("Train the model first: python train_model.py", "error")
        return redirect(url_for("index"))
    txn = random_transaction()
    result = score_form(txn)
    flash("Random transaction generated and scored.", "success")
    return render_template("prediction.html", result=result, choices=form_choices(), generated=True)


@app.route("/dashboard")
def dashboard():
    return render_template("dashboard.html", stats=dashboard_stats())


@app.route("/upload", methods=["POST"])
def upload():
    if not artefacts_ready():
        flash("Train the model first: python train_model.py", "error")
        return redirect(url_for("index"))
    file = request.files.get("file")
    if not file or file.filename == "":
        flash("Choose a CSV file first.", "error")
        return redirect(url_for("predict"))
    try:
        df = pd.read_csv(file)
    except Exception:
        flash("That file is not a readable CSV.", "error")
        return redirect(url_for("predict"))

    art = load_artefacts()
    rows = []
    for _, rec in df.iterrows():
        form = rec.fillna("").to_dict()
        raw = form_to_feature_row(form, art["lookups"])
        rows.append(raw)
    batch = pd.concat(rows, ignore_index=True)
    scored = score_frame(batch)
    # Store last batch for download
    scored.to_csv(config.OUTPUTS_DIR / "batch_predictions.csv", index=False)
    history = load_history()
    for _, rec in scored.iterrows():
        history.append(
            {
                "is_fraud": int(rec["prediction"]) == 1,
                "label": rec["label"],
                "probability": float(rec["fraud_probability"]),
                "confidence": float(rec["confidence"]),
                "explanation": "Batch upload",
                "inputs": rec.drop(labels=["fraud_probability", "prediction", "label", "confidence"]).to_dict(),
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            }
        )
    save_history(history)
    flash(f"Scored {len(scored)} transactions. Download is ready on this page.", "success")
    preview = scored.head(25).to_dict(orient="records")
    return render_template(
        "prediction.html",
        result=None,
        choices=form_choices(),
        batch_preview=preview,
        batch_count=len(scored),
        batch_fraud=int((scored["prediction"] == 1).sum()),
    )


@app.route("/download")
def download():
    path = config.OUTPUTS_DIR / "batch_predictions.csv"
    if not path.exists():
        flash("No batch predictions yet. Upload a CSV first.", "error")
        return redirect(url_for("predict"))
    return send_file(path, as_attachment=True, download_name="fraud_predictions.csv")


@app.route("/api/stats")
def api_stats():
    return jsonify(dashboard_stats())


if __name__ == "__main__":
    load_artefacts()
    # Port 5000 as required by the project specification
    app.run(host="0.0.0.0", port=5000, debug=False)
