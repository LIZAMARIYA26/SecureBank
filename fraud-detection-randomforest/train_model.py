"""
Train the BankSim fraud-detection Random Forest.

Run from the project root:

    python train_model.py

This script covers Modules 1–8, 10 (SHAP), 12 and 14 from the project
specification and writes artefacts into models/ and outputs/.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

import config
from utils.feature_engineering import build_lookups, engineer_features
from utils.preprocessing import (
    apply_label_encoders,
    dataset_overview,
    encode_age,
    fit_label_encoders,
    fit_scaler,
    handle_missing_values,
    load_banksim,
    print_dataset_report,
    remove_duplicates,
    split_xy,
    train_test_split_xy,
    transform_features,
    undersample_majority,
)
from utils.visualization import (
    plot_age_distribution,
    plot_amount_boxplot,
    plot_amount_distribution,
    plot_confusion_matrix,
    plot_correlation_heatmap,
    plot_feature_importance,
    plot_fraud_count,
    plot_gender_distribution,
    plot_merchant_category,
    plot_model_comparison,
    plot_pairplot,
    plot_pr_curve,
    plot_roc_curve,
    plot_tuning_comparison,
)

warnings.filterwarnings("ignore")

# Columns the classifier actually sees (ids are useful for engineering only)
MODEL_CATEGORICAL = ["gender", "category", "spending_category", "payment_mode", "device_type", "location"]
MODEL_NUMERIC = [
    "age",
    "amount",
    "tx_frequency",
    "avg_tx_amount",
    "merchant_risk_score",
    "category_risk_score",
    "hour_of_tx",
    "is_weekend",
    "tx_velocity",
    "amount_to_avg_ratio",
]
DROP_AFTER_ENGINEERING = ["step", "customer", "zipcodeOri", "merchant", "zipMerchant"]


def evaluate(name: str, y_true, y_pred, y_proba=None) -> dict:
    """Compute the metrics required by Module 8."""
    metrics = {
        "model": name,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }
    if y_proba is not None:
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_proba))
    return metrics


def maybe_shap(model, X_sample: pd.DataFrame) -> None:
    """Module 10 — save a SHAP summary plot when the library is available."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import shap
    except ImportError:
        print("SHAP is not installed — skipping Module 10 plot.")
        return

    print("MODULE 10 — SHAP explainability (sample of rows)...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)

    # Binary RF returns a list [class0, class1] on some sklearn versions
    if isinstance(shap_values, list):
        values = shap_values[1]
    else:
        values = shap_values
        if getattr(values, "ndim", 1) == 3:
            values = values[:, :, 1]

    shap.summary_plot(values, X_sample, show=False, max_display=12)
    path = config.OUTPUTS_DIR / "shap_summary.png"
    plt.tight_layout()
    plt.savefig(path, dpi=140, bbox_inches="tight")
    plt.close()
    print(f"Saved {path}")

    shap.summary_plot(values, X_sample, plot_type="bar", show=False, max_display=12)
    path_bar = config.OUTPUTS_DIR / "shap_feature_impact.png"
    plt.tight_layout()
    plt.savefig(path_bar, dpi=140, bbox_inches="tight")
    plt.close()
    print(f"Saved {path_bar}")


def main() -> None:
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    config.OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Module 1 — load and inspect
    # ------------------------------------------------------------------
    df = load_banksim()
    print_dataset_report(df)
    overview = dataset_overview(df)

    # ------------------------------------------------------------------
    # Module 2 — preprocess
    # ------------------------------------------------------------------
    print("=" * 70)
    print("MODULE 2 — DATA PREPROCESSING")
    print("=" * 70)
    df = remove_duplicates(df)
    df = handle_missing_values(df)
    df["age"] = encode_age(df["age"])
    df["fraud"] = df["fraud"].astype(int)

    # ------------------------------------------------------------------
    # Module 3 — EDA on a sample of the full table (before undersampling)
    # ------------------------------------------------------------------
    print("=" * 70)
    print("MODULE 3 — EXPLORATORY DATA ANALYSIS")
    print("=" * 70)
    eda = df.sample(n=min(25000, len(df)), random_state=config.RANDOM_STATE)
    plot_fraud_count(df)
    plot_amount_distribution(eda)
    plot_age_distribution(eda)
    plot_gender_distribution(eda)
    plot_merchant_category(eda)
    plot_amount_boxplot(eda)

    # ------------------------------------------------------------------
    # Module 4 — feature engineering (needs the fraud column)
    # ------------------------------------------------------------------
    print("=" * 70)
    print("MODULE 4 — FEATURE ENGINEERING")
    print("=" * 70)
    engineered = engineer_features(df)
    lookups = build_lookups(engineered)
    print("Created features: tx_frequency, avg_tx_amount, merchant_risk_score,")
    print("category_risk_score, hour_of_tx, is_weekend, spending_category,")
    print("tx_velocity, amount_to_avg_ratio, payment_mode, device_type, location.")

    # Heatmap / pairplot after numeric features exist
    plot_correlation_heatmap(
        engineered.sample(n=min(15000, len(engineered)), random_state=config.RANDOM_STATE),
        MODEL_NUMERIC + ["fraud"],
    )
    plot_pairplot(engineered, ["amount", "tx_frequency", "merchant_risk_score", "tx_velocity", "hour_of_tx"])

    # Undersample majority class for faster, more balanced training
    train_pool = undersample_majority(
        engineered, max_genuine=config.MAX_GENUINE_SAMPLES, random_state=config.RANDOM_STATE
    )
    print(
        f"Training pool size: {len(train_pool):,} "
        f"(fraud={int((train_pool.fraud == 1).sum()):,}, "
        f"genuine={int((train_pool.fraud == 0).sum()):,})"
    )

    model_df = train_pool.drop(columns=[c for c in DROP_AFTER_ENGINEERING if c in train_pool.columns])
    encoders = fit_label_encoders(model_df, MODEL_CATEGORICAL)
    encoded = apply_label_encoders(model_df, encoders)

    feature_cols = MODEL_NUMERIC + MODEL_CATEGORICAL
    X = encoded[feature_cols]
    y = encoded["fraud"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split_xy(X, y)
    scaler = fit_scaler(X_train)
    X_train_s = transform_features(X_train, scaler)
    X_test_s = transform_features(X_test, scaler)

    # ------------------------------------------------------------------
    # Module 6 — baseline Random Forest
    # ------------------------------------------------------------------
    print("=" * 70)
    print("MODULE 6 — RANDOM FOREST (before tuning)")
    print("=" * 70)
    rf_before = RandomForestClassifier(**config.RF_PARAMS)
    rf_before.fit(X_train_s, y_train)
    pred_before = rf_before.predict(X_test_s)
    proba_before = rf_before.predict_proba(X_test_s)[:, 1]
    metrics_before = evaluate("Random Forest (before)", y_test, pred_before, proba_before)
    print(metrics_before)

    # ------------------------------------------------------------------
    # Module 5 — feature importance
    # ------------------------------------------------------------------
    print("=" * 70)
    print("MODULE 5 — FEATURE SELECTION")
    print("=" * 70)
    importances = rf_before.feature_importances_
    ranking = sorted(zip(feature_cols, importances), key=lambda t: t[1], reverse=True)
    print("Top 15 features:")
    for name, score in ranking[:15]:
        print(f"  {name:24s}  {score:.4f}")
    plot_feature_importance(feature_cols, importances, top_n=15)

    # ------------------------------------------------------------------
    # Module 7 — GridSearchCV
    # ------------------------------------------------------------------
    print("=" * 70)
    print("MODULE 7 — HYPERPARAMETER TUNING (GridSearchCV)")
    print("=" * 70)
    search = GridSearchCV(
        RandomForestClassifier(
            random_state=config.RANDOM_STATE,
            n_jobs=-1,
            class_weight="balanced",
        ),
        param_grid=config.PARAM_GRID,
        scoring="f1",
        cv=2,
        n_jobs=-1,
        verbose=1,
    )
    search.fit(X_train_s, y_train)
    print("Best parameters:", search.best_params_)
    rf = search.best_estimator_

    pred = rf.predict(X_test_s)
    proba = rf.predict_proba(X_test_s)[:, 1]
    metrics_after = evaluate("Random Forest", y_test, pred, proba)
    print("After tuning:", metrics_after)
    plot_tuning_comparison(metrics_before["accuracy"], metrics_after["accuracy"])

    # ------------------------------------------------------------------
    # Module 8 — evaluation plots
    # ------------------------------------------------------------------
    print("=" * 70)
    print("MODULE 8 — MODEL EVALUATION")
    print("=" * 70)
    print(classification_report(y_test, pred, target_names=["Genuine", "Fraud"]))
    plot_confusion_matrix(y_test, pred)
    plot_roc_curve(y_test, proba)
    plot_pr_curve(y_test, proba)

    # ------------------------------------------------------------------
    # Module 12 — compare with other classifiers
    # ------------------------------------------------------------------
    print("=" * 70)
    print("MODULE 12 — PERFORMANCE COMPARISON")
    print("=" * 70)
    competitors = {
        "Logistic Regression": LogisticRegression(max_iter=400, class_weight="balanced"),
        "Decision Tree": DecisionTreeClassifier(max_depth=12, class_weight="balanced", random_state=config.RANDOM_STATE),
        "KNN": KNeighborsClassifier(n_neighbors=7),
    }
    comparison = {"Random Forest": metrics_after}
    # KNN is slower — fit on a cap of training rows
    knn_cap = min(8000, len(X_train_s))
    for name, clf in competitors.items():
        if name == "KNN":
            clf.fit(X_train_s.iloc[:knn_cap], y_train.iloc[:knn_cap])
        else:
            clf.fit(X_train_s, y_train)
        y_hat = clf.predict(X_test_s)
        y_p = clf.predict_proba(X_test_s)[:, 1] if hasattr(clf, "predict_proba") else None
        comparison[name] = evaluate(name, y_test, y_hat, y_p)
        print(name, comparison[name])

    # Save a readable model comparison table for reports and spreadsheets.
    comparison_df = pd.DataFrame.from_dict(comparison, orient="index").reset_index(drop=True)
    preferred_columns = ["model", "accuracy", "precision", "recall", "f1", "roc_auc"]
    comparison_df = comparison_df[
        [column for column in preferred_columns if column in comparison_df.columns]
    ]
    comparison_df.to_csv(config.COMPARISON_PATH.with_suffix(".csv"), index=False)
    print(f"Comparison CSV saved to: {config.COMPARISON_PATH.with_suffix('.csv')}")

    plot_model_comparison(comparison)

    # ------------------------------------------------------------------
    # Module 10 — SHAP
    # ------------------------------------------------------------------
    shap_sample = X_test_s.sample(n=min(180, len(X_test_s)), random_state=config.RANDOM_STATE)
    maybe_shap(rf, shap_sample)

    # ------------------------------------------------------------------
    # Persist artefacts for the Flask app
    # ------------------------------------------------------------------
    joblib.dump(rf, config.MODEL_PATH)
    joblib.dump(scaler, config.SCALER_PATH)
    joblib.dump(encoders, config.ENCODERS_PATH)
    joblib.dump(lookups, config.LOOKUPS_PATH)
    joblib.dump(feature_cols, config.FEATURE_COLUMNS_PATH)
    joblib.dump(shap_sample, config.SHAP_BACKGROUND_PATH)

    payload = {
        "overview": {k: v for k, v in overview.items() if k != "head"},
        "best_params": search.best_params_,
        "before_tuning": metrics_before,
        "after_tuning": metrics_after,
        "feature_ranking": [{"feature": n, "importance": float(s)} for n, s in ranking],
        "classification_report": classification_report(
            y_test, pred, target_names=["Genuine", "Fraud"], output_dict=True
        ),
    }
    config.METRICS_PATH.write_text(json.dumps(payload, indent=2))
    config.COMPARISON_PATH.write_text(json.dumps(comparison, indent=2))

    print("=" * 70)
    print("TRAINING COMPLETE")
    print(f"Model saved to {config.MODEL_PATH}")
    print("Start the web app with:  python app.py")
    print("=" * 70)


if __name__ == "__main__":
    main()
