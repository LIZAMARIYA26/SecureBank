"""
Module 3 + report-friendly charts.

All figures are saved under outputs/ so they can be dropped into a BE
project report. A non-interactive matplotlib backend is forced so the
scripts work on servers without a display.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)

from config import OUTPUTS_DIR

sns.set_theme(style="whitegrid", palette="Blues_r")


def _save(fig, name: str) -> Path:
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUTS_DIR / name
    fig.tight_layout()
    fig.savefig(path, dpi=140, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Saved {path}")
    return path


def plot_fraud_count(df: pd.DataFrame) -> Path:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    counts = df["fraud"].value_counts().sort_index()
    labels = ["Genuine", "Fraud"]
    bars = ax.bar(labels, [counts.get(0, 0), counts.get(1, 0)], color=["#1b6ca8", "#c0392b"])
    ax.set_title("Fraud vs Non-Fraud Transactions")
    ax.set_ylabel("Count")
    for bar in bars:
        ax.annotate(f"{int(bar.get_height()):,}", xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    ha="center", va="bottom", fontsize=9)
    return _save(fig, "fraud_count.png")


def plot_amount_distribution(df: pd.DataFrame) -> Path:
    sample = df.sample(n=min(8000, len(df)), random_state=42)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    sns.histplot(data=sample, x="amount", hue="fraud", bins=40, kde=True, ax=ax, palette={0: "#1b6ca8", 1: "#c0392b"})
    ax.set_title("Transaction Amount Distribution")
    ax.set_xlim(0, sample["amount"].quantile(0.98))
    return _save(fig, "amount_distribution.png")


def plot_age_distribution(df: pd.DataFrame) -> Path:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    sns.countplot(data=df, x="age", hue="fraud", ax=ax, palette={0: "#1b6ca8", 1: "#c0392b"})
    ax.set_title("Age Code Distribution")
    return _save(fig, "age_distribution.png")


def plot_gender_distribution(df: pd.DataFrame) -> Path:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    sns.countplot(data=df, x="gender", hue="fraud", ax=ax, palette={0: "#1b6ca8", 1: "#c0392b"})
    ax.set_title("Gender Distribution")
    return _save(fig, "gender_distribution.png")


def plot_merchant_category(df: pd.DataFrame) -> Path:
    fig, ax = plt.subplots(figsize=(10, 5.5))
    order = df["category"].value_counts().index
    sns.countplot(data=df, y="category", order=order, ax=ax, color="#1b6ca8")
    ax.set_title("Merchant Category Distribution")
    ax.set_xlabel("Count")
    return _save(fig, "merchant_category.png")


def plot_correlation_heatmap(df: pd.DataFrame, numeric_cols: list[str]) -> Path:
    cols = [c for c in numeric_cols if c in df.columns]
    corr = df[cols].corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="Blues", ax=ax, square=False)
    ax.set_title("Correlation Heatmap")
    return _save(fig, "correlation_heatmap.png")


def plot_amount_boxplot(df: pd.DataFrame) -> Path:
    sample = df.sample(n=min(8000, len(df)), random_state=42)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    sns.boxplot(data=sample, x="fraud", y="amount", hue="fraud", ax=ax, palette=["#1b6ca8", "#c0392b"], legend=False)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Genuine", "Fraud"])
    ax.set_title("Transaction Amount Boxplot")
    ax.set_ylim(0, sample["amount"].quantile(0.98))
    return _save(fig, "amount_boxplot.png")


def plot_pairplot(df: pd.DataFrame, cols: list[str]) -> Path:
    use = [c for c in cols if c in df.columns]
    sample = df[use + ["fraud"]].sample(n=min(600, len(df)), random_state=42)
    grid = sns.pairplot(sample, hue="fraud", corner=True, palette={0: "#1b6ca8", 1: "#c0392b"})
    grid.fig.suptitle("Pairplot of Important Features", y=1.02)
    path = OUTPUTS_DIR / "pairplot.png"
    grid.savefig(path, dpi=120)
    plt.close(grid.fig)
    print(f"Saved {path}")
    return path


def plot_feature_importance(names, importances, top_n: int = 15) -> Path:
    order = np.argsort(importances)[::-1][:top_n]
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(np.array(names)[order][::-1], np.array(importances)[order][::-1], color="#1b6ca8")
    ax.set_title(f"Top {top_n} Random Forest Features")
    ax.set_xlabel("Importance")
    return _save(fig, "feature_importance.png")


def plot_confusion_matrix(y_true, y_pred) -> Path:
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5.5, 5))
    disp = ConfusionMatrixDisplay(cm, display_labels=["Genuine", "Fraud"])
    disp.plot(cmap="Blues", ax=ax, colorbar=False)
    ax.set_title("Confusion Matrix")
    return _save(fig, "confusion_matrix.png")


def plot_roc_curve(y_true, y_proba) -> Path:
    fig, ax = plt.subplots(figsize=(6, 5))
    RocCurveDisplay.from_predictions(y_true, y_proba, ax=ax, name="Random Forest")
    ax.set_title("ROC Curve")
    return _save(fig, "roc_curve.png")


def plot_pr_curve(y_true, y_proba) -> Path:
    fig, ax = plt.subplots(figsize=(6, 5))
    PrecisionRecallDisplay.from_predictions(y_true, y_proba, ax=ax, name="Random Forest")
    ax.set_title("Precision–Recall Curve")
    return _save(fig, "pr_curve.png")


def plot_model_comparison(results: dict) -> Path:
    names = list(results.keys())
    acc = [results[n]["accuracy"] for n in names]
    colors = ["#0b3d73" if n == "Random Forest" else "#7aa6d4" for n in names]
    fig, ax = plt.subplots(figsize=(8, 4.8))
    bars = ax.bar(names, acc, color=colors)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Accuracy")
    ax.set_title("Model Accuracy Comparison")
    for bar, value in zip(bars, acc):
        ax.annotate(f"{value:.3f}", xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    ha="center", va="bottom")
    return _save(fig, "accuracy_comparison.png")


def plot_tuning_comparison(before: float, after: float) -> Path:
    fig, ax = plt.subplots(figsize=(6, 4.5))
    bars = ax.bar(["Before tuning", "After tuning"], [before, after], color=["#7aa6d4", "#0b3d73"])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Accuracy")
    ax.set_title("Random Forest: GridSearchCV Effect")
    for bar, value in zip(bars, [before, after]):
        ax.annotate(f"{value:.4f}", xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    ha="center", va="bottom")
    return _save(fig, "tuning_comparison.png")
