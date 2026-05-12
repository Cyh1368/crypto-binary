from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.calibration import calibration_curve
from sklearn.metrics import confusion_matrix

from src.utils.io import ensure_dir


def save_confusion_matrix(y_true: pd.Series, prob_up: pd.Series, path: str | Path) -> None:
    ensure_dir(Path(path).parent)
    cm = confusion_matrix(y_true, (prob_up >= 0.5).astype(int), labels=[0, 1])
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Down", "Up"], yticklabels=["Down", "Up"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def save_prediction_actual_heatmap(predictions: pd.DataFrame, path: str | Path) -> None:
    ensure_dir(Path(path).parent)
    plt.figure(figsize=(7, 5))
    plt.hist2d(predictions["prob_up"], predictions["future_return_15m"], bins=30, cmap="viridis")
    plt.colorbar(label="Count")
    plt.xlabel("Predicted probability up")
    plt.ylabel("Realized 15m log return")
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def save_feature_importance(importance: pd.DataFrame, path: str | Path) -> None:
    ensure_dir(Path(path).parent)
    top = importance.sort_values("gain", ascending=False).head(30)
    plt.figure(figsize=(8, 8))
    sns.barplot(data=top, x="gain", y="feature", color="#2f7ed8")
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def save_shap_importance(shap_importance: pd.DataFrame, path: str | Path) -> None:
    ensure_dir(Path(path).parent)
    top = shap_importance.sort_values("mean_abs_shap", ascending=False).head(30)
    plt.figure(figsize=(8, 8))
    sns.barplot(data=top, x="mean_abs_shap", y="feature", color="#44aa77")
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def save_calibration_curve(y_true: pd.Series, prob_up: pd.Series, path: str | Path) -> None:
    ensure_dir(Path(path).parent)
    frac_pos, mean_pred = calibration_curve(y_true, prob_up, n_bins=10, strategy="quantile")
    plt.figure(figsize=(6, 5))
    plt.plot(mean_pred, frac_pos, marker="o", label="Model")
    plt.plot([0, 1], [0, 1], linestyle="--", color="black", label="Perfect")
    plt.xlabel("Mean predicted probability")
    plt.ylabel("Realized frequency")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def save_pnl_curve(backtest: pd.DataFrame, path: str | Path) -> None:
    ensure_dir(Path(path).parent)
    plt.figure(figsize=(9, 4))
    backtest["cumulative_return"].plot()
    plt.xlabel("Time")
    plt.ylabel("Cumulative log return")
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def save_regime_performance(regime: dict[str, dict[str, float]], path: str | Path) -> None:
    ensure_dir(Path(path).parent)
    frame = pd.DataFrame(regime).T
    plt.figure(figsize=(9, 5))
    if frame.empty:
        plt.text(0.5, 0.5, "Insufficient regime samples", ha="center", va="center")
        plt.axis("off")
    else:
        frame["balanced_accuracy"].sort_values().plot(kind="barh")
        plt.xlabel("Balanced accuracy")
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()
