"""
Model Building & Evaluation for Dark Pattern Detection.
Trains multiple classifiers, compares performance, selects the best,
and generates evaluation reports.
"""

import json 
import os
import sys
import warnings

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC

import config

warnings.filterwarnings("ignore")

# -- Plot Style ---------------------------------------------------------------
plt.rcParams.update({
    "figure.facecolor": "#0d1117",
    "axes.facecolor": "#161b22",
    "axes.edgecolor": "#30363d",
    "axes.labelcolor": "#c9d1d9",
    "text.color": "#c9d1d9",
    "xtick.color": "#8b949e",
    "ytick.color": "#8b949e",
    "grid.color": "#21262d",
    "figure.dpi": 150,
    "font.size": 10,
})

RISK_COLORS = {"Low": "#3fb950", "Medium": "#d29922", "High": "#f85149"}
RISK_ORDER = ["Low", "Medium", "High"]


def load_and_prepare_data():
    """Load labeled data and prepare features/target."""
    df = pd.read_csv(config.FINAL_LABELED_CSV)
    print(f"Loaded {len(df)} labeled records\n")

    # Select feature columns that exist in the data
    available_features = [c for c in config.FEATURE_COLUMNS if c in df.columns]
    missing = set(config.FEATURE_COLUMNS) - set(available_features)
    if missing:
        print(f"  [WARN] Missing features (skipped): {missing}")

    X = df[available_features].copy()
    y = df[config.TARGET_COLUMN].copy()

    # Handle any remaining NaN in features
    X = X.fillna(0)

    # Encode boolean columns as int
    for col in X.columns:
        if X[col].dtype == bool:
            X[col] = X[col].astype(int)

    # Encode target
    le = LabelEncoder()
    le.classes_ = np.array(RISK_ORDER)
    y_encoded = le.transform(y)

    print(f"  Features: {list(X.columns)}")
    print(f"  Target classes: {list(le.classes_)}")
    print(f"  Class distribution: {dict(zip(*np.unique(y_encoded, return_counts=True)))}")

    return X, y_encoded, le, available_features, df


def define_models():
    """Define models and their hyperparameter search spaces."""
    models = {
        "Logistic Regression": {
            "model": LogisticRegression(max_iter=1000, random_state=config.RANDOM_STATE),
            "params": {
                "C": [0.01, 0.1, 1, 10],
                "solver": ["lbfgs"],
            },
        },
        "Random Forest": {
            "model": RandomForestClassifier(random_state=config.RANDOM_STATE),
            "params": {
                "n_estimators": [100, 200],
                "max_depth": [5, 10, 15, None],
                "min_samples_split": [2, 5],
                "min_samples_leaf": [1, 2],
            },
        },
        "Gradient Boosting": {
            "model": GradientBoostingClassifier(random_state=config.RANDOM_STATE),
            "params": {
                "n_estimators": [100, 200],
                "max_depth": [3, 5, 7],
                "learning_rate": [0.05, 0.1, 0.2],
                "subsample": [0.8, 1.0],
            },
        },
        "SVM": {
            "model": SVC(random_state=config.RANDOM_STATE, probability=True),
            "params": {
                "C": [0.1, 1, 10],
                "kernel": ["rbf", "linear"],
                "gamma": ["scale", "auto"],
            },
        },
    }
    return models


def train_and_evaluate(X_train, X_test, y_train, y_test, le, feature_names):
    """Train all models, tune hyperparameters, and evaluate."""
    models = define_models()
    results = {}

    # Scale features for SVM and Logistic Regression
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    cv = StratifiedKFold(n_splits=config.CV_FOLDS, shuffle=True, random_state=config.RANDOM_STATE)

    print("=" * 60)
    print("MODEL TRAINING & HYPERPARAMETER TUNING")
    print("=" * 60)

    for name, spec in models.items():
        print(f"\n{'-' * 50}")
        print(f"Training: {name}")
        print(f"{'-' * 50}")

        # Use scaled data for SVM and LR
        if name in ["SVM", "Logistic Regression"]:
            X_tr, X_te = X_train_scaled, X_test_scaled
        else:
            X_tr, X_te = X_train, X_test

        # GridSearchCV
        grid = GridSearchCV(
            spec["model"],
            spec["params"],
            cv=cv,
            scoring="f1_weighted",
            n_jobs=-1,
            refit=True,
        )
        grid.fit(X_tr, y_train)

        best_model = grid.best_estimator_
        y_pred = best_model.predict(X_te)

        # Metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average="weighted", zero_division=0)
        recall = recall_score(y_test, y_pred, average="weighted", zero_division=0)
        f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

        # ROC-AUC (one-vs-rest)
        try:
            y_proba = best_model.predict_proba(X_te)
            roc_auc = roc_auc_score(y_test, y_proba, multi_class="ovr", average="weighted")
        except Exception:
            roc_auc = None

        # Cross-validation score
        cv_scores = cross_val_score(best_model, X_tr, y_train, cv=cv, scoring="f1_weighted")

        results[name] = {
            "model": best_model,
            "scaler": scaler if name in ["SVM", "Logistic Regression"] else None,
            "best_params": grid.best_params_,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_weighted": f1,
            "roc_auc": roc_auc,
            "cv_mean": cv_scores.mean(),
            "cv_std": cv_scores.std(),
            "y_pred": y_pred,
            "y_proba": y_proba if roc_auc else None,
            "confusion_matrix": confusion_matrix(y_test, y_pred),
            "classification_report": classification_report(
                y_test, y_pred, target_names=le.classes_, output_dict=True
            ),
        }

        print(f"  Best Params: {grid.best_params_}")
        print(f"  Accuracy:    {accuracy:.4f}")
        print(f"  Precision:   {precision:.4f}")
        print(f"  Recall:      {recall:.4f}")
        print(f"  F1 (weighted):{f1:.4f}")
        if roc_auc:
            print(f"  ROC-AUC:     {roc_auc:.4f}")
        print(f"  CV F1:       {cv_scores.mean():.4f} +/- {cv_scores.std():.4f}")

    return results


def select_best_model(results):
    """Select the best model based on weighted F1 score."""
    print(f"\n{'=' * 60}")
    print("MODEL COMPARISON")
    print(f"{'=' * 60}\n")

    comparison = []
    for name, res in results.items():
        comparison.append({
            "Model": name,
            "Accuracy": res["accuracy"],
            "Precision": res["precision"],
            "Recall": res["recall"],
            "F1 (weighted)": res["f1_weighted"],
            "ROC-AUC": res["roc_auc"] if res["roc_auc"] else "N/A",
            "CV F1 (mean)": res["cv_mean"],
        })

    comp_df = pd.DataFrame(comparison)
    print(comp_df.to_string(index=False))

    # Select best by F1 weighted
    best_name = max(results, key=lambda k: results[k]["f1_weighted"])
    print(f"\n* Best Model: {best_name} (F1 = {results[best_name]['f1_weighted']:.4f})")

    return best_name, comp_df


def plot_model_comparison(comp_df):
    """Plot model comparison bar chart."""
    fig, ax = plt.subplots(figsize=(12, 6))
    metrics = ["Accuracy", "Precision", "Recall", "F1 (weighted)"]
    x = np.arange(len(comp_df))
    width = 0.18
    colors = ["#58a6ff", "#3fb950", "#d29922", "#f85149"]

    for i, metric in enumerate(metrics):
        vals = comp_df[metric].values.astype(float)
        bars = ax.bar(x + i * width, vals, width, label=metric, color=colors[i],
                      edgecolor="#30363d", linewidth=0.5)

    ax.set_xlabel("Model")
    ax.set_ylabel("Score")
    ax.set_title("Model Comparison -- Performance Metrics", fontweight="bold", fontsize=14)
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(comp_df["Model"], rotation=15, ha="right")
    ax.legend(framealpha=0.8)
    ax.set_ylim(0, 1.05)

    path = os.path.join(config.MODEL_DIR, "model_comparison.png")
    fig.savefig(path, bbox_inches="tight", pad_inches=0.3)
    plt.close(fig)
    print(f"  [OK] Saved: model_comparison.png")
    return path


def plot_confusion_matrices(results, le):
    """Plot confusion matrices for all models."""
    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5))
    if n == 1:
        axes = [axes]

    for ax, (name, res) in zip(axes, results.items()):
        cm = res["confusion_matrix"]
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                    xticklabels=le.classes_, yticklabels=le.classes_,
                    cbar=False, linewidths=0.5, linecolor="#30363d")
        ax.set_title(name, fontweight="bold", fontsize=12)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")

    fig.suptitle("Confusion Matrices", fontsize=15, fontweight="bold", y=1.02)
    fig.tight_layout()
    path = os.path.join(config.MODEL_DIR, "confusion_matrices.png")
    fig.savefig(path, bbox_inches="tight", pad_inches=0.3)
    plt.close(fig)
    print(f"  [OK] Saved: confusion_matrices.png")
    return path


def plot_feature_importance(best_model, feature_names, best_name):
    """Plot feature importance for tree-based models."""
    if hasattr(best_model, "feature_importances_"):
        importances = best_model.feature_importances_
    elif hasattr(best_model, "coef_"):
        importances = np.abs(best_model.coef_).mean(axis=0)
    else:
        print("  [WARN] Model doesn't support feature importance extraction")
        return None

    idx = np.argsort(importances)[::-1]
    sorted_features = [feature_names[i] for i in idx]
    sorted_importances = importances[idx]

    fig, ax = plt.subplots(figsize=(12, 7))
    colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(sorted_features)))
    bars = ax.barh(range(len(sorted_features)), sorted_importances[::-1],
                   color=colors[::-1], edgecolor="#30363d", linewidth=0.5)
    ax.set_yticks(range(len(sorted_features)))
    ax.set_yticklabels(sorted_features[::-1])
    ax.set_xlabel("Importance")
    ax.set_title(f"Feature Importance -- {best_name}", fontweight="bold", fontsize=14)

    # Add value labels
    for i, (bar, val) in enumerate(zip(bars, sorted_importances[::-1])):
        ax.text(val + 0.005, bar.get_y() + bar.get_height()/2,
                f"{val:.3f}", va="center", fontsize=9, color="#c9d1d9")

    path = os.path.join(config.MODEL_DIR, "feature_importance.png")
    fig.savefig(path, bbox_inches="tight", pad_inches=0.3)
    plt.close(fig)
    print(f"  [OK] Saved: feature_importance.png")
    return path


def save_results(best_name, results, comp_df, le, feature_names):
    """Save the best model, metrics, and comparison data."""
    best_res = results[best_name]
    best_model = best_res["model"]

    # Save model
    model_data = {
        "model": best_model,
        "scaler": best_res["scaler"],
        "label_encoder_classes": list(le.classes_),
        "feature_names": feature_names,
        "best_params": best_res["best_params"],
        "metrics": {
            "accuracy": best_res["accuracy"],
            "precision": best_res["precision"],
            "recall": best_res["recall"],
            "f1_weighted": best_res["f1_weighted"],
            "roc_auc": best_res["roc_auc"],
        },
    }
    joblib.dump(model_data, config.BEST_MODEL_PATH)
    print(f"\n[OK] Best model saved to {config.BEST_MODEL_PATH}")

    # Save comparison table
    comp_path = os.path.join(config.MODEL_DIR, "model_comparison.csv")
    comp_df.to_csv(comp_path, index=False)
    print(f"[OK] Comparison table saved to {comp_path}")

    # Save detailed metrics JSON
    metrics_json = {}
    for name, res in results.items():
        metrics_json[name] = {
            "best_params": {k: str(v) for k, v in res["best_params"].items()},
            "accuracy": round(res["accuracy"], 4),
            "precision": round(res["precision"], 4),
            "recall": round(res["recall"], 4),
            "f1_weighted": round(res["f1_weighted"], 4),
            "roc_auc": round(res["roc_auc"], 4) if res["roc_auc"] else None,
            "cv_mean": round(res["cv_mean"], 4),
            "cv_std": round(res["cv_std"], 4),
            "classification_report": res["classification_report"],
        }

    metrics_path = os.path.join(config.MODEL_DIR, "all_model_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics_json, f, indent=2, default=str)
    print(f"[OK] All metrics saved to {metrics_path}")


def main():
    """Run the full model training pipeline."""
    print("+==========================================================+")
    print("|  Model Training & Evaluation                             |")
    print("+==========================================================+\n")

    # -- Load & Prepare ---------------------------------------------------
    X, y, le, feature_names, df = load_and_prepare_data()

    # -- Train/Test Split -------------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE, stratify=y
    )
    print(f"\n  Train: {len(X_train)} | Test: {len(X_test)}")

    # -- Train & Evaluate -------------------------------------------------
    results = train_and_evaluate(X_train, X_test, y_train, y_test, le, feature_names)

    # -- Select Best ------------------------------------------------------
    best_name, comp_df = select_best_model(results)

    # -- Visualizations ---------------------------------------------------
    print(f"\n{'=' * 60}")
    print("GENERATING EVALUATION CHARTS")
    print(f"{'=' * 60}\n")
    plot_model_comparison(comp_df)
    plot_confusion_matrices(results, le)
    plot_feature_importance(results[best_name]["model"], feature_names, best_name)

    # -- Save Results -----------------------------------------------------
    save_results(best_name, results, comp_df, le, feature_names)

    # -- Final Summary ----------------------------------------------------
    print(f"\n{'=' * 60}")
    print("TRAINING COMPLETE")
    print(f"{'=' * 60}")
    print(f"  Best Model: {best_name}")
    best = results[best_name]
    print(f"  Accuracy:    {best['accuracy']:.4f}")
    print(f"  F1 Score:    {best['f1_weighted']:.4f}")
    if best["roc_auc"]:
        print(f"  ROC-AUC:     {best['roc_auc']:.4f}")
    print(f"  Best Params: {best['best_params']}")
    print(f"\n  Classification Report:")
    print(classification_report(y_test, best["y_pred"], target_names=le.classes_))

    return results, best_name


if __name__ == "__main__":
    main()
