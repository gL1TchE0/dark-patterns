"""
Exploratory Data Analysis (EDA) for Dark Patterns Detection.
Generates comprehensive visualizations of the labeled dataset.
"""

import os
import sys
import warnings

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import config

warnings.filterwarnings("ignore")

# -- Style Setup --------------------------------------------------------------
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
    "font.family": "sans-serif",
})

# Color palette for risk levels
RISK_COLORS = {"Low": "#3fb950", "Medium": "#d29922", "High": "#f85149"}
RISK_ORDER = ["Low", "Medium", "High"]
PLATFORM_COLORS = {"Flipkart": "#2874f0", "Amazon": "#ff9900"}


def save_plot(fig, name):
    """Save figure to charts directory."""
    path = os.path.join(config.CHARTS_DIR, f"{name}.png")
    fig.savefig(path, bbox_inches="tight", pad_inches=0.3)
    plt.close(fig)
    print(f"  [OK] Saved: {name}.png")
    return path


def plot_risk_distribution(df):
    """1. Overall manipulation risk distribution."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Bar chart
    counts = df["manipulation_risk"].value_counts().reindex(RISK_ORDER)
    colors = [RISK_COLORS[r] for r in RISK_ORDER]
    bars = axes[0].bar(RISK_ORDER, counts.values, color=colors, edgecolor="#30363d", linewidth=0.5)
    for bar, count in zip(bars, counts.values):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                     str(count), ha="center", va="bottom", fontweight="bold", color="#c9d1d9")
    axes[0].set_title("Manipulation Risk Distribution", fontweight="bold", fontsize=13)
    axes[0].set_ylabel("Number of Listings")
    axes[0].set_xlabel("Risk Level")

    # Pie chart
    axes[1].pie(counts.values, labels=RISK_ORDER, colors=colors, autopct="%1.1f%%",
                startangle=90, textprops={"color": "#c9d1d9", "fontweight": "bold"},
                wedgeprops={"edgecolor": "#0d1117", "linewidth": 2})
    axes[1].set_title("Risk Level Proportions", fontweight="bold", fontsize=13)

    fig.suptitle("Manipulation Risk -- Overall Distribution", fontsize=15, fontweight="bold", y=1.02)
    return save_plot(fig, "01_risk_distribution")


def plot_risk_by_platform(df):
    """2. Risk distribution by platform."""
    fig, ax = plt.subplots(figsize=(10, 6))
    ct = pd.crosstab(df["platform"], df["manipulation_risk"])[RISK_ORDER]
    ct.plot(kind="bar", ax=ax, color=[RISK_COLORS[r] for r in RISK_ORDER],
            edgecolor="#30363d", linewidth=0.5)
    ax.set_title("Risk Distribution by Platform", fontweight="bold", fontsize=14)
    ax.set_xlabel("Platform")
    ax.set_ylabel("Number of Listings")
    ax.legend(title="Risk Level", framealpha=0.8)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
    return save_plot(fig, "02_risk_by_platform")


def plot_risk_by_category(df):
    """3. Risk distribution by category."""
    fig, ax = plt.subplots(figsize=(12, 6))
    ct = pd.crosstab(df["category"], df["manipulation_risk"])[RISK_ORDER]
    ct.plot(kind="bar", ax=ax, color=[RISK_COLORS[r] for r in RISK_ORDER],
            edgecolor="#30363d", linewidth=0.5)
    ax.set_title("Risk Distribution by Product Category", fontweight="bold", fontsize=14)
    ax.set_xlabel("Category")
    ax.set_ylabel("Number of Listings")
    ax.legend(title="Risk Level", framealpha=0.8)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=15, ha="right")
    return save_plot(fig, "03_risk_by_category")


def plot_discount_distribution(df):
    """4. Discount % distribution by risk level."""
    fig, ax = plt.subplots(figsize=(12, 6))
    for risk in RISK_ORDER:
        subset = df[df["manipulation_risk"] == risk]["discount_percentage"]
        ax.hist(subset, bins=30, alpha=0.6, label=risk, color=RISK_COLORS[risk], edgecolor="#30363d")
    ax.axvline(x=70, color="#f85149", linestyle="--", alpha=0.7, label="Extreme Threshold (70%)")
    ax.set_title("Discount Percentage Distribution by Risk Level", fontweight="bold", fontsize=14)
    ax.set_xlabel("Discount (%)")
    ax.set_ylabel("Frequency")
    ax.legend(framealpha=0.8)
    return save_plot(fig, "04_discount_distribution")


def plot_rating_vs_discount(df):
    """5. Scatter: Rating vs Discount colored by risk."""
    fig, ax = plt.subplots(figsize=(12, 7))
    for risk in RISK_ORDER:
        subset = df[df["manipulation_risk"] == risk]
        ax.scatter(subset["discount_percentage"], subset["rating"],
                   c=RISK_COLORS[risk], label=risk, alpha=0.5, s=30, edgecolors="#30363d", linewidths=0.3)
    ax.set_title("Rating vs. Discount -- Colored by Risk Level", fontweight="bold", fontsize=14)
    ax.set_xlabel("Discount (%)")
    ax.set_ylabel("Rating")
    ax.legend(title="Risk Level", framealpha=0.8)
    ax.axhline(y=4.0, color="#8b949e", linestyle=":", alpha=0.5)
    ax.axvline(x=70, color="#f85149", linestyle=":", alpha=0.5)
    return save_plot(fig, "05_rating_vs_discount")


def plot_correlation_heatmap(df):
    """6. Correlation heatmap of numeric features."""
    numeric_cols = [c for c in config.FEATURE_COLUMNS if c in df.columns]
    corr = df[numeric_cols].corr()

    fig, ax = plt.subplots(figsize=(14, 10))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="RdYlGn_r",
                center=0, ax=ax, square=True, linewidths=0.5,
                cbar_kws={"shrink": 0.8, "label": "Correlation"},
                annot_kws={"size": 8})
    ax.set_title("Feature Correlation Heatmap", fontweight="bold", fontsize=14, pad=15)
    return save_plot(fig, "06_correlation_heatmap")


def plot_promotional_intensity(df):
    """7. Promotional intensity by risk level."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Box plot
    data_boxes = [df[df["manipulation_risk"] == r]["promotional_intensity"] for r in RISK_ORDER]
    bp = axes[0].boxplot(data_boxes, tick_labels=RISK_ORDER, patch_artist=True, notch=True)
    for patch, risk in zip(bp["boxes"], RISK_ORDER):
        patch.set_facecolor(RISK_COLORS[risk])
        patch.set_alpha(0.7)
    axes[0].set_title("Promotional Intensity by Risk", fontweight="bold", fontsize=13)
    axes[0].set_ylabel("Promotional Intensity Score")

    # Stacked view: what promotional cues are most common per risk
    promo_features = ["is_sponsored", "has_coupon", "has_scarcity_message"]
    promo_means = df.groupby("manipulation_risk")[promo_features].mean().reindex(RISK_ORDER)
    promo_means.plot(kind="bar", stacked=True, ax=axes[1],
                     color=["#58a6ff", "#d29922", "#f85149"], edgecolor="#30363d")
    axes[1].set_title("Promotional Cue Prevalence by Risk", fontweight="bold", fontsize=13)
    axes[1].set_ylabel("Proportion of Listings")
    axes[1].set_xlabel("Risk Level")
    axes[1].legend(title="Cue Type", framealpha=0.8)
    axes[1].set_xticklabels(RISK_ORDER, rotation=0)

    fig.suptitle("Promotional Intensity Analysis", fontsize=15, fontweight="bold", y=1.02)
    return save_plot(fig, "07_promotional_intensity")


def plot_trust_vs_urgency(df):
    """8. Trust Score vs Urgency Score scatter."""
    fig, ax = plt.subplots(figsize=(12, 7))
    for risk in RISK_ORDER:
        subset = df[df["manipulation_risk"] == risk]
        ax.scatter(subset["urgency_score"], subset["trust_score"],
                   c=RISK_COLORS[risk], label=risk, alpha=0.5, s=35,
                   edgecolors="#30363d", linewidths=0.3)
    ax.set_title("Trust Score vs. Urgency Score", fontweight="bold", fontsize=14)
    ax.set_xlabel("Urgency Score (Higher = More Manipulative Signals)")
    ax.set_ylabel("Trust Score (Higher = More Trustworthy)")
    ax.legend(title="Risk Level", framealpha=0.8)
    return save_plot(fig, "08_trust_vs_urgency")


def plot_price_analysis(df):
    """9. Price analysis -- MRP inflation patterns."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Log price distribution by risk
    for risk in RISK_ORDER:
        subset = df[df["manipulation_risk"] == risk]
        axes[0].hist(np.log1p(subset["selling_price"]), bins=30, alpha=0.5,
                     label=risk, color=RISK_COLORS[risk], edgecolor="#30363d")
    axes[0].set_title("Selling Price Distribution (log scale)", fontweight="bold", fontsize=13)
    axes[0].set_xlabel("log(Selling Price)")
    axes[0].set_ylabel("Frequency")
    axes[0].legend(framealpha=0.8)

    # MRP vs Selling Price with discount line
    sample = df.sample(min(200, len(df)), random_state=42)
    colors = [RISK_COLORS[r] for r in sample["manipulation_risk"]]
    axes[1].scatter(sample["original_price"], sample["selling_price"],
                    c=colors, alpha=0.5, s=30, edgecolors="#30363d", linewidths=0.3)
    max_price = max(sample["original_price"].max(), sample["selling_price"].max())
    axes[1].plot([0, max_price], [0, max_price], "--", color="#8b949e", alpha=0.5, label="No Discount Line")
    axes[1].set_title("Original Price vs. Selling Price", fontweight="bold", fontsize=13)
    axes[1].set_xlabel("Original Price (MRP)")
    axes[1].set_ylabel("Selling Price")
    axes[1].legend(framealpha=0.8)

    fig.suptitle("Price Analysis -- MRP Inflation Patterns", fontsize=15, fontweight="bold", y=1.02)
    return save_plot(fig, "09_price_analysis")


def plot_feature_distributions(df):
    """10. Key feature distributions (small multiples)."""
    features = ["discount_percentage", "rating", "rating_count", "review_count",
                 "promotional_intensity", "urgency_score", "trust_score", "price_rating_mismatch"]
    n = len(features)
    cols = 4
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(20, rows * 4))
    axes = axes.flatten()

    for i, feat in enumerate(features):
        for risk in RISK_ORDER:
            subset = df[df["manipulation_risk"] == risk][feat].dropna()
            axes[i].hist(subset, bins=20, alpha=0.5, label=risk, color=RISK_COLORS[risk])
        axes[i].set_title(feat, fontweight="bold", fontsize=11)
        axes[i].legend(fontsize=7, framealpha=0.6)

    # Hide unused axes
    for j in range(n, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Feature Distributions by Risk Level", fontsize=16, fontweight="bold", y=1.01)
    fig.tight_layout()
    return save_plot(fig, "10_feature_distributions")


def main():
    """Run all EDA visualizations."""
    print("+==========================================================+")
    print("|  Exploratory Data Analysis (EDA)                         |")
    print("+==========================================================+\n")

    df = pd.read_csv(config.LABELED_CSV)
    print(f"Loaded {len(df)} labeled records\n")
    print("Generating charts...\n")

    charts = []
    charts.append(plot_risk_distribution(df))
    charts.append(plot_risk_by_platform(df))
    charts.append(plot_risk_by_category(df))
    charts.append(plot_discount_distribution(df))
    charts.append(plot_rating_vs_discount(df))
    charts.append(plot_correlation_heatmap(df))
    charts.append(plot_promotional_intensity(df))
    charts.append(plot_trust_vs_urgency(df))
    charts.append(plot_price_analysis(df))
    charts.append(plot_feature_distributions(df))

    print(f"\n[OK] Generated {len(charts)} charts in {config.CHARTS_DIR}")

    # -- Quick stats summary ----------------------------------------------
    print(f"\n{'=' * 60}")
    print("DATASET SUMMARY")
    print(f"{'=' * 60}")
    print(f"Total listings: {len(df)}")
    print(f"Platforms: {df['platform'].nunique()} ({', '.join(df['platform'].unique())})")
    print(f"Categories: {df['category'].nunique()} ({', '.join(df['category'].unique())})")
    print(f"\nRisk Distribution:")
    for risk in RISK_ORDER:
        count = (df["manipulation_risk"] == risk).sum()
        print(f"  {risk}: {count} ({count/len(df)*100:.1f}%)")

    print(f"\nMean discount by risk:")
    print(df.groupby("manipulation_risk")["discount_percentage"].mean().reindex(RISK_ORDER).round(1).to_string())

    print(f"\nMean rating by risk:")
    print(df.groupby("manipulation_risk")["rating"].mean().reindex(RISK_ORDER).round(2).to_string())

    return charts


if __name__ == "__main__":
    main()
