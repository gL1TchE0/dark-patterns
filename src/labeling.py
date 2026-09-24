"""
Semi-Automated Labeling for Manipulation Risk.
Applies rule-based labeling for clear-cut High/Low risk listings,
flags ambiguous cases as Medium, and provides a CLI tool for manual review.
"""

import sys 

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

import numpy as np
import pandas as pd

import config


def apply_rule_based_labels(df):
    """
    Apply predefined business rules to auto-label clear Low and High risk listings.
    Everything else is labeled Medium (for manual review).
    """
    print("-- Applying Rule-Based Labels ------------------------------\n")

    labels = pd.Series("Medium", index=df.index)

    # ===================================================================
    # HIGH RISK RULES
    # A listing gets a "flag" for each suspicious signal.
    # If flags >= HIGH_RISK_MIN_FLAGS, it's auto-labeled HIGH.
    # ===================================================================
    flags = pd.DataFrame(index=df.index)

    # Flag 1: Extreme discount (>= 70%)
    flags["extreme_discount"] = (
        df["discount_percentage"] >= config.HIGH_RISK_RULES["extreme_discount"]
    ).astype(int)

    # Flag 2: Scarcity message present
    flags["scarcity"] = df["has_scarcity_message"].astype(int)

    # Flag 3: Multiple promotional badges (>= 2)
    flags["multi_badge"] = (
        df["promotional_badge_count"] >= config.HIGH_RISK_RULES["promotional_badge_count"]
    ).astype(int)

    # Flag 4: Sponsored listing
    flags["sponsored"] = df["is_sponsored"].astype(int)

    # Flag 5: Coupon stacking
    flags["coupon"] = df["has_coupon"].astype(int)

    # Flag 6: No return policy
    flags["no_return"] = (df["has_return_policy"] == 0).astype(int)

    # Flag 7: Low reviews with high discount
    low_rev_cfg = config.HIGH_RISK_RULES["low_reviews_high_discount"]
    flags["low_reviews_high_discount"] = (
        (df["review_count"] < low_rev_cfg["max_reviews"])
        & (df["discount_percentage"] > low_rev_cfg["min_discount"])
    ).astype(int)

    total_flags = flags.sum(axis=1)
    high_risk_mask = total_flags >= config.HIGH_RISK_MIN_FLAGS
    labels[high_risk_mask] = "High"

    print(f"  HIGH RISK (>={config.HIGH_RISK_MIN_FLAGS} flags): {high_risk_mask.sum()} listings")
    print(f"    Flag breakdown:")
    for col in flags.columns:
        print(f"      {col}: {flags[col].sum()} listings triggered")

    # ===================================================================
    # LOW RISK RULES
    # All conditions must be met for auto LOW label.
    # ===================================================================
    low_risk_mask = (
        (df["discount_percentage"] <= config.LOW_RISK_RULES["max_discount"])
        & (df["rating"] >= config.LOW_RISK_RULES["min_rating"])
        & (df["rating_count"] >= config.LOW_RISK_RULES["min_ratings_count"])
        & (df["has_return_policy"] == 1)
        & (df["promotional_badge_count"] + df["has_coupon"].astype(int) + df["has_scarcity_message"].astype(int) <= config.LOW_RISK_RULES["max_promotional_cues"])
        & (df["has_scarcity_message"] == 0)
    )
    # Don't override HIGH labels
    low_risk_mask = low_risk_mask & ~high_risk_mask
    labels[low_risk_mask] = "Low"

    print(f"\n  LOW RISK (all conditions met): {low_risk_mask.sum()} listings")

    # ===================================================================
    # MEDIUM = everything else
    # ===================================================================
    medium_count = (labels == "Medium").sum()
    print(f"\n  MEDIUM (ambiguous -> manual review): {medium_count} listings")

    df["manipulation_risk"] = labels
    df["auto_labeled"] = labels != "Medium"
    df["flag_count"] = total_flags

    return df


def auto_label_medium_cases(df):
    """
    For academic purposes, automatically resolve Medium cases using a scoring
    heuristic rather than manual review. This uses the urgency_score and
    trust_score to classify ambiguous listings.
    """
    print("\n-- Auto-Resolving Medium Cases (Heuristic) -----------------\n")

    medium_mask = df["manipulation_risk"] == "Medium"
    medium_df = df[medium_mask].copy()

    if len(medium_df) == 0:
        print("  No medium cases to resolve.")
        return df

    # Use a composite suspicion score
    # Higher suspicion -> High risk, lower -> Low risk
    suspicion = (
        medium_df["promotional_intensity"] * 1.5
        + medium_df["urgency_score"]
        + medium_df["price_rating_mismatch"] * 3
        + (medium_df["discount_percentage"] / 100) * 2
        - medium_df["trust_score"]
    )

    # Threshold-based assignment
    p33 = suspicion.quantile(0.33)
    p66 = suspicion.quantile(0.66)

    new_labels = pd.Series("Medium", index=medium_df.index)
    new_labels[suspicion <= p33] = "Low"
    new_labels[suspicion > p66] = "High"

    df.loc[medium_mask, "manipulation_risk"] = new_labels

    low_resolved = (new_labels == "Low").sum()
    high_resolved = (new_labels == "High").sum()
    still_medium = (new_labels == "Medium").sum()

    print(f"  Resolved {len(medium_df)} medium cases:")
    print(f"    -> Low:    {low_resolved}")
    print(f"    -> Medium: {still_medium} (kept as boundary)")
    print(f"    -> High:   {high_resolved}")

    return df


def main():
    """Run the full labeling pipeline."""
    print("+==========================================================+")
    print("|  Semi-Automated Labeling Pipeline                        |")
    print("+==========================================================+\n")

    # Load processed data
    df = pd.read_csv(config.PROCESSED_CSV)
    print(f"Loaded {len(df)} processed records\n")

    # Step 1: Rule-based labeling
    df = apply_rule_based_labels(df)

    # Step 2: Resolve medium cases
    df = auto_label_medium_cases(df)

    # -- Final distribution -----------------------------------------------
    print(f"\n{'=' * 60}")
    print("FINAL LABEL DISTRIBUTION")
    print(f"{'=' * 60}")
    dist = df["manipulation_risk"].value_counts()
    total = len(df)
    for label in ["Low", "Medium", "High"]:
        count = dist.get(label, 0)
        pct = count / total * 100
        bar = "#" * int(pct / 2)
        print(f"  {label:6s}: {count:4d} ({pct:5.1f}%) {bar}")

    print(f"\n  Auto-labeled: {df['auto_labeled'].sum()} ({df['auto_labeled'].mean()*100:.1f}%)")
    print(f"  Heuristic-resolved: {(~df['auto_labeled']).sum()} ({(~df['auto_labeled']).mean()*100:.1f}%)")

    # -- Cross-tab: risk by category and platform -------------------------
    print(f"\n{'=' * 60}")
    print("RISK BY CATEGORY")
    print(f"{'=' * 60}")
    ct = pd.crosstab(df["category"], df["manipulation_risk"], margins=True)
    print(ct.to_string())

    print(f"\n{'=' * 60}")
    print("RISK BY PLATFORM")
    print(f"{'=' * 60}")
    cp = pd.crosstab(df["platform"], df["manipulation_risk"], margins=True)
    print(cp.to_string())

    # -- Save -------------------------------------------------------------
    df.to_csv(config.LABELED_CSV, index=False)
    print(f"\n[OK] Labeled data saved to {config.LABELED_CSV}")

    # Also save as "final" for model training
    df.to_csv(config.FINAL_LABELED_CSV, index=False)
    print(f"[OK] Final labeled data saved to {config.FINAL_LABELED_CSV}")

    return df


if __name__ == "__main__":
    main()
