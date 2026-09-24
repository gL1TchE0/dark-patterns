"""
Data Preprocessing & Feature Engineering.
Cleans raw scraped data and engineers derived features for dark-pattern analysis.
"""

import sys
import numpy as np
import pandas as pd 

import config

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


def load_raw_data():
    """Load the raw merged dataset."""
    df = pd.read_csv(config.RAW_MERGED_CSV)
    print(f"Loaded {len(df)} raw records from {config.RAW_MERGED_CSV}")
    return df


def clean_data(df):
    """Clean and standardize raw data."""
    print("\n-- Data Cleaning ------------------------------------------")
    initial = len(df)

    # -- Drop duplicates --------------------------------------------------
    df = df.drop_duplicates(subset=["product_name", "platform", "selling_price"], keep="first")
    print(f"  Removed {initial - len(df)} duplicates -> {len(df)} records")

    # -- Handle missing prices --------------------------------------------
    # Drop rows with no selling price (useless)
    df = df.dropna(subset=["selling_price"])
    df["selling_price"] = pd.to_numeric(df["selling_price"], errors="coerce")
    df = df[df["selling_price"] > 0]

    # Fill missing original_price with selling_price (no discount)
    df["original_price"] = pd.to_numeric(df["original_price"], errors="coerce")
    df["original_price"] = df["original_price"].fillna(df["selling_price"])
    
    # Sanity check: if original_price is unreasonably large (> 10x selling) from scraper string concatenation, recompute
    df["discount_percentage"] = pd.to_numeric(df["discount_percentage"], errors="coerce")
    mask_crazy_orig = (df["original_price"] > df["selling_price"] * 10) & (df["discount_percentage"] > 0) & (df["discount_percentage"] < 95)
    df.loc[mask_crazy_orig, "original_price"] = (df.loc[mask_crazy_orig, "selling_price"] / (1 - df.loc[mask_crazy_orig, "discount_percentage"] / 100)).round(2)
    mask_still_crazy = df["original_price"] > df["selling_price"] * 10
    df.loc[mask_still_crazy, "original_price"] = df.loc[mask_still_crazy, "selling_price"]

    # Ensure original >= selling
    df.loc[df["original_price"] < df["selling_price"], "original_price"] = df["selling_price"]

    # -- Recompute discount if missing ------------------------------------
    df["discount_percentage"] = pd.to_numeric(df["discount_percentage"], errors="coerce")
    mask_no_discount = df["discount_percentage"].isna()
    df.loc[mask_no_discount, "discount_percentage"] = np.where(
        df.loc[mask_no_discount, "original_price"] > 0,
        ((1 - df.loc[mask_no_discount, "selling_price"] / df.loc[mask_no_discount, "original_price"]) * 100).round(1),
        0,
    )
    df["discount_percentage"] = df["discount_percentage"].clip(0, 100)

    # -- Numeric fields ---------------------------------------------------
    for col in ["rating", "rating_count", "review_count", "delivery_days", "promotional_badge_count"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["rating"] = df["rating"].clip(0, 5)
    df["rating"] = df["rating"].fillna(0)
    df["rating_count"] = df["rating_count"].fillna(0).astype(int)
    df["review_count"] = df["review_count"].fillna(0).astype(int)
    df["delivery_days"] = df["delivery_days"].fillna(df["delivery_days"].median())
    df["promotional_badge_count"] = df["promotional_badge_count"].fillna(0).astype(int)

    # -- Boolean fields ---------------------------------------------------
    bool_cols = ["free_delivery", "is_sponsored", "has_scarcity_message", "has_coupon", "has_return_policy"]
    for col in bool_cols:
        df[col] = df[col].fillna(False).astype(bool).astype(int)

    print(f"  After cleaning: {len(df)} records, {len(df.columns)} columns")
    return df


def engineer_features(df):
    """
    Engineer derived features that capture dark-pattern signals.
    """
    print("\n-- Feature Engineering -------------------------------------")

    # -- 1. Promotional Intensity -----------------------------------------
    # Count of simultaneous promotional cues (badges + coupon + scarcity + sponsored)
    df["promotional_intensity"] = (
        df["promotional_badge_count"]
        + df["has_coupon"].astype(int)
        + df["has_scarcity_message"].astype(int)
        + df["is_sponsored"].astype(int)
    )
    print("  [OK] promotional_intensity (sum of all promotional cues)")

    # -- 2. Price-Rating Mismatch -----------------------------------------
    # High discount + low rating = suspicious.
    # Scale: 0 = no mismatch, higher = more suspicious
    df["price_rating_mismatch"] = np.where(
        df["rating"] > 0,
        (df["discount_percentage"] / 100) * (5 - df["rating"]) / 5,
        df["discount_percentage"] / 100,
    )
    df["price_rating_mismatch"] = df["price_rating_mismatch"].round(3)
    print("  [OK] price_rating_mismatch (discount x inverse rating)")

    # -- 3. Review-to-Rating Ratio ----------------------------------------
    # Products with very few reviews relative to ratings may have fake ratings
    df["review_to_rating_ratio"] = np.where(
        df["rating_count"] > 0,
        (df["review_count"] / df["rating_count"]).clip(0, 1),
        0,
    )
    df["review_to_rating_ratio"] = df["review_to_rating_ratio"].round(3)
    print("  [OK] review_to_rating_ratio (reviews / ratings, clipped 0-1)")

    # -- 4. Urgency Score -------------------------------------------------
    # Composite: scarcity message + extreme discount + limited delivery
    df["urgency_score"] = (
        df["has_scarcity_message"].astype(int) * 3          # Scarcity is strongest signal
        + (df["discount_percentage"] > 60).astype(int) * 2  # Extreme discount
        + (df["has_coupon"]).astype(int) * 1                 # Coupon stacking
        + (df["delivery_days"].fillna(99) <= 1).astype(int) * 1  # Rush delivery
    )
    print("  [OK] urgency_score (weighted: scarcityx3 + extreme_discountx2 + coupon + rush_delivery)")

    # -- 5. Trust Score ---------------------------------------------------
    # Higher = more trustworthy listing. Inverse signal to manipulation.
    df["trust_score"] = (
        (df["rating"] / 5) * 2                              # Good rating
        + np.log1p(df["rating_count"]).clip(0, 5) / 5       # Many ratings
        + df["has_return_policy"].astype(int) * 1.5          # Return policy
        + (df["review_to_rating_ratio"] > 0.15).astype(int)  # Reasonable review ratio
        - df["is_sponsored"].astype(int) * 0.5               # Sponsored = less trust
        - (df["discount_percentage"] > 70).astype(int) * 1   # Extreme discount = less trust
    )
    df["trust_score"] = df["trust_score"].round(3)
    print("  [OK] trust_score (composite: rating, reviews, returns, minus suspicion)")

    print(f"\n  Final feature set: {len(df.columns)} columns")
    return df


def main():
    """Run the full preprocessing pipeline."""
    print("+==========================================================+")
    print("|  Data Preprocessing & Feature Engineering                |")
    print("+==========================================================+\n")

    df = load_raw_data()
    df = clean_data(df)
    df = engineer_features(df)

    # -- Save processed data ----------------------------------------------
    df.to_csv(config.PROCESSED_CSV, index=False)
    print(f"\n[OK] Processed data saved to {config.PROCESSED_CSV}")

    # -- Quick summary stats ----------------------------------------------
    print(f"\n{'=' * 60}")
    print("PREPROCESSING SUMMARY")
    print(f"{'=' * 60}")
    print(f"Records: {len(df)}")
    print(f"Features: {len(df.columns)}")
    print(f"\nKey Statistics:")
    summary_cols = [
        "selling_price", "discount_percentage", "rating", "rating_count",
        "promotional_intensity", "urgency_score", "trust_score"
    ]
    print(df[summary_cols].describe().round(2).to_string())

    return df


if __name__ == "__main__":
    main()
