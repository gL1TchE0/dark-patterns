"""
Orchestrator: runs both live web scrapers (Amazon India + Flipkart),
merges results into a single unified raw dataset, and validates data authenticity.
NO SYNTHETIC DATA IS USED.
"""

import os
import sys
import pandas as pd 

import config
from scraper_amazon import scrape_amazon
from scraper_flipkart import scrape_flipkart
from scraper_snapdeal import scrape_snapdeal

# Ensure UTF-8 output on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


def merge_datasets():
    """Merge Amazon, Flipkart, and Snapdeal CSVs into one unified raw dataset."""
    dfs = []

    if os.path.exists(config.FLIPKART_RAW_CSV):
        df_fk = pd.read_csv(config.FLIPKART_RAW_CSV)
        print(f"  Flipkart:  {len(df_fk)} genuine listings")
        dfs.append(df_fk)
    else:
        print("  [WARN] Flipkart CSV not found")

    if os.path.exists(config.AMAZON_RAW_CSV):
        df_az = pd.read_csv(config.AMAZON_RAW_CSV)
        print(f"  Amazon:    {len(df_az)} genuine listings")
        dfs.append(df_az)
    else:
        print("  [WARN] Amazon CSV not found")

    if os.path.exists(config.SNAPDEAL_RAW_CSV):
        df_sd = pd.read_csv(config.SNAPDEAL_RAW_CSV)
        print(f"  Snapdeal:  {len(df_sd)} genuine listings")
        dfs.append(df_sd)
    else:
        print("  [WARN] Snapdeal CSV not found")

    if not dfs:
        print("  [FAIL] No scraped data available to merge")
        return None

    merged = pd.concat(dfs, ignore_index=True)
    
    # Drop empty or corrupted rows if any
    merged = merged.dropna(subset=["product_name", "selling_price"])
    
    merged.to_csv(config.RAW_MERGED_CSV, index=False)
    print(f"\n[OK] Merged dataset: {len(merged)} live scraped listings -> {config.RAW_MERGED_CSV}")
    return merged


def main(force_scrape=False):
    """Run all live scrapers and merge results."""
    print("=" * 60)
    print("LIVE E-COMMERCE SCRAPING PIPELINE (ALL PLATFORMS)")
    print("=" * 60)

    # Check if fresh genuine datasets already exist
    fk_exists = os.path.exists(config.FLIPKART_RAW_CSV)
    az_exists = os.path.exists(config.AMAZON_RAW_CSV)
    sd_exists = os.path.exists(config.SNAPDEAL_RAW_CSV)
    
    fk_count = len(pd.read_csv(config.FLIPKART_RAW_CSV)) if fk_exists else 0
    az_count = len(pd.read_csv(config.AMAZON_RAW_CSV)) if az_exists else 0
    sd_count = len(pd.read_csv(config.SNAPDEAL_RAW_CSV)) if sd_exists else 0

    if not force_scrape and fk_count >= 150 and az_count >= 150 and sd_count >= 150:
        print(f"[OK] Found verified live scraped datasets: Flipkart ({fk_count}), Amazon ({az_count}), Snapdeal ({sd_count})")
        print("  Merging and validating...")
        return merge_datasets()

    # -- Step 1: Scrape Amazon India --------------------------------------
    if force_scrape or az_count < 150:
        print("\n--- PHASE 1: Scraping Amazon India ---")
        try:
            amazon_listings = scrape_amazon()
            print(f"  Amazon scraping finished: {len(amazon_listings)} listings")
        except Exception as e:
            print(f"  [FAIL] Amazon scraping error: {e}")
    else:
        print(f"\n[OK] Using existing Amazon live scraped dataset ({az_count} listings)")

    # -- Step 2: Scrape Flipkart via Selenium Headless Chrome -------------
    if force_scrape or fk_count < 150:
        print("\n--- PHASE 2: Scraping Flipkart (Selenium Headless Chrome) ---")
        try:
            flipkart_listings = scrape_flipkart()
            print(f"  Flipkart scraping finished: {len(flipkart_listings)} listings")
        except Exception as e:
            print(f"  [FAIL] Flipkart scraping error: {e}")
    else:
        print(f"\n[OK] Using existing Flipkart live scraped dataset ({fk_count} listings)")

    # -- Step 3: Scrape Snapdeal ------------------------------------------
    if force_scrape or sd_count < 150:
        print("\n--- PHASE 3: Scraping Snapdeal ---")
        try:
            snapdeal_listings = scrape_snapdeal(target_count=200)
            print(f"  Snapdeal scraping finished: {len(snapdeal_listings)} listings")
        except Exception as e:
            print(f"  [FAIL] Snapdeal scraping error: {e}")
    else:
        print(f"\n[OK] Using existing Snapdeal live scraped dataset ({sd_count} listings)")

    # -- Step 4: Merge and Validate ---------------------------------------
    print("\n--- PHASE 4: Merging & Validating Datasets ---")
    merged_df = merge_datasets()

    if merged_df is not None and len(merged_df) >= 300:
        print("\n" + "=" * 60)
        print(f"SUCCESS: Collected {len(merged_df)} genuine listings across platforms!")
        print(f"Platforms: {merged_df['platform'].value_counts().to_dict()}")
        print(f"Categories: {merged_df['category'].value_counts().to_dict()}")
        print("=" * 60)
    elif merged_df is not None:
        print(f"\n[WARN] Dataset contains {len(merged_df)} listings (target: 300-500).")
    else:
        print("\n[FAIL] Failed to produce raw dataset.")

if __name__ == "__main__":
    main()
