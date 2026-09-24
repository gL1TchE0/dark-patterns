"""
Central configuration for the Dark Patterns Detection project.
All scraping targets, thresholds, and paths in one place.
"""

import os

# ─── Project Paths ──────────────────────────────────────────────────────────────
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SRC_DIR)  # Project root (parent of src/)
DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_DIR = os.path.join(DATA_DIR, "raw")
OUTPUT_DIR = os.path.join(SRC_DIR, "output")
CHARTS_DIR = os.path.join(OUTPUT_DIR, "charts")
MODEL_DIR = os.path.join(OUTPUT_DIR, "model_results")

# Ensure directories exist
for d in [RAW_DIR, CHARTS_DIR, MODEL_DIR]:
    os.makedirs(d, exist_ok=True)

# ─── Scraping Configuration ─────────────────────────────────────────────────────

# Categories prone to dark patterns (high discount inflation, fake urgency, etc.)
CATEGORIES = {
    "electronics": {
        "flipkart_query": "smartphones under 15000",
        "amazon_query": "smartphones under 15000",
    },
    "fashion": {
        "flipkart_query": "men casual shoes",
        "amazon_query": "men casual shoes",
    },
    "audio": {
        "flipkart_query": "bluetooth headphones",
        "amazon_query": "bluetooth headphones",
    },
    "wearables": {
        "flipkart_query": "smart watch for men",
        "amazon_query": "smart watch for men",
    },
    "beauty_health": {
        "flipkart_query": "face cream moisturizer",
        "amazon_query": "face cream moisturizer",
    },
    "supplements": {
        "flipkart_query": "protein powder supplement",
        "amazon_query": "protein powder supplement",
    },
}

# Number of search result pages to scrape per category per platform
PAGES_PER_CATEGORY = 2

# Listings target per platform (~250 each → ~500 total)
TARGET_PER_PLATFORM = 250

# Delay between HTTP requests (seconds) — polite scraping
MIN_DELAY = 1.5
MAX_DELAY = 3.5

# Rotating User-Agent strings
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/129.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:131.0) Gecko/20100101 Firefox/131.0",
]

# HTTP request headers
BASE_HEADERS = {
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
    "Connection": "keep-alive",
    "Upgrade-Insecure-Requests": "1",
}

# ─── Flipkart Specific ──────────────────────────────────────────────────────────
FLIPKART_BASE_URL = "https://www.flipkart.com/search"
FLIPKART_RAW_CSV = os.path.join(RAW_DIR, "flipkart_raw.csv")

# ─── Amazon Specific ────────────────────────────────────────────────────────────
AMAZON_BASE_URL = "https://www.amazon.in/s"
AMAZON_RAW_CSV = os.path.join(RAW_DIR, "amazon_raw.csv")

# ─── Snapdeal Specific (Method 4) ───────────────────────────────────────────────
SNAPDEAL_BASE_URL = "https://www.snapdeal.com/search"
SNAPDEAL_RAW_CSV = os.path.join(RAW_DIR, "snapdeal_raw.csv")

# ─── Merged / Processed Data ────────────────────────────────────────────────────
RAW_MERGED_CSV = os.path.join(DATA_DIR, "raw_merged.csv")
PROCESSED_CSV = os.path.join(DATA_DIR, "processed_listings.csv")
LABELED_CSV = os.path.join(DATA_DIR, "labeled_listings.csv")
FINAL_LABELED_CSV = os.path.join(DATA_DIR, "final_labeled_listings.csv")

# ─── Labeling Thresholds ────────────────────────────────────────────────────────
# These rules define automatic labeling for clear-cut cases

# HIGH RISK if ≥ HIGH_RISK_MIN_FLAGS of these are true:
HIGH_RISK_MIN_FLAGS = 3
HIGH_RISK_RULES = {
    "extreme_discount": 70,          # Discount >= 70%
    "has_scarcity_message": True,     # "Only X left", "Hurry", etc.
    "promotional_badge_count": 2,     # ≥2 promotional badges
    "is_sponsored": True,            # Paid placement
    "has_coupon": True,              # Coupon stacking on top of discount
    "no_return_policy": True,        # No returns = red flag
    "low_reviews_high_discount": {   # < 50 reviews but > 50% discount
        "max_reviews": 50,
        "min_discount": 50,
    },
}

# LOW RISK — all of these must hold:
LOW_RISK_RULES = {
    "max_discount": 30,              # Discount ≤ 30%
    "min_rating": 4.0,               # Rating ≥ 4.0
    "min_ratings_count": 100,        # At least 100 ratings
    "has_return_policy": True,       # Returns allowed
    "max_promotional_cues": 1,       # ≤1 promotional element
    "no_scarcity_message": True,     # No urgency messaging
}

# Everything else → MEDIUM (flagged for manual review)

# ─── Model Training ─────────────────────────────────────────────────────────────
TEST_SIZE = 0.2
RANDOM_STATE = 42
CV_FOLDS = 5
BEST_MODEL_PATH = os.path.join(MODEL_DIR, "best_model.joblib")

# ─── Feature Columns (used in model training) ───────────────────────────────────
FEATURE_COLUMNS = [
    "selling_price",
    "original_price",
    "discount_percentage",
    "rating",
    "rating_count",
    "review_count",
    "is_sponsored",
    "promotional_badge_count",
    "has_scarcity_message",
    "has_coupon",
    "has_return_policy",
    "free_delivery",
    "delivery_days",
    "promotional_intensity",
    "price_rating_mismatch",
    "review_to_rating_ratio",
    "urgency_score",
    "trust_score",
]

TARGET_COLUMN = "manipulation_risk"

# Risk level encoding
RISK_LABELS = {"Low": 0, "Medium": 1, "High": 2}
RISK_LABELS_INV = {v: k for k, v in RISK_LABELS.items()}
