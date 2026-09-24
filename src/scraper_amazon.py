"""
Amazon India product listing scraper.
Extracts product details from Amazon.in search results for dark-pattern analysis.
"""

import csv
import random
import re
import time
from urllib.parse import urlencode

import requests
from bs4 import BeautifulSoup

import config


import sys

# Ensure UTF-8 output on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


def get_session():
    """Create a requests session with rotating Chrome headers."""
    session = requests.Session()
    session.headers.update(config.BASE_HEADERS)
    session.headers["User-Agent"] = random.choice(config.USER_AGENTS)
    return session


def random_delay():
    """Polite delay between requests."""
    time.sleep(random.uniform(config.MIN_DELAY, config.MAX_DELAY))


def fetch_page(session, url, retries=3):
    """Fetch a page with retries and random User-Agent."""
    for attempt in range(retries):
        try:
            session.headers["User-Agent"] = random.choice(config.USER_AGENTS)
            resp = session.get(url, timeout=15)
            if resp.status_code == 200:
                text_lower = resp.text.lower()
                if "validatecaptcha" in text_lower or "enter the characters you see below" in text_lower:
                    print(f"  [WARN] Amazon CAPTCHA encountered (attempt {attempt+1}). Retrying...")
                    time.sleep(5)
                    continue
                return resp.text
            elif resp.status_code in (503, 429):
                wait = (attempt + 1) * 6
                print(f"  [WARN] HTTP {resp.status_code}. Waiting {wait}s...")
                time.sleep(wait)
            else:
                print(f"  [WARN] HTTP {resp.status_code} for {url}")
                time.sleep(3)
        except requests.RequestException as e:
            print(f"  [FAIL] Request error (attempt {attempt+1}): {e}")
            time.sleep(3)
    return None


def extract_price(text):
    """Extract numeric price from text like '₹1,299' -> 1299.0"""
    if not text:
        return None
    text = text.replace('Rs.', '').replace('Rs', '').replace(',', '').replace('₹', '').strip()
    nums = re.sub(r"[^\d.]", "", text)
    try:
        return float(nums) if nums else None
    except ValueError:
        return None


def extract_number(text):
    """Extract first number from text like '1,234' -> 1234"""
    if not text:
        return 0
    nums = re.sub(r"[^\d]", "", text.replace(",", ""))
    try:
        return int(nums) if nums else 0
    except ValueError:
        return 0


def parse_amazon_listings(html, category):
    """
    Parse Amazon India search results HTML and extract product listing data.
    """
    soup = BeautifulSoup(html, "lxml")
    listings = []

    # Amazon product cards have data-component-type="s-search-result"
    result_cards = soup.find_all("div", attrs={"data-component-type": "s-search-result"})

    if not result_cards:
        # Fallback: look for divs with data-asin
        result_cards = soup.find_all("div", attrs={"data-asin": True})
        # Filter out empty asin (non-product elements)
        result_cards = [c for c in result_cards if c.get("data-asin", "").strip()]

    for card in result_cards:
        try:
            listing = parse_single_amazon_card(card, category)
            if listing and listing.get("product_name"):
                listings.append(listing)
        except Exception:
            continue

    return listings


def parse_single_amazon_card(card, category):
    """Parse a single Amazon product card."""
    text_content = card.get_text(separator="|", strip=True)
    all_text = text_content.lower()

    listing = {
        "platform": "Amazon",
        "category": category,
        "product_name": "",
        "selling_price": None,
        "original_price": None,
        "discount_percentage": None,
        "rating": None,
        "rating_count": 0,
        "review_count": 0,
        "seller_name": "",
        "seller_type": "Third-party",
        "delivery_info": "",
        "delivery_days": None,
        "free_delivery": False,
        "is_sponsored": False,
        "promotional_badges": "",
        "promotional_badge_count": 0,
        "scarcity_message": "",
        "has_scarcity_message": False,
        "has_coupon": False,
        "coupon_text": "",
        "return_policy": "",
        "has_return_policy": True,
        "product_url": "",
    }

    # -- Product Name -----------------------------------------------------
    # Amazon uses h2 > a > span for product titles
    title_tag = card.find("h2")
    if title_tag:
        listing["product_name"] = title_tag.get_text(strip=True)
        # URL from the title link
        title_link = title_tag.find("a")
        if title_link:
            href = title_link.get("href", "")
            if href.startswith("/"):
                listing["product_url"] = "https://www.amazon.in" + href.split("?")[0]
    else:
        # Fallback: longest text that looks like a title
        for span in card.find_all("span"):
            t = span.get_text(strip=True)
            if 20 < len(t) < 300 and "₹" not in t:
                listing["product_name"] = t
                break

    # -- Prices -----------------------------------------------------------
    # Current price: span.a-price-whole
    price_whole = card.find("span", class_="a-price-whole")
    if price_whole:
        listing["selling_price"] = extract_price(price_whole.get_text())

    # Original price: span.a-price with data-a-strike or span.a-text-price
    orig_price_tag = card.find("span", class_="a-price", attrs={"data-a-strike": "true"}) or card.find("span", class_="a-text-price")
    if orig_price_tag:
        offscreen = orig_price_tag.find("span", class_="a-offscreen")
        if offscreen:
            listing["original_price"] = extract_price(offscreen.get_text())
        else:
            # First ₹ amount in the tag
            m = re.search(r"₹\s*([\d,]+)", orig_price_tag.get_text())
            if m:
                listing["original_price"] = extract_price(m.group(1))
    
    if not listing["original_price"]:
        # Look for MRP text
        mrp_match = re.search(r"m\.?r\.?p\.?:?\s*₹?([\d,]+)", all_text)
        if mrp_match:
            listing["original_price"] = extract_price(mrp_match.group(1))

    # If we only got selling price, look for original in text
    if listing["selling_price"] and not listing["original_price"]:
        price_tags = card.find_all(string=re.compile(r"₹"))
        prices = []
        for pt in price_tags:
            p = extract_price(pt.strip())
            if p and p > 0:
                prices.append(p)
        if len(prices) >= 2:
            listing["original_price"] = max(prices)

    # If original price wasn't found, default to selling price
    if listing["selling_price"] and not listing["original_price"]:
        listing["original_price"] = listing["selling_price"]

    # -- Discount ---------------------------------------------------------
    discount_match = re.search(r"\((\d{1,2})%\s*off\)", text_content, re.IGNORECASE)
    if not discount_match:
        discount_match = re.search(r"(\d{1,2})%\s*off", text_content, re.IGNORECASE)
    if discount_match:
        listing["discount_percentage"] = float(discount_match.group(1))
    elif listing["selling_price"] and listing["original_price"] and listing["original_price"] > listing["selling_price"]:
        listing["discount_percentage"] = round(
            (1 - listing["selling_price"] / listing["original_price"]) * 100, 1
        )

    # -- Rating -----------------------------------------------------------
    # Amazon: span with "a-icon-alt" containing "X.X out of 5 stars"
    rating_tag = card.find("span", class_="a-icon-alt")
    if rating_tag:
        rm = re.search(r"(\d\.?\d?)\s*out of", rating_tag.get_text())
        if rm:
            listing["rating"] = float(rm.group(1))

    if not listing["rating"]:
        rm = re.search(r"(\d\.?\d?)\s*out of\s*5", all_text)
        if rm:
            listing["rating"] = float(rm.group(1))

    # -- Rating Count -----------------------------------------------------
    # Amazon shows rating count as a link like "(1,234)"
    rating_link = card.find("a", href=re.compile(r"#customerReviews|product-reviews"))
    if rating_link:
        listing["rating_count"] = extract_number(rating_link.get_text())
        listing["review_count"] = listing["rating_count"]  # Amazon conflates these
    else:
        rc_match = re.search(r"([\d,]+)\s*(?:ratings?|reviews?)", all_text)
        if rc_match:
            listing["rating_count"] = extract_number(rc_match.group(1))
            listing["review_count"] = listing["rating_count"]

    # -- Delivery ---------------------------------------------------------
    delivery_match = re.search(
        r"(free delivery|get it by [^|]+|delivery [^|]+)", all_text
    )
    if delivery_match:
        listing["delivery_info"] = delivery_match.group(1).strip()

    if "free delivery" in all_text or "free" in listing.get("delivery_info", "").lower():
        listing["free_delivery"] = True

    # Amazon Prime implies free delivery
    if "prime" in all_text:
        listing["free_delivery"] = True

    days_match = re.search(r"(\d+)\s*days?", listing.get("delivery_info", ""))
    if days_match:
        listing["delivery_days"] = int(days_match.group(1))

    # -- Sponsored --------------------------------------------------------
    sponsored_tag = card.find("span", string=re.compile(r"Sponsored", re.IGNORECASE))
    if sponsored_tag or "sponsored" in all_text:
        listing["is_sponsored"] = True

    # -- Promotional Badges -----------------------------------------------
    badges = []
    badge_patterns = [
        ("amazon's choice", "Amazon's Choice"),
        ("best seller", "Best Seller"),
        ("bestseller", "Best Seller"),
        ("limited time deal", "Limited Time Deal"),
        ("lightning deal", "Lightning Deal"),
        ("deal of the day", "Deal of the Day"),
        ("top rated", "Top Rated"),
        ("climate pledge friendly", "Climate Pledge Friendly"),
        ("subscribe & save", "Subscribe & Save"),
        ("subscribe and save", "Subscribe & Save"),
        ("combo", "Combo Offer"),
    ]
    for pattern, badge_name in badge_patterns:
        if pattern in all_text:
            if badge_name not in badges:
                badges.append(badge_name)

    listing["promotional_badges"] = "; ".join(badges)
    listing["promotional_badge_count"] = len(badges)

    # -- Scarcity Messages ------------------------------------------------
    scarcity_patterns = [
        r"only\s+\d+\s+left\s+in\s+stock",
        r"only\s+\d+\s+left",
        r"limited\s+stock",
        r"hurry",
        r"selling\s+fast",
        r"almost\s+gone",
        r"few\s+left",
        r"order\s+soon",
        r"ending\s+soon",
    ]
    scarcity_found = []
    for sp in scarcity_patterns:
        m = re.search(sp, all_text)
        if m:
            scarcity_found.append(m.group(0).strip())

    if scarcity_found:
        listing["scarcity_message"] = "; ".join(scarcity_found)
        listing["has_scarcity_message"] = True

    # -- Coupons ----------------------------------------------------------
    coupon_patterns = [
        r"save\s+\d+%\s+with\s+coupon",
        r"apply\s+\d+%\s+coupon",
        r"coupon",
        r"extra\s+\d+%\s*off",
        r"cashback",
        r"bank\s+offer",
    ]
    for cp in coupon_patterns:
        cm = re.search(cp, all_text)
        if cm:
            listing["has_coupon"] = True
            listing["coupon_text"] = cm.group(0).strip()
            break

    # -- Seller Type ------------------------------------------------------
    if "fulfilled by amazon" in all_text or "prime" in all_text:
        listing["seller_type"] = "Marketplace Fulfilled"

    return listing


def scrape_amazon(session=None):
    """
    Scrape Amazon India product listings across configured categories.
    Returns list of listing dicts and saves to CSV.
    """
    if session is None:
        session = get_session()

    all_listings = []
    print("=" * 60)
    print("AMAZON INDIA SCRAPER")
    print("=" * 60)

    for cat_name, cat_config in config.CATEGORIES.items():
        query = cat_config["amazon_query"]
        print(f"\n Category: {cat_name} | Query: '{query}'")
        print("-" * 40)

        for page_num in range(1, config.PAGES_PER_CATEGORY + 1):
            params = {"k": query, "page": page_num}
            url = f"{config.AMAZON_BASE_URL}?{urlencode(params)}"
            print(f"  Page {page_num}: {url}")

            html = fetch_page(session, url)
            if not html:
                print(f"  [FAIL] Failed to fetch page {page_num}")
                continue

            listings = parse_amazon_listings(html, cat_name)
            print(f"  [OK] Extracted {len(listings)} listings")
            all_listings.extend(listings)
            random_delay()

            if len(all_listings) >= config.TARGET_PER_PLATFORM:
                print(f"\n  [OK] Reached target ({config.TARGET_PER_PLATFORM} listings)")
                break

        if len(all_listings) >= config.TARGET_PER_PLATFORM:
            break

    # Save to CSV
    if all_listings:
        save_to_csv(all_listings, config.AMAZON_RAW_CSV)
        print(f"\n[OK] Saved {len(all_listings)} Amazon listings to {config.AMAZON_RAW_CSV}")
    else:
        print("\n[FAIL] No listings scraped from Amazon India")

    return all_listings


def save_to_csv(listings, filepath):
    """Save listing dicts to CSV."""
    if not listings:
        return
    fieldnames = listings[0].keys()
    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(listings)


if __name__ == "__main__":
    results = scrape_amazon()
    print(f"\nTotal Amazon listings: {len(results)}")
