#  E-Commerce Dark Patterns Detection

> Identifying manipulative pricing & scarcity strategies on Indian e-commerce platforms using web scraping, feature engineering, and machine learning.

**Author:** Balaji N (CB.SC.U4CSE23011)  
**Course:** Business Analytics -- 23CSE452

---
 
##  Overview

This project scrapes product listings from **Amazon India**, **Flipkart**, and **Snapdeal**, engineers features that capture dark-pattern signals (inflated discounts, fake scarcity, coupon stacking, etc.), and trains ML classifiers to flag high-risk listings automatically.

| Metric | Value |
|--------|-------|
| Listings analyzed | 613 |
| Platforms | Amazon, Flipkart, Snapdeal |
| Best model | SVM (F1 = 0.967) |
| Categories | Electronics, Fashion, Audio, Wearables |

---

##  Repository Structure

```
|-- data/                         # Datasets (raw + processed)
|   |-- raw/                      # Per-platform scraped CSVs
|   |-- raw_merged.csv
|   |-- processed_listings.csv
|   |-- labeled_listings.csv
|   +-- final_labeled_listings.csv
|-- src/                          # Source code & generated outputs
|   |-- config.py                 # Central configuration (paths, thresholds)
|   |-- scrape_all.py             # Scraping orchestrator
|   |-- scraper_amazon.py         # Amazon India scraper
|   |-- scraper_flipkart.py       # Flipkart scraper
|   |-- scraper_snapdeal.py       # Snapdeal scraper
|   |-- preprocessing.py          # Data cleaning & feature engineering
|   |-- labeling.py               # Semi-automated risk labeling
|   |-- eda.py                    # Exploratory data analysis & charts
|   |-- model.py                  # Model training & evaluation
|   |-- generate_report.py        # Auto-generates analytics report
|   |-- run_pipeline.py           # End-to-end pipeline runner
|   |-- REPORT.md                 # Generated analytics report
|   |-- dashboard.html            # Interactive dashboard
|   +-- output/
|       |-- charts/               # EDA visualizations (10 PNGs)
|       +-- model_results/        # Metrics, confusion matrices, feature importance
|-- dark_patterns_pipeline.ipynb  # Jupyter notebook -- runs the full pipeline
|-- CB.SC.U4CSE23011_Case_Study_Report.docx
|-- CB.SC.U4CSE23011_Case_Study_Report.pdf
|-- requirements.txt
|-- .gitignore
+-- README.md
```

---

##  Quick Start

### Prerequisites

- **Python 3.10+**
- **pip** (comes with Python)
- A working internet connection (for scraping)

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/dark-patterns-detection.git
cd dark-patterns-detection
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. (Optional) Install Jupyter if not already available

```bash
pip install jupyter
```

---

## > Running the Pipeline

### Option A: Jupyter Notebook (Recommended)

Open `dark_patterns_pipeline.ipynb` in Jupyter and run all cells:

```bash
jupyter notebook dark_patterns_pipeline.ipynb
```

The notebook walks through each phase with explanations and runs the full pipeline interactively.

### Option B: Command Line

```bash
python src/run_pipeline.py
```

This runs all 7 phases end-to-end:

1. **Data Collection** -- Scrapes Amazon, Flipkart, Snapdeal
2. **Preprocessing** -- Cleans data & engineers 18 features
3. **Labeling** -- Semi-automated risk classification (High / Medium / Low)
4. **EDA** -- Generates 10 visualization charts
5. **Model Training** -- Trains 4 classifiers with GridSearchCV
6. **Evaluation** -- Compares models, selects best by weighted F1
7. **Report Generation** -- Produces `src/REPORT.md`

---

##  Outputs

| Output | Location |
|--------|----------|
| EDA Charts | `src/output/charts/` |
| Model Metrics | `src/output/model_results/` |
| Analytics Report | `src/REPORT.md` |
| Dashboard | `src/dashboard.html` |
| Processed Data | `data/` |

---

##  Report

The full case study report is available in two formats:
- **PDF:** `CB.SC.U4CSE23011_Case_Study_Report.pdf`

---

## [WARN] Notes

- **Scraping may take 5-10 minutes** depending on network speed and platform rate limits.
- The scrapers use polite delays (1.5-3.5s between requests) and rotating user agents.
- If scraping fails (sites block requests), the pipeline will continue with existing data in `data/raw/`.
- The trained model (`.joblib`) is excluded from git via `.gitignore` due to its size.

---

##  License

This project is for academic purposes as part of the Business Analytics coursework at Amrita Vishwa Vidyapeetham.
