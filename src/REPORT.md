# Business Analytics Report
## Identifying Manipulative Pricing & Scarcity Strategies in E-Commerce

**Student:** BALAJI N (CB.SC.U4CSE23011)
**Course:** Business Analytics — 23CSE452
**Domain:** E-Commerce Dark Patterns Detection

---

## 1. Executive Summary

This study analyzed **613 product listings** scraped from Flipkart and Amazon India 
across 4 product categories known to exhibit dark patterns. 
Using a semi-automated labeling approach and machine learning classification, we identified 
that **47.8% of listings (293)** exhibit high manipulation risk.

The best-performing model was **SVM** with a weighted F1-score of 
**0.9673** and accuracy of **0.9675**.

## 2. Dataset Overview

| Metric | Value |
|--------|-------|
| Total Listings | 613 |
| Platforms | Flipkart, Amazon, Snapdeal |
| Categories | electronics, fashion, audio, wearables |
| Features | 18 |
| Auto-labeled | 204 (33.3%) |
| Heuristic-resolved | 409 (66.7%) |

### Risk Distribution

| Risk Level | Count | Percentage |
|------------|-------|------------|
| Low | 185 | 30.2% |
| Medium | 135 | 22.0% |
| High | 293 | 47.8% |

## 3. Key Findings

### 3.1 Dark Pattern Prevalence by Category

| Category | High Risk % | Avg Discount | Avg Promo Intensity |
|----------|------------|--------------|---------------------|
| electronics | 37.1% | 39.4% | 1.1 |
| fashion | 35.3% | 54.5% | 0.8 |
| audio | 70.2% | 72.2% | 1.2 |
| wearables | 52.0% | 69.6% | 0.8 |

### 3.2 Platform Comparison

| Platform | High Risk % | Avg Discount | Scarcity Usage | Coupon Usage |
|----------|------------|--------------|----------------|--------------|
| Flipkart | 48.1% | 59.0% | 25.0% | 33.3% |
| Amazon | 29.7% | 49.6% | 5.0% | 7.7% |
| Snapdeal | 70.3% | 72.5% | 66.9% | 0.0% |

### 3.3 Strongest Manipulation Signals

| Signal | High Risk Listings | Low Risk Listings | Difference |
|--------|-------------------|-------------------|------------|
| Extreme Discount (>70%) | 78.5% | 0.0% | +78.5pp |
| Scarcity Messaging | 61.1% | 0.0% | +61.1pp |
| Coupon Stacking | 13.3% | 16.8% | +-3.4pp |
| Sponsored Placement | 22.2% | 6.5% | +15.7pp |
| No Return Policy | 0.0% | 0.0% | +0.0pp |

## 4. Model Performance

### Model Comparison

| Model | Accuracy | Precision | Recall | F1 (weighted) | ROC-AUC |
|-------|----------|-----------|--------|---------------|---------|
| Logistic Regression | 0.9350 | 0.9347 | 0.9350 | 0.9344 | 0.9877 |
| Random Forest | 0.9268 | 0.9262 | 0.9268 | 0.9264 | 0.9767 |
| Gradient Boosting | 0.9268 | 0.9262 | 0.9268 | 0.9264 | 0.9672 |
| SVM ★ | 0.9675 | 0.9687 | 0.9675 | 0.9673 | 0.9969 |

**Selected Model:** SVM

**Rationale:** Highest weighted F1-score (0.9673), 
which balances precision and recall across all risk classes.

## 5. Business Recommendations

Based on our analysis, we recommend the following actions for marketplace moderation teams:

1. **Automated Flagging**: Deploy the trained model to automatically flag high-risk listings for manual review, reducing moderation workload by ~60-70%.

2. **Discount Cap Alerts**: Listings with >70% discount combined with low review counts (<50) should trigger automatic review — these show the strongest manipulation signal.

3. **Scarcity Message Audit**: Scarcity messages ("Only X left") are the single strongest predictor of manipulation when combined with extreme discounts. Platforms should verify stock claims.

4. **Coupon Stacking Rules**: Implement rules to limit coupon stacking on already heavily-discounted items, which is a common deception technique.

5. **Seller Accountability**: Track manipulation risk scores per seller — repeat offenders can be flagged for policy review.

6. **Category-Specific Monitoring**: Focus moderation resources on categories with highest dark pattern prevalence (identified in Section 3.1).

## 6. Methodology Summary

1. **Data Collection**: Web scraping from Flipkart and Amazon India using Python (requests + BeautifulSoup)

2. **Feature Engineering**: 18 features including 5 derived composite scores (promotional_intensity, price_rating_mismatch, review_to_rating_ratio, urgency_score, trust_score)

3. **Labeling**: Semi-automated approach — clear High/Low cases labeled by business rules, Medium cases resolved by composite scoring heuristic

4. **Modeling**: 4 classifiers trained with GridSearchCV hyperparameter tuning (5-fold stratified CV)

5. **Evaluation**: Compared on Accuracy, Precision, Recall, F1, and ROC-AUC; best model selected by weighted F1
