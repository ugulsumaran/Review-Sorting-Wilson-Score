# Amazon Product Rating & Review Sorting Analysis
## 📌 Project Overview
This project aims to solve two major problems in e-commerce product analytics:

Rating Accuracy: Calculating a more accurate product rating by weighing recent reviews higher than older ones (Time-Based Weighted Average).

Review Sorting: Sorting reviews based on their helpfulness using the Wilson Lower Bound Score, rather than simple "Helpful/Total" ratios which can be misleading.

The analysis is performed on a dataset containing Amazon product reviews (specifically for electronics/tech products).

## 📂 Dataset Features
The dataset (amazon_review.csv) contains the following variables:

<img width="436" height="279" alt="image" src="https://github.com/user-attachments/assets/283a7626-f961-402e-bebf-70faaec13076" />


## ⚙️ Methodology
### 1. Time-Based Weighted Average
Instead of a simple arithmetic mean, this project applies a weighted average based on the recency of the review. The assumption is that recent reviews reflect the current state of the product better (e.g., after software updates or batch changes).

Weights used:
- Newest 25% (Q1): 28%
- 2nd Quarter (Q2): 26%
- 3rd Quarter (Q3): 24%
- Oldest 25% (Q4): 22%

### 2. Wilson Lower Bound Score (Sorting)
The project implements the Wilson Lower Bound score to rank reviews. This method provides a confidence interval for the "helpfulness" proportion.

Problem with Simple Average: A review with 1 positive vote out of 1 (100%) appears better than a review with 900 positive out of 1000 (90%).

Solution: The Wilson score balances the proportion of positive ratings with the uncertainty of a small number of observations. It is the industry standard for "Best" or "Most Helpful" sorting (used by Reddit, Yelp, Amazon).

## 📊 Key Results & Strategic Insights
This analysis of real Amazon microSD card data reveals critical patterns in customer satisfaction and review reliability:

Rating Trend Analysis: The product displays a noticeable downward trend over its lifecycle. While recent reviews (last ~9 months) maintain a high average of ≈4.70, older reviews (>2 years) drop to ≈4.45, suggesting that early expectations or product batches might differ from long-term usage.

Time-Based Performance: The calculated Time-Based Weighted Average (4.596) is higher than the Simple Average (4.588). This proves that recent customer satisfaction is on an upward trend, and the product's latest performance is driving the overall score higher.

The "Helpful" Paradox: Data shows that 1-star reviews can receive thousands of "helpful" votes. These reviews act as significant "red flags," highlighting serious issues like overheating, counterfeit products, or sudden failures that the simple average rating often masks.

Reliable Ranking via Wilson Lower Bound: Traditional sorting methods often fail by promoting low-volume "perfect" scores. The Wilson Lower Bound algorithm successfully surfaces reviews that are both high-quality and validated by a large sample size, providing the most trustworthy feedback for potential buyers.

## Installation & Usage
1. Clone the repository:
```bash
git clone https://github.com/ugulsumaran/amazon-rating-analysis.git
```

3. Install the required libraries:
```bash
pip install pandas numpy matplotlib scipy
```

3. Run the analysis:
```bash
python Review-Sorting-Wilson-Score.py
```

## 📈 Visualizations
The script generates a horizontal bar chart comparing the average ratings across different time segments, providing a visual representation of the product's performance trend.
<img width="987" height="540" alt="download" src="https://github.com/user-attachments/assets/e27491cc-4088-4e33-b39b-03bf44f6fd8d" />


## LICENSE 
Copyright (c) 2025 Ümmügülsüm Aran
MIT License - See [LICENSE](LICENSE) file for details
