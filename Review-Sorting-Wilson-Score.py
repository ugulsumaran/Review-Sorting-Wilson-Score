import pandas as pd           # For data manipulation, analysis, and handling tabular data (DataFrames).
import numpy as np            # For numerical computing, array operations, and linear algebra.
import matplotlib.pyplot as plt # For creating static, interactive, and animated data visualizations.

import math                   # For accessing basic mathematical functions and constants (e.g., pi, sqrt).
from scipy.stats import norm  # For statistical analysis and working with the normal (Gaussian) distribution.

# Read data
df = pd.read_csv("amazon_review.csv")

# Exploratory Data Analysis (EDA)
def check_df(dataframe, target=None):
    print("############################")
    print("DATASET OVERVIEW")
    print("############################\n")

    # Basic Info
    print("##### Shape #####")
    print(dataframe.shape)
    
    print("\n##### Data Types #####")
    print(dataframe.dtypes)
    
    print("\n##### First 5 Rows #####")
    display(dataframe.head(5))
    
    print("\n##### Last 5 Rows #####")
    display(dataframe.tail(5))

    # Missing Values
    print("\n##### Missing Values #####")
    print(dataframe.isnull().sum())

check_df(df, target="Performance Index")

# ---------------------------------------------------------------------------------------
# Product Rating Analysis - Time-based weighted Average

## DISTRIBUTION OF RATINGS AND TOTAL REVIEWS BY YEAR
df_ = df.copy() # Create a copy of the original dataframe to avoid modifying the source

# Extract the year (if it doesn't already exist)
if 'year' not in df_.columns:
    # Convert Unix timestamp to datetime and extract the year component
    df_['year'] = pd.to_datetime(df_['unixReviewTime'], unit='s').dt.year

# Count how many times each rating was given per year
# Group by year, count occurrences of each score, and pivot into a table format
rating_counts = df_.groupby('year')['overall'].value_counts().unstack(fill_value=0).astype(int)

# Total number of reviews for each year
total_reviews = df_['year'].value_counts().sort_index() # Count total entries and sort by year

# Printing in a clean format
print("DISTRIBUTION OF RATINGS AND TOTAL REVIEWS BY YEAR\n") # Main header for output
for year in sorted(total_reviews.index): # Iterate through each year in chronological order
    print(f"★ {int(year)} ★") # Display the year as a header
    print(f"   Total reviews: {total_reviews[year]}") # Print total review count for that year
    for rating in [1.0, 2.0, 3.0, 4.0, 5.0]: # Loop through each possible star rating
        # Get count for specific rating, default to 0 if that rating doesn't exist in that year
        count = rating_counts.loc[year, rating] if rating in rating_counts.columns else 0
        print(f"   {int(rating)}★ → {count} count") # Print the individual rating breakdown
    print("-" * 45) # Visual separator between years


    # 1. Simple (unweighted) average score
simple_avg = df['overall'].mean() # Calculate the arithmetic mean of the 'overall' column

print("┌──────────────────────────────────────────────┐")
print("│            PRODUCT RATING ANALYSIS           │") # Header for the analysis output
print("└──────────────────────────────────────────────┘")
print(f"Simple (unweighted) average score:      {simple_avg:.3f}") # Display the simple average
print("-" * 50)

# 2. Find time interval boundaries (quantiles)
quantiles = df['day_diff'].quantile([0.25, 0.50, 0.75]) # Calculate quartiles for days since review

print("Time interval boundaries (day_diff - days ago):")
print(f"  Q1 (newest 25%)  → ≤ {quantiles[0.25]:.0f} days") # Newest reviews boundary
print(f"  Q2 (median)     →   {quantiles[0.50]:.0f} days") # Median boundary
print(f"  Q3              →   {quantiles[0.75]:.0f} days") # Oldest reviews boundary
print("-" * 50)

print("Approximate time intervals:")
print(f"  Newest 25% reviews → 0        - {quantiles[0.25]:.0f} days ago") # Interval for w1
print(f"  2nd quarter        → {quantiles[0.25]:.0f} - {quantiles[0.50]:.0f} days ago") # Interval for w2
print(f"  3rd quarter        → {quantiles[0.50]:.0f} - {quantiles[0.75]:.0f} days ago") # Interval for w3
print(f"  Oldest 25% reviews → {quantiles[0.75]:.0f}+ days ago") # Interval for w4
print("-" * 50)

# 3. Time-based weighted average function
def time_based_weighted_average(dataframe, w1=28, w2=26, w3=24, w4=22):
    weights = [w1, w2, w3, w4] # Store weights in a list
    """
    Splits reviews into 4 groups based on day_diff quartiles
    and assigns different weights to each group (total 100)
    """
    q25 = dataframe['day_diff'].quantile(0.25) # Define the 25th percentile
    q50 = dataframe['day_diff'].quantile(0.50) # Define the 50th percentile
    q75 = dataframe['day_diff'].quantile(0.75) # Define the 75th percentile
    
    # Calculate means for each time-based segment
    avg_newest    = dataframe.loc[dataframe['day_diff'] <= q25, 'overall'].mean()
    avg_second    = dataframe.loc[(dataframe['day_diff'] > q25) & (dataframe['day_diff'] <= q50), 'overall'].mean()
    avg_third     = dataframe.loc[(dataframe['day_diff'] > q50) & (dataframe['day_diff'] <= q75), 'overall'].mean()
    avg_oldest    = dataframe.loc[dataframe['day_diff'] > q75, 'overall'].mean()
    
    # Calculate the final weighted average score
    weighted_avg = (
        avg_newest * w1 +
        avg_second * w2 +
        avg_third  * w3 +
        avg_oldest * w4
    ) / 100
    
    return weighted_avg, avg_newest, avg_second, avg_third, avg_oldest, weights


# Calculation
weighted, avg1, avg2, avg3, avg4, weights = time_based_weighted_average(df) # Run the weighting function

print("WEIGHTED AVERAGE BY TIME INTERVALS")
print(f"Weighted average score:          {weighted:.3f}") # Display weighted result
print()
print("Detailed breakdown:")
print(f"  Newest 25% reviews → {avg1:>6.3f}  (weight: 28)") # Score and weight for Q1
print(f"  2nd quarter        → {avg2:>6.3f}  (weight: 26)") # Score and weight for Q2
print(f"  3rd quarter        → {avg3:>6.3f}  (weight: 24)") # Score and weight for Q3
print(f"  Oldest 25% reviews → {avg4:>6.3f}  (weight: 22)") # Score and weight for Q4
print("-" * 50)

# Short summary/commentary
diff = weighted - simple_avg # Calculate the difference between methods
direction = "higher" if diff > 0 else "lower" # Determine the direction of the change
print(f"Conclusion:")
print(f"The weighted average is {abs(diff):.3f} points {direction} than the simple average.") # Final output

# Average Scores by Time Intervals
segments = ["Newest 25%", "2nd Quarter", "3rd Quarter", "Oldest 25%"] # Labels for time segments
avg_scores = [avg1, avg2, avg3, avg4]           # Mean scores derived from the function
simple_avg = df['overall'].mean()                # Calculate the simple arithmetic mean
weighted = time_based_weighted_average(df)[0]   # Extract weighted average (first return value)
weights = time_based_weighted_average(df)[5]    # Extract weights list (sixth return value)
colors = ['#1f77b4', '#5dade2', '#85c1e9', '#d5d8dc']  # Blue shades (Newest → Oldest)

# Plot settings
fig, ax = plt.subplots(figsize=(10, 5.5)) # Create a figure and axis with specific size

# Horizontal bar chart
bars = ax.barh(segments[::-1], avg_scores[::-1], height=0.65, color=colors[::-1]) # Draw bars in reversed order

# Adding text values on/beside the bars
for bar, score, weight in zip(bars, avg_scores[::-1], weights[::-1]):
    width = bar.get_width() # Get the length of the bar
    ax.text(width + 0.01, bar.get_y() + bar.get_height()/2,
            f'{score:.3f}\n(weight: {weight})', # Display score and weight percentage
            va='center', fontsize=10, fontweight='bold')

# Simple average reference line
ax.axvline(simple_avg, color='red', linestyle='--', linewidth=1.0, alpha=0.6,
           label=f'Simple Average: {simple_avg:.3f}') # Vertical dashed line for simple mean

# Weighted average reference line (added to legend)
ax.axvline(weighted, color='#27ae60', linestyle='-', linewidth=1.5, alpha=0.7,
           label=f'Weighted Average: {weighted:.3f}') # Vertical solid line for weighted mean

# Appearance improvements
ax.set_xlim(4.2, 4.9) # Set the range for the X-axis
ax.set_xlabel('Average Score (1–5)', fontsize=11) # Set X-axis label
ax.set_title('Average Scores by Time Intervals\n(Newest → Oldest)', fontsize=14, pad=15) # Set chart title
ax.invert_yaxis()  # Ensure newest reviews appear at the top
ax.grid(axis='x', linestyle='--', alpha=0.4) # Add subtle vertical grid lines

# Legend configuration
ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1), fontsize=10,
          frameon=True, edgecolor='gray') # Place legend outside the main plot area

plt.tight_layout() # Adjust layout to prevent overlapping
plt.show() # Display the final plot

# ---------------------------------------------------------------------------------------
# Calculate score_pos_neg_diff, score_average_rating, and wilson_lower_bound scores and add to the data

df["helpful_no"] = df["total_vote"] - df["helpful_yes"] # Calculate negative votes by subtracting helpful votes from total votes

# 1. Positive - Negative difference score (simple but effective)
def score_pos_neg_diff(up, down):
    return up - down # Subtract negative votes from positive votes

# 2. Average rating score (up / total ratio)
def score_average_rating(up, down):
    total = up + down # Calculate total votes
    return up / total if total > 0 else 0 # Return positive ratio or 0 if no votes exist

# 3. Wilson Lower Bound (most reliable method - provides statistical confidence even with low vote counts)
def wilson_lower_bound(up, down, confidence=0.95):
    n = up + down # Total number of observations
    if n == 0:
        return 0.0 # Return 0 if there are no votes
    
    p = up / n # Observed proportion of positive votes
    z = 1.96  # Z-score for 95% confidence level
    
    # Calculate the lower bound of the Wilson score interval
    lower_bound = (p + z**2/(2*n) - z * np.sqrt((p*(1-p) + z**2/(4*n))/n)) / (1 + z**2/n)
    return lower_bound


# Calculate positive - negative difference
df['score_pos_neg_diff'] = df.apply(
    lambda x: score_pos_neg_diff(x['helpful_yes'], x['helpful_no']), axis=1) # Apply simple difference to each row

# Calculate average rating ratio (up / total)
df['score_average_rating'] = df.apply(
    lambda x: score_average_rating(x['helpful_yes'], x['helpful_no']), axis=1) # Apply average ratio to each row

# Calculate Wilson Lower Bound (for most reliable ranking)
df['wilson_lower_bound'] = df.apply(
    lambda x: wilson_lower_bound(x['helpful_yes'], x['helpful_no']), axis=1) # Apply Wilson formula to each row

print("\n=== Top 20 Reviews - Based on score_pos_neg_diff ===") # Header for simple difference ranking
print(df.sort_values('score_pos_neg_diff', ascending=False)[ # Sort by positive-negative difference descending
    ['reviewText', 'overall', 'helpful_yes', 'helpful_no', 'total_vote', 'score_pos_neg_diff'] # Select relevant columns
].head(20)) # Display the first 20 records

print("\n=== Top 20 Reviews - Based on score_average_rating ===") # Header for average ratio ranking
print(df.sort_values('score_average_rating', ascending=False)[ # Sort by average rating ratio descending
    ['reviewText', 'overall', 'helpful_yes', 'helpful_no', 'total_vote', 'score_average_rating'] # Select relevant columns
].head(20)) # Display the first 20 records

print("\n=== Top 20 Reviews - Based on wilson_lower_bound (MOST RELIABLE) ===") # Header for Wilson ranking
print(df.sort_values('wilson_lower_bound', ascending=False)[ # Sort by Wilson Lower Bound score descending
    ['reviewText', 'overall', 'helpful_yes', 'helpful_no', 'total_vote', 'wilson_lower_bound'] # Select relevant columns
].head(20)) # Display the first 20 records
