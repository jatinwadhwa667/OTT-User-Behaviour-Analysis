"""
======================================================================
PROJECT: OTT USER BEHAVIOUR ANALYSIS
======================================================================

Project Overview:
My project analyzes user watching habits on an OTT platform like Amazon Prime Video. It helps us understand user preferences, popular devices, and subscription plans. These insights can help improve customer satisfaction and reduce customer cancellations.

Dataset Information:
Dataset name: ott_user_behavior.csv
It contains user details and viewing information, including Age, Gender, City, Device, Genre, Watch Hours per Week, Rating Given, Subscription Type, and Subscription Months.

Tools Used:
- Pandas: Data loading, cleaning, grouping, and manipulation
- NumPy: Statistical calculations and percentile thresholding
- Matplotlib: Data visualization and charting
======================================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ============================================================
# 1. INTRODUCTION & DATA LOADING
# ============================================================
print("\n" + "="*60)
print("1. INTRODUCTION ABOUT DATASET & LOADING")
print("="*60)

DATASET_PATH = Path("ott_user_behavior.csv")
OUTPUT_DIR = Path("ott_project_output")
OUTPUT_DIR.mkdir(exist_ok=True)

df = pd.read_csv(DATASET_PATH)
print("\nDataset loaded successfully. Shape:", df.shape)

# ============================================================
# 2. DATA OVERVIEW AND CLEANING
# ============================================================
print("\n" + "="*60)
print("2. DATA OVERVIEW AND CLEANING")
print("="*60)

# Create a detailed summary table
summary_df = pd.DataFrame({
    "Column Name": df.columns,
    "Data Type": df.dtypes,
    "Non-Null Count": df.count(),
    "Null Values": df.isnull().sum(),
    "Unique Values": df.nunique()
})
print("\n--- Initial Dataset Summary ---")
print(summary_df)

# --- Cleaning Process ---
# 1. Convert numeric columns
for col in ["age", "watch_hours_per_week", "rating_given", "subscription_months"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# 2. Clean text columns (Standardize capitalization and remove spaces)
text_columns = ["gender", "city", "device", "genre", "subscription_type"]
for col in text_columns:
    df[col] = df[col].astype(str).str.strip().str.title()

# 3. Replace typos with standardized values
corrections = {
    "subscription_type": {"Std": "Standard", "Nan": np.nan},
    "genre": {"Documentry": "Documentary", "Nan": np.nan},
    "device": {"Phone": "Mobile", "Smart Tv": "Tv", "Nan": np.nan}
}
df = df.replace(corrections)

# 4. Drop rows missing critical analysis data
analysis_df = df.dropna(subset=[
    "age", "watch_hours_per_week", "rating_given", 
    "subscription_months", "genre", "subscription_type"
]).copy()

print("\nData cleaning complete. Usable rows:", len(analysis_df))

# ============================================================
# 3. ANALYSING OVERALL GENRE SUCCESS RATE
# ============================================================
print("\n" + "="*60)
print("3. ANALYSING OVERALL CONTENT PREFERENCES (GENRE)")
print("="*60)

# Calculate percentage distribution of genres
df_genre_success = analysis_df['genre'].value_counts(normalize=True) * 100
print("\nGenre Distribution (%):")
print(df_genre_success.round(2).to_string())

# Visualization
plt.figure(figsize=(8, 5))
df_genre_success.plot(kind='bar', color='skyblue', edgecolor='black')
plt.title('Overall Genre Popularity Comparison (%)')
plt.xlabel('Genre')
plt.ylabel('Percentage of Users (%)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "genre_success_rate.png", dpi=300)
plt.close()

top_genre = df_genre_success.index[0]
print(f"\n---> INSIGHT: It clearly shows that '{top_genre}' has the highest success rate and popularity among users.")

# ============================================================
# 4. ANALYSING WATCH DURATION IMPACT BY SUBSCRIPTION
# ============================================================
print("\n" + "="*60)
print("4. ANALYSING WATCH DURATION IMPACT BY SUBSCRIPTION")
print("="*60)

df_duration = analysis_df.groupby('subscription_type')['watch_hours_per_week'].mean().sort_values(ascending=False)
print("\nAverage Watch Hours by Subscription Type:")
print(df_duration.round(2).to_string())

# Visualization
plt.figure(figsize=(8, 5))
df_duration.plot(kind='bar', color='teal', edgecolor='black')
plt.title('Average Watch Duration by Subscription Tier')
plt.xlabel('Subscription Type')
plt.ylabel('Average Watch Hours / Week')
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "duration_by_subscription.png", dpi=300)
plt.close()

top_sub = df_duration.index[0]
print(f"\n---> INSIGHT: Having a '{top_sub}' subscription shows a person is more likely to spend longer duration watching content.")

# ============================================================
# 5. AGE AS A DECIDING FACTOR
# ============================================================
print("\n" + "="*60)
print("5. AGE AS DECIDING FACTOR")
print("="*60)

# Create age groups exactly binning strategy
bins = [0, 25, 45, 100]
labels = ['Young Adult (0-25)', 'Adult (26-45)', 'Senior (46+)']
analysis_df['age_group'] = pd.cut(analysis_df['age'], bins=bins, labels=labels, right=False)

df_age_group = analysis_df['age_group'].value_counts(normalize=True) * 100
print("\nUser Distribution by Age Group (%):")
print(df_age_group.round(2).to_string())

# Visualization
plt.figure(figsize=(8, 5))
df_age_group.plot(kind='pie', autopct='%1.1f%%', colors=['salmon', 'skyblue', 'lightgreen'])
plt.title('Percentage of Users by Age Group')
plt.ylabel('') # Hide the y-label for pie charts
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "age_group_distribution.png", dpi=300)
plt.close()

top_age = df_age_group.index[0]
print(f"\n---> INSIGHT: '{top_age}' users are much more likely to be enrolled in and actively use the platform.")

# ============================================================
# 6. IMPACT OF DEVICES ON PREMIUM ACCOUNTS
print("\n" + "=" * 60)
print("6. IMPACT OF DEVICES ON PREMIUM ACCOUNTS")
print("=" * 60)

# Filter Premium users
premium_df = analysis_df[
    analysis_df['subscription_type'] == 'Premium'
]

# Calculate device preference percentages
df_devices = premium_df['device'].value_counts(normalize=True) * 100

print("\nPremium User Device Preferences (%):")
print(df_devices.round(2).to_string())

# Identify the most common device
top_devices = df_devices[df_devices == df_devices.max()].index.tolist()

print(f"\n---> INSIGHT: {', '.join(top_devices)} are the most commonly used devices among Premium users, with {df_devices.max():.2f}% each.")

# Visualization
plt.figure(figsize=(8, 5))
df_devices.plot(kind='bar', color='skyblue', edgecolor='black')

plt.title('Device Preferences Among Premium Users (%)')
plt.xlabel('Device')
plt.ylabel('Percentage of Premium Users (%)')
plt.xticks(rotation=45)
plt.tight_layout()

plt.savefig(OUTPUT_DIR / "premium_device_preferences.png", dpi=300)
plt.close()

# ============================================================
# 7. IDENTIFYING AT-RISK CUSTOMERS (LOW ENGAGEMENT)
# ============================================================
print("\n" + "="*60)
print("7. CHURN RISK ANALYSIS (LOW ENGAGEMENT DEFAULTERS)")
print("="*60)

# Using the 25th percentile as the "defaulter/risk" threshold
risk_threshold = np.percentile(analysis_df["watch_hours_per_week"].dropna(), 25)
at_risk_users = analysis_df[analysis_df["watch_hours_per_week"] <= risk_threshold]
risk_percentage = (len(at_risk_users) / len(analysis_df)) * 100

print(f"Risk Threshold: Bottom 25% watch <= {risk_threshold:.2f} hours/week.")
print(f"Total At-Risk Users: {len(at_risk_users)} ({risk_percentage:.1f}% of base)")

print(f"\n---> INSIGHT: Being a low-engagement user (<= {risk_threshold:.2f} hrs/wk) heavily increases chances of subscription cancellation.")

# Visualization
other_users = len(analysis_df) - len(at_risk_users)

engagement_data = pd.Series({
    "At-Risk Users": len(at_risk_users),
    "Other Users": other_users
})

plt.figure(figsize=(8, 5))

bars = plt.bar(
    engagement_data.index,
    engagement_data.values,
    color=["salmon", "skyblue"],
    edgecolor="black"
)

# Display values above bars
for bar in bars:
    height = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        height,
        f"{int(height)}",
        ha="center",
        va="bottom"
    )

plt.title("Low-Engagement vs Other Users")
plt.xlabel("User Category")
plt.ylabel("Number of Users")
plt.tight_layout()

plt.savefig(OUTPUT_DIR / "churn_risk_analysis.png", dpi=300)
plt.close()


# ============================================================
# 8. KEY INSIGHTS & STRATEGY
# ============================================================

print("\n" + "=" * 60)
print("8. KEY INSIGHTS")
print("=" * 60)

print(f"- Most popular genre: {top_genre}")
print(f"- Subscription with highest watch time: {top_sub}")
print(f"- Largest age group: {top_age}")
print(f"- Most popular devices among Premium users: {', '.join(top_devices)}")
print(f"- Low-engagement users watch {risk_threshold:.2f} hours or less per week.")

print(f"\nAll charts are saved in the '{OUTPUT_DIR.name}' folder.")