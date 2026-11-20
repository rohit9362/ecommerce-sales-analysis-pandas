import pandas as pd
import random

# 1. GENERATE REALISTIC MESSY DATASET (50,000 ROWS) USING PURE PYTHON & PANDAS
random.seed(42)
n_rows = 50000

categories = ["Electronics", "electronics ", "Fashion", "FASHION", "Home & Kitchen", "Books", "Beauty"]
cities = ["Bengaluru", "Mumbai", "Delhi NCR", "Hyderabad", "Pune", "Kolkata", "Chennai", None]
dates = ["2026-01-15", "15/02/2026", "2026-03-22", "2026/04/10", "05-05-2026", None]
quantities = [1, 1, 1, 2, 2, 3, 4, -1, None]
delivery_options = [2, 3, 3, 4, 5, 7, 10, None]
statuses = ["Delivered", "Delivered", "Delivered", "Returned", "Cancelled"]

data = {
    "order_id": [f"ORD-{100000 + i}" for i in range(n_rows)],
    "order_date": [random.choice(dates) for _ in range(n_rows)],
    "customer_id": [random.randint(1001, 8500) for _ in range(n_rows)],
    "city": [random.choice(cities) for _ in range(n_rows)],
    "category": [random.choice(categories) for _ in range(n_rows)],
    "quantity": [random.choice(quantities) for _ in range(n_rows)],
    "unit_price_inr": [round(random.uniform(299, 25000), 2) for _ in range(n_rows)],
    "delivery_days": [random.choice(delivery_options) for _ in range(n_rows)],
    "order_status": [random.choice(statuses) for _ in range(n_rows)]
}

df = pd.DataFrame(data)

# Inject 500 duplicate rows intentionally
df = pd.concat([df, df.sample(500, random_state=42)], ignore_index=True)
df.to_csv("raw_ecommerce_orders.csv", index=False)
print(f"Raw dataset created with shape: {df.shape}")

# ==========================================
# 2. DATA CLEANING & PREPROCESSING (PURE PANDAS)
# ==========================================

# A. Remove exact duplicate orders
df = df.drop_duplicates(subset=["order_id"], keep="first").copy()

# B. Standardize text formatting (strip spaces & Title Case)
df["category"] = df["category"].str.strip().str.title()

# C. Handle missing values
df["city"] = df["city"].fillna("Unknown")
df["quantity"] = df["quantity"].fillna(1)
df["delivery_days"] = df["delivery_days"].fillna(df["delivery_days"].median())

# D. Filter out invalid negative quantities & drop rows with missing/invalid dates
df = df[df["quantity"] > 0]
df["order_date"] = pd.to_datetime(df["order_date"], format="mixed", dayfirst=True, errors="coerce")
df = df.dropna(subset=["order_date"])

# E. Feature Engineering using pure Pandas
df["total_amount_inr"] = (df["quantity"] * df["unit_price_inr"]).round(2)
df["is_delayed"] = df["delivery_days"].gt(5).map({True: "Delayed (>5 Days)", False: "On-Time"})

df.to_csv("cleaned_ecommerce_orders.csv", index=False)
print(f"Cleaned dataset saved with shape: {df.shape}\n")

# ==========================================
# 3. EXPLORATORY DATA ANALYSIS (PURE PANDAS)
# ==========================================

# Insight 1: Revenue & Order Volume by Category (Delivered Orders Only)
delivered_df = df[df["order_status"] == "Delivered"]
category_summary = (
    delivered_df.groupby("category")
    .agg(
        total_orders=("order_id", "count"),
        total_revenue_lakhs=("total_amount_inr", lambda x: round(x.sum() / 100000, 2)),
        avg_order_value=("total_amount_inr", lambda x: round(x.mean(), 2))
    )
    .sort_values(by="total_revenue_lakhs", ascending=False)
)
print("--- 1. Category-Wise Revenue Summary (Delivered Orders) ---")
print(category_summary, "\n")

# Insight 2: Top 5 Cities by Revenue
city_summary = (
    delivered_df.groupby("city")["total_amount_inr"]
    .sum()
    .div(100000)
    .round(2)
    .sort_values(ascending=False)
    .head(5)
)
print("--- 2. Top 5 Cities by Revenue (in INR Lakhs) ---")
print(city_summary, "\n")

# Insight 3: Impact of Delivery Delays on Order Return/Cancellation Rate
delay_impact = pd.crosstab(df["is_delayed"], df["order_status"], normalize="index").mul(100).round(2)
print("--- 3. Delivery Speed vs. Order Status (%) ---")
print(delay_impact)