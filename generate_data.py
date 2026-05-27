"""
generate_data.py
Generates a realistic synthetic retail sales dataset.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import random

np.random.seed(42)
random.seed(42)

REGIONS     = ["North", "South", "East", "West"]
CATEGORIES  = ["Electronics", "Clothing", "Groceries", "Home & Kitchen", "Sports"]
PRODUCTS    = {
    "Electronics":    ["Laptop", "Smartphone", "Headphones", "Tablet", "Camera"],
    "Clothing":       ["T-Shirt", "Jeans", "Jacket", "Dress", "Shoes"],
    "Groceries":      ["Rice", "Cooking Oil", "Flour", "Sugar", "Pulses"],
    "Home & Kitchen": ["Mixer", "Cookware Set", "Bedsheet", "Curtains", "Water Purifier"],
    "Sports":         ["Cricket Bat", "Football", "Yoga Mat", "Dumbbells", "Cycle"],
}
BASE_PRICE  = {
    "Electronics": 25000, "Clothing": 1200, "Groceries": 300,
    "Home & Kitchen": 3500, "Sports": 2000,
}
CHANNELS    = ["Online", "Retail Store", "Wholesale"]
DISCOUNT_MAP = {"Online": 0.12, "Retail Store": 0.05, "Wholesale": 0.18}

def seasonal_factor(month):
    # Higher sales in Oct-Dec (festive), dip in Feb-Mar
    factors = {1:0.85, 2:0.78, 3:0.80, 4:0.88, 5:0.90, 6:0.92,
               7:0.95, 8:0.97, 9:1.05, 10:1.20, 11:1.35, 12:1.40}
    return factors[month]

def generate_dataset(n_rows=3000):
    rows = []
    start = datetime(2021, 1, 1)

    for _ in range(n_rows):
        date      = start + timedelta(days=random.randint(0, 3*365))
        region    = random.choice(REGIONS)
        category  = random.choice(CATEGORIES)
        product   = random.choice(PRODUCTS[category])
        channel   = random.choice(CHANNELS)

        base      = BASE_PRICE[category]
        price     = round(base * np.random.uniform(0.7, 1.4), 2)
        units     = int(np.random.poisson(lam=8) + 1)
        discount  = DISCOUNT_MAP[channel] + np.random.uniform(-0.03, 0.03)
        discount  = round(max(0, min(discount, 0.35)), 3)

        sf        = seasonal_factor(date.month)
        units     = max(1, int(units * sf * np.random.uniform(0.8, 1.2)))
        revenue   = round(price * units * (1 - discount), 2)

        # Derived features
        profit_margin = round(np.random.uniform(0.10, 0.35), 3)
        profit        = round(revenue * profit_margin, 2)
        returns_pct   = round(np.random.uniform(0.0, 0.08), 3)
        net_revenue   = round(revenue * (1 - returns_pct), 2)

        rows.append({
            "date":           date.strftime("%Y-%m-%d"),
            "year":           date.year,
            "month":          date.month,
            "quarter":        (date.month - 1) // 3 + 1,
            "day_of_week":    date.weekday(),
            "region":         region,
            "category":       category,
            "product":        product,
            "channel":        channel,
            "unit_price":     price,
            "units_sold":     units,
            "discount_pct":   discount,
            "revenue":        revenue,
            "profit_margin":  profit_margin,
            "profit":         profit,
            "returns_pct":    returns_pct,
            "net_revenue":    net_revenue,
        })

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    return df

if __name__ == "__main__":
    df = generate_dataset(3000)
    df.to_csv("data/sales_data.csv", index=False)
    print(f"Dataset created: {df.shape[0]} rows × {df.shape[1]} columns")
    print(df.head())
