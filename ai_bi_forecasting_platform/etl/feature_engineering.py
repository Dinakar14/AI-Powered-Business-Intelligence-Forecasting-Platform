import pandas as pd

df = pd.read_csv("data/processed/business_sales_clean.csv")

df["Date"] = pd.to_datetime(df["Date"])

# Time features
df["Month"] = df["Date"].dt.month
df["Year"] = df["Date"].dt.year
df["DayOfWeek"] = df["Date"].dt.dayofweek

# Business feature
df["Revenue_per_Unit"] = df["Revenue"] / df["Units_Sold"]

df.to_csv("data/processed/business_sales_features.csv", index=False)
print("Feature-engineered dataset ready!")
