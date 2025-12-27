import os
import pandas as pd
import numpy as np

# -----------------------------
# FORCE CREATE DIRECTORIES
# -----------------------------
os.makedirs("data/raw", exist_ok=True)

# -----------------------------
# CONFIG
# -----------------------------
np.random.seed(42)
N_ROWS = 6000

START_DATE = "2022-01-01"
CATEGORIES = ["Electronics", "Grocery", "Fashion"]
REGIONS = ["North", "South", "East", "West"]

# -----------------------------
# DATE GENERATION
# -----------------------------
dates = pd.date_range(start=START_DATE, periods=N_ROWS, freq="D")

# -----------------------------
# CORE DATA
# -----------------------------
data = pd.DataFrame({
    "Date": dates,
    "Product_ID": np.random.randint(1000, 1100, N_ROWS),
    "Category": np.random.choice(CATEGORIES, N_ROWS, p=[0.35, 0.40, 0.25]),
    "Region": np.random.choice(REGIONS, N_ROWS),
    "Units_Sold": np.random.poisson(lam=20, size=N_ROWS),
    "Customer_ID": np.random.randint(2000, 3000, N_ROWS),
    "Customer_Age": np.random.randint(18, 65, N_ROWS),
    "Purchase_Frequency": np.random.randint(1, 15, N_ROWS)
})

# -----------------------------
# PRICE LOGIC
# -----------------------------
price_map = {
    "Electronics": 1200,
    "Grocery": 150,
    "Fashion": 800
}

data["Unit_Price"] = data["Category"].map(price_map)

# -----------------------------
# REVENUE
# -----------------------------
data["Revenue"] = data["Units_Sold"] * data["Unit_Price"]

# -----------------------------
# CHURN LOGIC
# -----------------------------
data["Churn"] = np.where(
    (data["Purchase_Frequency"] <= 3) &
    (data["Units_Sold"] <= 10),
    1,
    0
)

# -----------------------------
# DROP HELPER COLUMN
# -----------------------------
data.drop(columns=["Unit_Price"], inplace=True)

# -----------------------------
# SAVE FILE
# -----------------------------
output_path = "data/raw/business_sales_raw.csv"
data.to_csv(output_path, index=False)

print("✅ Dataset generated successfully")
print("📁 Saved at:", output_path)
print("📊 Shape:", data.shape)
