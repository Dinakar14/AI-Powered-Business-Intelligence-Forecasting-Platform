import os
import pandas as pd

# -----------------------------
# FORCE CREATE DIRECTORY
# -----------------------------
os.makedirs("data/processed", exist_ok=True)

# -----------------------------
# LOAD RAW DATA
# -----------------------------
input_path = "data/raw/business_sales_raw.csv"
df = pd.read_csv(input_path)

# -----------------------------
# DATE CONVERSION
# -----------------------------
df["Date"] = pd.to_datetime(df["Date"])

# -----------------------------
# HANDLE MISSING VALUES
# -----------------------------
df.fillna({
    "Units_Sold": df["Units_Sold"].median(),
    "Revenue": df["Revenue"].median(),
    "Purchase_Frequency": df["Purchase_Frequency"].median()
}, inplace=True)

# -----------------------------
# OUTLIER CAPPING
# -----------------------------
for col in ["Units_Sold", "Revenue"]:
    upper = df[col].quantile(0.99)
    df[col] = df[col].clip(upper=upper)

# -----------------------------
# SAVE CLEAN DATA
# -----------------------------
output_path = "data/processed/business_sales_clean.csv"
df.to_csv(output_path, index=False)

print("✅ Cleaned dataset saved successfully")
print("📁 Saved at:", output_path)
