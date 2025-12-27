import pandas as pd
from sklearn.metrics import mean_absolute_percentage_error
from prophet import Prophet
from models.utils import save_model

# Load data
df = pd.read_csv("data/processed/business_sales_features.csv")

# Aggregate revenue by date
ts = df.groupby("Date")["Revenue"].sum().reset_index()
ts.columns = ["ds", "y"]

# Train-test split
train = ts.iloc[:-180]
test = ts.iloc[-180:]

# Model
model = Prophet()
model.fit(train)

# Forecast
future = model.make_future_dataframe(periods=180)
forecast = model.predict(future)

# Evaluation
y_true = test["y"].values
y_pred = forecast.iloc[-180:]["yhat"].values

mape = mean_absolute_percentage_error(y_true, y_pred)
print(f"Sales Forecasting MAPE: {mape:.2%}")

# Save model
save_model(model, "artifacts/sales_forecast_model.pkl")
