import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
from models.utils import save_model

df = pd.read_csv("data/processed/business_sales_features.csv")

# Risk labeling logic
df["Risk_Level"] = "Low"
df.loc[df["Revenue"] < df["Revenue"].quantile(0.25), "Risk_Level"] = "High"
df.loc[
    (df["Revenue"] >= df["Revenue"].quantile(0.25)) &
    (df["Revenue"] < df["Revenue"].quantile(0.50)),
    "Risk_Level"
] = "Medium"

features = ["Units_Sold", "Revenue", "Purchase_Frequency", "Month"]

X = df[features]
y = df["Risk_Level"]

le = LabelEncoder()
y = le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = XGBClassifier(
    n_estimators=150,
    max_depth=6,
    learning_rate=0.1,
    eval_metric="mlogloss"
)

model.fit(X_train, y_train)

preds = model.predict(X_test)
acc = accuracy_score(y_test, preds)

print("Revenue Risk Accuracy:", acc)

save_model(model, "artifacts/risk_model.pkl")
save_model(le, "artifacts/risk_label_encoder.pkl")
