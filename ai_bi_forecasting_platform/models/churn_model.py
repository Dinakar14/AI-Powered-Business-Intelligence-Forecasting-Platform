import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, f1_score
from sklearn.ensemble import RandomForestClassifier
from models.utils import save_model

df = pd.read_csv("data/processed/business_sales_features.csv")

features = [
    "Units_Sold", "Revenue", "Customer_Age",
    "Purchase_Frequency", "Month", "DayOfWeek"
]

X = df[features]
y = df["Churn"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    random_state=42
)

model.fit(X_train, y_train)

preds = model.predict(X_test)
f1 = f1_score(y_test, preds)

print("Churn Prediction F1-score:", f1)
print(classification_report(y_test, preds))

save_model(model, "artifacts/churn_model.pkl")
save_model(scaler, "artifacts/churn_scaler.pkl")
