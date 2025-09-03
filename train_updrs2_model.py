import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import os

# === 1. Load and clean data ===
df = pd.read_csv("train_clinical_data.csv")
df = df.fillna(method="ffill").fillna(method="bfill")

# Fill medication status
df["upd23b_clinical_state_on_medication"] = df["upd23b_clinical_state_on_medication"].fillna("Unknown")

# === 2. Feature Engineering ===
df["updrs3_x_updrs4"] = df["updrs_3"] * df["updrs_4"]
df["updrs2_x_updrs3"] = df["updrs_2"] * df["updrs_3"]
df["updrs2_x_updrs4"] = df["updrs_2"] * df["updrs_4"]
df["visit_x_updrs3"] = df["visit_month"] * df["updrs_3"]
df["visit_x_updrs4"] = df["visit_month"] * df["updrs_4"]
df["visit_squared"] = df["visit_month"] ** 2
df["updrs3_squared"] = df["updrs_3"] ** 2
df["updrs_4_squared"] = df["updrs_4"] ** 2

# One-hot encode medication state
df = pd.get_dummies(df, columns=["upd23b_clinical_state_on_medication"], drop_first=True)

# === 3. Define Features/Target ===
feature_cols = [
    "visit_month", "updrs_3", "updrs_4",
    "updrs3_x_updrs4", "updrs2_x_updrs3", "updrs2_x_updrs4",
    "visit_x_updrs3", "visit_x_updrs4", "visit_squared",
    "updrs3_squared", "updrs_4_squared"  
] + [col for col in df.columns if col.startswith("upd23b_clinical_state_on_medication_")]

X = df[feature_cols]
y = df["updrs_2"]

# === 4. Split Data ===
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# === 5. Define & Tune Model ===
param_grid = {
    'max_depth': [3, 4, 5],
    'learning_rate': [0.01, 0.05, 0.1],
    'n_estimators': [100, 200],
    'subsample': [0.8, 1.0]
}
model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)
grid = GridSearchCV(model, param_grid, scoring='r2', cv=5, verbose=1)
grid.fit(X_train, y_train)

# === 6. Evaluate Model ===
best_model = grid.best_estimator_
preds = best_model.predict(X_test)
rmse = mean_squared_error(y_test, preds) ** 0.5
r2 = r2_score(y_test, preds)

print(f"[UPDRS_2 Model] RMSE: {rmse:.2f}")
print(f"[UPDRS_2 Model] R²: {r2:.2f}")
print(f"Best Parameters: {grid.best_params_}")

# === 7. Save Model ===
os.makedirs("models", exist_ok=True)
best_model.save_model("models/xgboost_updrs2_model.json")
joblib.dump(feature_cols, "models/updrs2_feature_columns.pkl")

# Save model
best_model = grid.best_estimator_
best_model.get_booster().save_model("models/xgboost_updrs2_model.json")
print("Model saved to models/xgboost_updrs2_model.json")

