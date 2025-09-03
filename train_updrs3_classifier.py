# train_updrs3_forecast.py  — one-step-ahead forecaster (leak-free, patient-aware)
import os, joblib, numpy as np, pandas as pd, xgboost as xgb
from math import sqrt
from sklearn.model_selection import GroupShuffleSplit, GridSearchCV, KFold
from sklearn.metrics import mean_squared_error, r2_score

df = pd.read_csv("train_clinical_data.csv")
df = df.ffill().bfill()
df["upd23b_clinical_state_on_medication"] = df["upd23b_clinical_state_on_medication"].fillna("Unknown")

# Sort per patient for proper lags
df = df.sort_values(["patient_id", "visit_month"])

# --- Lagged features (only past info) ---
for col in ["updrs_2", "updrs_3", "updrs_4"]:
    df[f"{col}_lag1"] = df.groupby("patient_id")[col].shift(1)
    df[f"{col}_roll3"] = df.groupby("patient_id")[col].transform(lambda s: s.shift(1).rolling(3, min_periods=1).mean())
    df[f"{col}_diff1"] = df[col] - df.groupby("patient_id")[col].shift(1)

# Time since baseline (helps model slope)
df["months_since_first"] = df["visit_month"] - df.groupby("patient_id")["visit_month"].transform("min")

# One-hot med state at time t
df = pd.get_dummies(df, columns=["upd23b_clinical_state_on_medication"], drop_first=True)

# Target = next visit's UPDRS_3
df["updrs_3_next"] = df.groupby("patient_id")["updrs_3"].shift(-1)

# Keep rows where we have full past (lag1) and a next target
df = df.dropna(subset=["updrs_3_next", "updrs_2_lag1", "updrs_3_lag1", "updrs_4_lag1"])

feature_cols = [
    "visit_month", "months_since_first",
    "updrs_2_lag1", "updrs_3_lag1", "updrs_4_lag1",
    "updrs_2_roll3", "updrs_3_roll3", "updrs_4_roll3",
    "updrs_2_diff1", "updrs_3_diff1", "updrs_4_diff1",
] + [c for c in df.columns if c.startswith("upd23b_clinical_state_on_medication_")]

X = df[feature_cols]
y = df["updrs_3_next"]
groups = df["patient_id"]

# Grouped train/val split (no patient leakage)
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, test_idx = next(gss.split(X, y, groups=groups))
X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

# Model + tuning
param_grid = {
    "max_depth": [3, 4, 5],
    "learning_rate": [0.03, 0.05, 0.1],
    "n_estimators": [200, 400],
    "subsample": [0.8, 1.0],
    "colsample_bytree": [0.8, 1.0],
}
model = xgb.XGBRegressor(objective="reg:squarederror", random_state=42)
cv = KFold(n_splits=5, shuffle=True, random_state=42)

grid = GridSearchCV(model, param_grid, scoring="r2", cv=cv, verbose=1, n_jobs=-1)
grid.fit(X_train, y_train)

best = grid.best_estimator_
preds = best.predict(X_test)

rmse = sqrt(mean_squared_error(y_test, preds))
r2 = r2_score(y_test, preds)

print(f"[UPDRS_3 Forecast] RMSE: {rmse:.2f}")
print(f"[UPDRS_3 Forecast] R²: {r2:.4f}")
print(f"Best Parameters: {grid.best_params_}")

os.makedirs("models", exist_ok=True)
best.get_booster().save_model("models/xgboost_updrs3_forecast.json")
joblib.dump(feature_cols, "models/updrs3_forecast_feature_columns.pkl")
print("Saved model to models/xgboost_updrs3_forecast.json")
