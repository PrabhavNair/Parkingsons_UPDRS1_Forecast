import os
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from math import sqrt
import joblib

# ----------------------------
# Load data
# ----------------------------
df = pd.read_csv("train_clinical_data.csv")

# keep only rows with target
df = df[df["updrs_1"].notna()].copy()
# light imputation
df = df.ffill().bfill()

# ----------------------------
# Feature engineering
# ----------------------------
def fe(d):
    d = d.copy()
    d["visit_squared"]   = d["visit_month"] ** 2
    d["log_updrs3"]      = np.log1p(d["updrs_3"].clip(lower=0))
    d["updrs_4_squared"] = d["updrs_4"] ** 2
    d["interaction_234"] = d["updrs_2"] * d["updrs_3"] * d["updrs_4"]
    d["updrs3_x_updrs4"] = d["updrs_3"] * d["updrs_4"]
    d["updrs2_x_updrs3"] = d["updrs_2"] * d["updrs_3"]
    d["updrs2_x_updrs4"] = d["updrs_2"] * d["updrs_4"]
    d["visit_x_updrs2"]  = d["visit_month"] * d["updrs_2"]
    d["visit_x_updrs3"]  = d["visit_month"] * d["updrs_3"]
    d["visit_x_updrs4"]  = d["visit_month"] * d["updrs_4"]

    # rolling (global simple proxy – OK for small project)
    d["updrs_1_roll3"] = d["updrs_1"].rolling(window=3, min_periods=1).mean()
    d["updrs_2_roll3"] = d["updrs_2"].rolling(window=3, min_periods=1).mean()
    d["updrs_3_roll3"] = d["updrs_3"].rolling(window=3, min_periods=1).mean()

    d = d.dropna()
    return d

df = fe(df)

feature_cols = [
    "visit_month", "updrs_2", "updrs_3", "updrs_4",
    "updrs3_x_updrs4", "updrs2_x_updrs3", "updrs2_x_updrs4",
    "visit_x_updrs2", "visit_x_updrs3", "visit_x_updrs4",
    "visit_squared", "log_updrs3", "updrs_4_squared",
    "interaction_234", "updrs_1_roll3", "updrs_2_roll3", "updrs_3_roll3"
]

X_real = df[feature_cols].copy()
y_real = df["updrs_1"].values

# ----------------------------
# Add a small "healthy baseline" cohort (all 0s)
# ----------------------------
months = np.arange(0, 25)  # 0..24 months
healthy = pd.DataFrame({
    "visit_month": months,
    "updrs_1": np.zeros_like(months, dtype=float),
    "updrs_2": 0.0,
    "updrs_3": 0.0,
    "updrs_4": 0.0,
})
healthy = fe(healthy)

X_healthy = healthy[feature_cols].copy()
y_healthy = healthy["updrs_1"].values

# Concatenate and apply sample weights (upweight the rule)
X_all = pd.concat([X_real, X_healthy], axis=0).reset_index(drop=True)
y_all = np.concatenate([y_real, y_healthy], axis=0)
w_all = np.concatenate([
    np.ones_like(y_real),          # real rows weight = 1
    np.ones_like(y_healthy) * 3.0  # healthy-constraint rows weight = 3
])

# ----------------------------
# Train/test split
# ----------------------------
X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(
    X_all, y_all, w_all, test_size=0.2, random_state=42
)

# ----------------------------
# Monotonic constraints:
# +1 where increasing symptom should not decrease predicted UPDRS_1
#  0 for neutral/derived features where sign is unclear
# Order must match feature_cols
# ----------------------------
mono = [
    0,   # visit_month (let data learn; healthy cohort keeps baseline flat)
    +1,  # updrs_2
    +1,  # updrs_3
    +1,  # updrs_4
    +1,  # updrs3_x_updrs4
    +1,  # updrs2_x_updrs3
    +1,  # updrs2_x_updrs4
    +1,  # visit_x_updrs2
    +1,  # visit_x_updrs3
    +1,  # visit_x_updrs4
    0,   # visit_squared
    +1,  # log_updrs3
    +1,  # updrs_4_squared
    +1,  # interaction_234
    +1,  # updrs_1_roll3
    +1,  # updrs_2_roll3
    +1,  # updrs_3_roll3
]
mono_str = "(" + ",".join(str(int(m)) for m in mono) + ")"

# ----------------------------
# Model + hyperparameter search
# ----------------------------
xgb_base = xgb.XGBRegressor(
    objective="reg:squarederror",
    tree_method="hist",
    monotone_constraints=mono_str,
    random_state=42
)

param_grid = {
    "max_depth": [3, 4, 5],
    "learning_rate": [0.03, 0.05, 0.1],
    "n_estimators": [200, 300],
    "subsample": [0.8, 1.0],
    "colsample_bytree": [0.8, 1.0],
}

grid = GridSearchCV(
    xgb_base,
    param_grid,
    scoring="r2",
    cv=5,
    n_jobs=-1,
    verbose=1
)

grid.fit(X_train, y_train, sample_weight=w_train)

best = grid.best_estimator_
pred = best.predict(X_test)
rmse = sqrt(mean_squared_error(y_test, pred))
r2 = r2_score(y_test, pred)

print(f"[UPDRS_1 Model w/ healthy-constraint] RMSE: {rmse:.2f}")
print(f"[UPDRS_1 Model w/ healthy-constraint] R²: {r2:.4f}")
print("Best Parameters:", grid.best_params_)

# ----------------------------
# Save
# ----------------------------
os.makedirs("models", exist_ok=True)
# Save booster for Streamlit booster loader OR use joblib for sklearn API
best.get_booster().save_model("models/xgboost_updrs1_model.json")
joblib.dump(feature_cols, "models/updrs1_feature_columns.pkl")
print("Saved model to models/xgboost_updrs1_model.json")
