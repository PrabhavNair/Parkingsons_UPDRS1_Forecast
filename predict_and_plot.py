import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
import joblib

# Load and clean dataset
df = pd.read_csv('train_clinical_data.csv')
df.ffill(inplace=True)
df.bfill(inplace=True)

# Feature engineering
df['updrs3_x_updrs4'] = df['updrs_3'] * df['updrs_4']
df['updrs2_x_updrs3'] = df['updrs_2'] * df['updrs_3']
df['updrs2_x_updrs4'] = df['updrs_2'] * df['updrs_4']
df['visit_x_updrs3'] = df['visit_month'] * df['updrs_3']
df['visit_x_updrs4'] = df['visit_month'] * df['updrs_4']
df['visit_x_updrs2'] = df['visit_month'] * df['updrs_2']
df['visit_squared'] = df['visit_month'] ** 2
df['log_updrs3'] = np.log1p(df['updrs_3'])
df['updrs_4_squared'] = df['updrs_4'] ** 2
df['interaction_234'] = df['updrs_2'] * df['updrs_3'] * df['updrs_4']
df['patient_id_plot'] = df['patient_id']
df['patient_id_str'] = df['patient_id'].astype(str)
df = df.sort_values(by=['patient_id_plot', 'visit_month'])
df['updrs_1_roll3'] = df.groupby('patient_id_plot')['updrs_1'].transform(lambda x: x.rolling(window=3, min_periods=1).mean())
df['updrs_2_roll3'] = df.groupby('patient_id_plot')['updrs_2'].transform(lambda x: x.rolling(window=3, min_periods=1).mean())
df['updrs_3_roll3'] = df.groupby('patient_id_plot')['updrs_3'].transform(lambda x: x.rolling(window=3, min_periods=1).mean())

# One-hot encode medication status
df = pd.get_dummies(df, columns=['upd23b_clinical_state_on_medication'])

# Save feature columns for classifier usage
from sklearn.preprocessing import LabelEncoder

clf_features = [
    'visit_month', 'updrs_2', 'updrs_4',
    'updrs3_x_updrs4', 'updrs2_x_updrs3', 'updrs2_x_updrs4',
    'visit_x_updrs2', 'visit_x_updrs3', 'visit_x_updrs4',
    'visit_squared', 'log_updrs3', 'updrs_4_squared', 'interaction_234',
    'updrs_1_roll3', 'updrs_2_roll3', 'updrs_3_roll3',
    'upd23b_clinical_state_on_medication_Off',
    'upd23b_clinical_state_on_medication_On',
    'upd23b_clinical_state_on_medication_Unknown'
]

clf_target = 'updrs_3'

bins = [0, 10, 20, 30, 80]
labels = ['none-slight', 'mild', 'moderate', 'severe']
df['updrs_3_severity'] = pd.cut(df[clf_target], bins=bins, labels=labels, include_lowest=True)
label_encoder = LabelEncoder()
df['updrs_3_encoded'] = label_encoder.fit_transform(df['updrs_3_severity'])

X_clf = df[clf_features]
y_clf = df['updrs_3_encoded']

X_train, X_test, y_train, y_test = train_test_split(X_clf, y_clf, test_size=0.2, random_state=42)

# Train classifier
clf = xgb.XGBClassifier(objective='multi:softprob', num_class=4, eval_metric='mlogloss')
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [3, 4],
    'learning_rate': [0.1, 0.05]
}
cv = KFold(n_splits=3, shuffle=True, random_state=42)
grid = GridSearchCV(clf, param_grid, cv=cv)
grid.fit(X_train, y_train)

# Save model and encoders
grid.best_estimator_.save_model("models/xgboost_updrs3_classifier.json")
joblib.dump(clf_features, "models/updrs3_feature_columns.pkl")
label_map = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
joblib.dump(label_map, "models/updrs3_label_map.pkl")

# Confirm success
from sklearn.metrics import classification_report, confusion_matrix
print("Accuracy:", grid.score(X_test, y_test))
print("\nClassification Report:\n", classification_report(y_test, grid.predict(X_test), target_names=label_encoder.classes_))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, grid.predict(X_test)))
