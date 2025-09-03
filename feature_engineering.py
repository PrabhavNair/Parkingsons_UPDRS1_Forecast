import numpy as np
import pandas as pd

def create_features(df):
    df['visit_squared'] = df['visit_month'] ** 2
    df['updrs_4_squared'] = df['updrs_4'] ** 2
    df['updrs3_squared'] = df['updrs_3'] ** 2
    df['log_updrs3'] = np.log1p(df['updrs_3'])

    df['interaction_234'] = df['updrs_2'] * df['updrs_3'] * df['updrs_4']
    df['visit_x_updrs2'] = df['visit_month'] * df['updrs_2']
    df['visit_x_updrs3'] = df['visit_month'] * df['updrs_3']
    df['visit_x_updrs4'] = df['visit_month'] * df['updrs_4']

    # Add rolling features
    df['updrs_2_roll3'] = df['updrs_2'].rolling(window=3, min_periods=1).mean()
    df['updrs_3_roll3'] = df['updrs_3'].rolling(window=3, min_periods=1).mean()

    # Add placeholder one-hot columns for medication status
    df['upd23b_clinical_state_on_medication_Off'] = 0
    df['upd23b_clinical_state_on_medication_On'] = 1
    df['upd23b_clinical_state_on_medication_Unknown'] = 0

    return df
