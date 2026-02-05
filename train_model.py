import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score, brier_score_loss, mean_squared_error

# 1. Load Data
df = pd.read_csv('collections_data.csv')

# 2. Feature Engineering (The "Inputs")
features = ['age', 'annual_income', 'days_past_due', 'outstanding_amount']
X = df[features]
y_default = df['default_flag'] # Target 1: Did they default?
y_lgd = df['true_lgd']         # Target 2: How much did we lose? (Only for defaulters)

# Split Data (80% Train, 20% Test)
X_train, X_test, y_train, y_test = train_test_split(X, y_default, test_size=0.2, random_state=42)

# --- MODEL 1: PROBABILITY OF DEFAULT (PD) ---

# Train a base model (Random Forest)
rf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
rf.fit(X_train, y_train)

# CALIBRATION (Crucial Step!)
# Raw Random Forests output "scores", not real probabilities. 
# We "calibrate" them so 0.7 means actually 70% risk.
calibrated_rf = CalibratedClassifierCV(rf, method='isotonic', cv=3)
calibrated_rf.fit(X_train, y_train)

# Predict Probabilities
probs = calibrated_rf.predict_proba(X_test)[:, 1]

# Evaluate
auc = roc_auc_score(y_test, probs)
print(f"PD Model AUC: {auc:.3f} (Should be > 0.70)")

# --- MODEL 2: LOSS GIVEN DEFAULT (LGD) ---
# We only train this on people who ACTUALLY defaulted.
mask_train = y_train == 1
X_train_lgd = X_train[mask_train]
y_train_lgd = y_lgd.iloc[X_train_lgd.index] # Match indices

lgd_model = RandomForestRegressor(n_estimators=50, max_depth=3, random_state=42)
lgd_model.fit(X_train_lgd, y_train_lgd)

# Evaluate (RMSE)
# Create a mask for the test set to evaluate LGD
mask_test = y_test == 1
X_test_lgd = X_test[mask_test]
y_test_lgd = y_lgd.iloc[X_test_lgd.index]

preds_lgd = lgd_model.predict(X_test_lgd)
rmse = np.sqrt(mean_squared_error(y_test_lgd, preds_lgd))
print(f"LGD Model RMSE: {rmse:.3f}")

# --- SAVE PREDICTIONS FOR THE OPTIMIZER ---
# We need these predictions to run the optimization in Step 3
df['pred_pd'] = calibrated_rf.predict_proba(X)[:, 1]
df['pred_lgd'] = lgd_model.predict(X)

# Save to a new CSV for the next step
df.to_csv('modeled_data.csv', index=False)
print("Success! Models trained and 'modeled_data.csv' saved.")