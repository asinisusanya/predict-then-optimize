import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from pathlib import Path
import joblib

# -----------------------------
# 1. PATHS
# -----------------------------

TRAIN_PATH = "../../data/processed/train_fe.csv"
TEST_PATH = "../../data/processed/test_fe.csv"

MODEL_DIR = "../../results/models"
PRED_DIR = "../../results/predictions"

Path(MODEL_DIR).mkdir(parents=True, exist_ok=True)
Path(PRED_DIR).mkdir(parents=True, exist_ok=True)

# -----------------------------
# 2. LOAD DATA
# -----------------------------

train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)

FEATURES = ["dow", "lag_1", "lag_7"]
TARGET = "required_nurses"

# -----------------------------
# 3. TRAIN MODELS PER SHIFT
# -----------------------------

results = []

for shift in ["M", "E", "N"]:
    print(f"\n🔹 Training models for shift: {shift}")

    train_s = train_df[train_df["shift"] == shift]
    test_s = test_df[test_df["shift"] == shift]

    X_train = train_s[FEATURES]
    y_train = train_s[TARGET]

    X_test = test_s[FEATURES]
    y_test = test_s[TARGET]

    # -------------------------
    # Linear Regression
    # -------------------------

    lr = LinearRegression()
    lr.fit(X_train, y_train)

    y_pred_lr = lr.predict(X_test)

    mae_lr = mean_absolute_error(y_test, y_pred_lr)
    rmse_lr = np.sqrt(mean_squared_error(y_test, y_pred_lr))

    joblib.dump(lr, f"{MODEL_DIR}/lr_shift_{shift}.pkl")

    # -------------------------
    # Random Forest
    # -------------------------

    rf = RandomForestRegressor(
        n_estimators=200,
        max_depth=5,
        random_state=42
    )
    rf.fit(X_train, y_train)

    y_pred_rf = rf.predict(X_test)

    mae_rf = mean_absolute_error(y_test, y_pred_rf)
    rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred_rf))

    joblib.dump(rf, f"{MODEL_DIR}/rf_shift_{shift}.pkl")

    # -------------------------
    # SAVE PREDICTIONS
    # -------------------------

    pred_df = test_s[["date", "shift", TARGET]].copy()
    pred_df["pred_lr"] = y_pred_lr
    pred_df["pred_rf"] = y_pred_rf

    pred_df.to_csv(
        f"{PRED_DIR}/predictions_shift_{shift}.csv",
        index=False
    )

    # -------------------------
    # COLLECT METRICS
    # -------------------------

    results.append({
        "shift": shift,
        "model": "LinearRegression",
        "MAE": mae_lr,
        "RMSE": rmse_lr
    })

    results.append({
        "shift": shift,
        "model": "RandomForest",
        "MAE": mae_rf,
        "RMSE": rmse_rf
    })

    print(f"LR → MAE: {mae_lr:.2f}, RMSE: {rmse_lr:.2f}")
    print(f"RF → MAE: {mae_rf:.2f}, RMSE: {rmse_rf:.2f}")

# -----------------------------
# 4. SAVE METRICS
# -----------------------------

metrics_df = pd.DataFrame(results)
metrics_df.to_csv("../../results/forecast_metrics.csv", index=False)

print("\n✅ Model training complete")
print("\n📊 Forecast accuracy summary:")
print(metrics_df)
