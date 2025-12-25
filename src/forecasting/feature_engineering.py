import pandas as pd
from pathlib import Path

# -----------------------------
# 1. PATHS
# -----------------------------

TRAIN_PATH = "../../data/processed/train.csv"
TEST_PATH = "../../data/processed/test.csv"

OUTPUT_TRAIN = "../../data/processed/train_fe.csv"
OUTPUT_TEST = "../../data/processed/test_fe.csv"

# -----------------------------
# 2. LOAD DATA
# -----------------------------

train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)

for df in [train_df, test_df]:
    df["date"] = pd.to_datetime(df["date"])

# -----------------------------
# 3. FEATURE ENGINEERING FUNCTION
# -----------------------------

def add_features(df):
    df = df.sort_values(["shift", "date"]).copy()

    # Day of week
    df["dow"] = df["date"].dt.dayofweek

    # Lag features (per shift)
    df["lag_1"] = df.groupby("shift")["required_nurses"].shift(1)
    df["lag_7"] = df.groupby("shift")["required_nurses"].shift(7)

    # Shift encoding
    shift_map = {"M": 0, "E": 1, "N": 2}
    df["shift_enc"] = df["shift"].map(shift_map)

    return df

# -----------------------------
# 4. APPLY FEATURES
# -----------------------------

train_fe = add_features(train_df)
test_fe = add_features(test_df)

# Drop rows with missing lags
train_fe = train_fe.dropna().reset_index(drop=True)
test_fe = test_fe.dropna().reset_index(drop=True)

# -----------------------------
# 5. SAVE
# -----------------------------

train_fe.to_csv(OUTPUT_TRAIN, index=False)
test_fe.to_csv(OUTPUT_TEST, index=False)

# -----------------------------
# 6. SANITY CHECKS
# -----------------------------

print("✅ Feature engineering complete\n")

print("📊 Feature columns:")
print(train_fe.columns.tolist())

print("\n🔎 Train feature preview:")
print(train_fe.head(5))

print("\n🔎 Test feature preview:")
print(test_fe.head(5))
