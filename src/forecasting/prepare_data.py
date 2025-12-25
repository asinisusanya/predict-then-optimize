import pandas as pd
from pathlib import Path

# -----------------------------
# 1. PATHS
# -----------------------------

RAW_DATA_PATH = "../../data/raw/synthetic_ground_truth.csv"
PROCESSED_DIR = "../../data/processed"

Path(PROCESSED_DIR).mkdir(parents=True, exist_ok=True)

TRAIN_PATH = f"{PROCESSED_DIR}/train.csv"
TEST_PATH = f"{PROCESSED_DIR}/test.csv"

# -----------------------------
# 2. LOAD DATA
# -----------------------------

df = pd.read_csv(RAW_DATA_PATH)

# Ensure correct types
df["date"] = pd.to_datetime(df["date"])

# Sort by time (IMPORTANT)
df = df.sort_values(["date", "shift"]).reset_index(drop=True)

# -----------------------------
# 3. TRAIN / TEST SPLIT
# -----------------------------
# 80% train, 20% test (time-based)

split_ratio = 0.8
split_index = int(len(df) * split_ratio)

train_df = df.iloc[:split_index]
test_df = df.iloc[split_index:]

# -----------------------------
# 4. SAVE FILES
# -----------------------------

train_df.to_csv(TRAIN_PATH, index=False)
test_df.to_csv(TEST_PATH, index=False)

# -----------------------------
# 5. SANITY CHECKS
# -----------------------------

print("✅ Data preparation complete\n")

print("📊 Dataset sizes:")
print(f"Total records : {len(df)}")
print(f"Train records : {len(train_df)}")
print(f"Test records  : {len(test_df)}\n")

print("📅 Train date range:")
print(train_df['date'].min(), "→", train_df['date'].max())

print("\n📅 Test date range:")
print(test_df['date'].min(), "→", test_df['date'].max())

print("\n🔎 Train preview:")
print(train_df.head(6))

print("\n🔎 Test preview:")
print(test_df.head(6))
