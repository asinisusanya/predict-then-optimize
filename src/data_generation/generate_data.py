import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import random

# -----------------------------
# 1. CONFIGURATION (ASSUMPTIONS)
# -----------------------------

START_DATE = "2024-01-01"
MONTHS = 12
DAYS_PER_MONTH = 30
CASUALTY_DAYS_PER_MONTH = 7
RANDOM_SEED = 42

SHIFTS = ["M", "E", "N"]

# Required nurse ranges
NON_CASUALTY_DEMAND = {
    "M": (6, 9),
    "E": (5, 8),
    "N": (3, 5)
}

CASUALTY_DEMAND = {
    "M": (9, 12),
    "E": (8, 11),
    "N": (5, 7)
}

NOISE_STD = 0.5  # small random noise

OUTPUT_PATH = "../../data/raw/synthetic_ground_truth.csv"

# -----------------------------
# 2. INITIALIZATION
# -----------------------------

np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

start_date = datetime.strptime(START_DATE, "%Y-%m-%d")
records = []

# -----------------------------
# 3. DATA GENERATION
# -----------------------------

current_date = start_date

for month in range(MONTHS):
    # Randomly choose casualty days for this month
    casualty_days = set(
        random.sample(range(DAYS_PER_MONTH), CASUALTY_DAYS_PER_MONTH)
    )

    for day in range(DAYS_PER_MONTH):
        is_casualty = day in casualty_days

        for shift in SHIFTS:
            if is_casualty:
                low, high = CASUALTY_DEMAND[shift]
            else:
                low, high = NON_CASUALTY_DEMAND[shift]

            base_demand = np.random.randint(low, high + 1)
            noise = np.random.normal(0, NOISE_STD)

            required_nurses = max(0, round(base_demand + noise))

            records.append({
                "date": current_date.strftime("%Y-%m-%d"),
                "shift": shift,
                "required_nurses": required_nurses
            })

        current_date += timedelta(days=1)

# -----------------------------
# 4. SAVE DATASET
# -----------------------------

df = pd.DataFrame(records)
df.to_csv(OUTPUT_PATH, index=False)

print("✅ Synthetic dataset generated successfully!")
print(f"📄 Saved to: {OUTPUT_PATH}")
print("\nPreview:")
print(df.head(9))
