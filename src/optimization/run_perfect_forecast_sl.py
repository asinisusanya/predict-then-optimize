import pandas as pd
import pulp

# -----------------------------
# 1. LOAD TRUE DEMAND
# -----------------------------

DATA_PATH = "../../data/raw/synthetic_ground_truth.csv"
df = pd.read_csv(DATA_PATH)

df["date"] = pd.to_datetime(df["date"])
df = df[df["date"] >= "2024-10-15"]

# -----------------------------
# 2. PARAMETERS
# -----------------------------

SERVICE_LEVEL = 0.95

SHIFT_CAPACITY = {
    "M": 12,
    "E": 10,
    "N": 7
}

WAGE_COST = {
    "M": 1.0,
    "E": 1.0,
    "N": 1.5
}

UNDERSTAFF_PENALTY = {
    "M": 3.0,
    "E": 3.0,
    "N": 5.0
}

# -----------------------------
# 3. OPTIMIZATION MODEL
# -----------------------------

model = pulp.LpProblem("Perfect_Forecast_Staffing_SL", pulp.LpMinimize)

x = {}
u = {}

for idx, row in df.iterrows():
    key = (row["date"], row["shift"])
    x[key] = pulp.LpVariable(
        f"x_{idx}",
        lowBound=0,
        upBound=SHIFT_CAPACITY[row["shift"]],
        cat="Integer"
    )
    u[key] = pulp.LpVariable(f"u_{idx}", lowBound=0)

# Objective
model += pulp.lpSum(
    WAGE_COST[row["shift"]] * x[(row["date"], row["shift"])] +
    UNDERSTAFF_PENALTY[row["shift"]] * u[(row["date"], row["shift"])]
    for _, row in df.iterrows()
)

# Constraints
for idx, row in df.iterrows():
    key = (row["date"], row["shift"])
    demand = row["required_nurses"]

    # Demand satisfaction
    model += x[key] + u[key] >= demand

    # Service level constraint (NEW)
    model += x[key] >= SERVICE_LEVEL * demand

# -----------------------------
# 4. SOLVE
# -----------------------------

model.solve(pulp.PULP_CBC_CMD(msg=0))

total_cost = pulp.value(model.objective)

print("✅ Perfect forecast (95% SL) optimization complete")
print(f"💰 Total cost (perfect forecast, SL): {total_cost:.2f}")
